# -*- coding: utf-8 -*-
"""
实验5：小型 UNet 重建 MNIST + 中间特征 + 前向分类学习（实验-分化v1.py / learn_v3 规则）

流程：
  阶段一（反向传播 / torch）：
      用一个小型 UNet 作为自编码器重建 MNIST 图像（28x28 -> 4x4 瓶颈 -> 28x28），
      用 MSE 损失 + Adam 反向传播训练，只学习"重建"，不使用任何标签。
  阶段二（前向学习 / 无反向传播）：
      取 UNet 编码器的中间特征（瓶颈层 C x 4 x 4），池化 + 归一化后拉平成特征向量 f，
      用 numpy 实现 实验-分化v1.py 中 MyNet.learn_v3 的规则，在线更新分类权重矩阵 W（10 x D）。
      全程只做前向计算 + 局部权重修改，不使用梯度。

用法示例：
    python 实验5.py                       # 完整实验（UNet 训练 + 前向分类，自动搜 alpha）
    python 实验5.py --unet-epochs 3 --cls-epochs 3 --alpha 0.01
    python 实验5.py --load-weights        # 跳过 UNet 训练，复用已保存权重
    python 实验5.py --smoke               # 极小数据量冒烟测试
"""
import os
import gzip
import struct
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')          # 默认不弹窗，图片统一保存到 figures/
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
# 0. 配置
# =============================================================================
class Cfg:
    seed = 0
    data_dir = 'data/MNIST/raw'

    # --- UNet 重建 ---
    unet_epochs = 8
    batch_size = 256
    unet_lr = 2e-3
    base_ch = 16               # 特征通道基数：16 -> [16, 32, 64]
    enc_depth = 2              # 下采样层数：2 -> 瓶颈 7x7（默认）；3 -> 瓶颈 4x4
    feature_stage = 'multi'    # bottleneck / multi（最深两层拼接，实测判别性更好）
    ckpt = None                # None -> 自动命名 unet_mnist_e{depth}_c{base}.pt
    recon_samples = 8          # 可视化重建的样本数

    # --- 前向分类学习 ---
    head = 1                   # 每个类别的子模板数（同 实验-分化v1.py 的 head）
    ksz = 8                    # 中间特征自适应池化到 ksz x ksz
    pool_mode = 'avg'          # 瓶颈特征池化方式：avg / max
    feat_norm = 'l2'           # 特征归一化：l2 / max / mean / none
    feat_scale = 1.0           # 归一化后再乘的幅度因子
    feat_clip = 0.0            # >0 时把归一化后的特征裁剪到 [0, feat_clip]
    row_norm = False           # True 时模板行保持单位模长（余弦打分）；实测会退化，默认关闭
    thr = 1.0                  # 规则内部响应阈值（对应源码 yp > 1）
    conf = 2.0                 # 确信阈值（对应源码 yp[idx] > 2）
    alpha = None               # None 表示自动搜索；给数值则直接使用
    alpha_list = (0.05, 0.1, 0.2, 0.4)
    alpha_tune_epochs = 1
    alpha_tune_num = 30000     # 自动搜 alpha 时使用的训练样本数
    cls_epochs = 6
    forget_period = 0          # >0 时每 N 个样本执行一次 forget()（0.99 衰减）
    cls_train_num = None       # None -> 全部 60000；否则只用前 N 个训练样本
    cls_test_num = None        # None -> 全部 10000
    log_bad = True             # 记录误分类样本（特征 + 权重），便于分析
    log_bad_dir = 'log/bad'

    figures_dir = 'figures'
    show = False               # True 时训练结束后 plt.show()


CFG = Cfg()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# 1. 数据读取（IDX 格式，兼容 .gz 与解压后的原始文件）
# =============================================================================
def read_idx(filename):
    """读取 MNIST 的 IDX 格式文件（自动判断是否 gzip 压缩）"""
    opener = gzip.open if filename.endswith('.gz') else open
    with opener(filename, 'rb') as f:
        magic_number = struct.unpack('>I', f.read(4))[0]
        if magic_number == 2051:            # 图像
            num_images = struct.unpack('>I', f.read(4))[0]
            num_rows = struct.unpack('>I', f.read(4))[0]
            num_cols = struct.unpack('>I', f.read(4))[0]
            data = np.frombuffer(f.read(), dtype=np.uint8)
            data = data.reshape(num_images, num_rows, num_cols)
        elif magic_number == 2049:          # 标签
            num_labels = struct.unpack('>I', f.read(4))[0]
            data = np.frombuffer(f.read(), dtype=np.uint8)
            assert data.size == num_labels, (data.size, num_labels)
        else:
            raise ValueError(f"未知的魔数: {magic_number}")
    return data


def load_mnist(data_dir=CFG.data_dir):
    """读取 MNIST，返回 float32 图像（[0,1]）与 int64 标签"""
    x_train = read_idx(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'))
    y_train = read_idx(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'))
    x_test = read_idx(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'))
    y_test = read_idx(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'))

    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)
    assert x_train.shape[0] == y_train.shape[0] == 60000
    assert x_test.shape[0] == y_test.shape[0] == 10000
    assert y_train.min() >= 0 and y_train.max() <= 9
    assert y_test.min() >= 0 and y_test.max() <= 9
    return (x_train, y_train), (x_test, y_test)


# =============================================================================
# 2. 小型 UNet（自编码器）：28x28 -> 14x14 -> 7x7 -> 4x4(瓶颈) -> 7x7 -> 14x14 -> 28x28
# =============================================================================
class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(cin, cout))

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """上采样 + 与编码器跳跃连接拼接 + 卷积"""
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = DoubleConv(cin + cskip, cout)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='nearest')
        return self.conv(torch.cat([skip, x], dim=1))


class TinyUNet(nn.Module):
    """按 enc_depth 决定下采样层数：
        enc_depth=3 -> 28->14->7->4，瓶颈 4x4（通道 base*4）
        enc_depth=2 -> 28->14->7，   瓶颈 7x7（通道 base*2），重建质量与特征分辨率更好
    """
    def __init__(self, cin=1, base=16, enc_depth=2):
        super().__init__()
        assert enc_depth in (2, 3)
        self.enc_depth = enc_depth
        chs = [base * (2 ** i) for i in range(enc_depth + 1)]   # [c0, c1, c2(, c3)]
        self.inc = DoubleConv(cin, chs[0])                      # 28x28
        self.downs = nn.ModuleList([Down(chs[i], chs[i + 1]) for i in range(enc_depth)])
        # 瓶颈先做一次尺度不变的精修，再逐级上采样并与对应的跳跃连接拼接
        self.mid = DoubleConv(chs[-1], chs[-1])
        # 每一级上采样都拼接对应尺度的编码特征（最深一级拼的是瓶颈前的卷积输出，
        # 通道数与瓶颈相同，构成"没有跳跃连接"的那一层）
        self.ups = nn.ModuleList()
        for j in range(enc_depth):
            self.ups.append(Up(chs[enc_depth - j], chs[enc_depth - 1 - j],
                               chs[enc_depth - 1 - j]))
        self.out = nn.Conv2d(chs[0], cin, 1)

    def encode(self, x):
        """返回瓶颈层中间特征 (B, C, h, w)"""
        h = self.inc(x)
        for d in self.downs:
            h = d(h)
        return h

    def forward(self, x):
        hs = [self.inc(x)]                      # 28x28
        for d in self.downs:
            hs.append(d(hs[-1]))                # 14x14 ... 最深层
        y = self.mid(hs[-1])
        # 第 j 级上采样：拼接编码器同尺度特征 hs[-j-2]（j=0 时拼的是瓶颈输入，与瓶颈同尺度）
        for j, up in enumerate(self.ups):
            y = up(y, hs[-2 - j])
        return self.out(y)


# =============================================================================
# 3. 阶段一：UNet 重建训练（反向传播）
# =============================================================================
def build_model(cfg=CFG):
    return TinyUNet(1, cfg.base_ch, cfg.enc_depth)


def train_unet(x_train, device, cfg=CFG):
    model = build_model(cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f'[UNet] 参数量: {n_par:,}')

    x = torch.from_numpy(x_train[:, None, :, :])          # (N,1,28,28)
    ds = torch.utils.data.TensorDataset(x)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                                     num_workers=0, drop_last=True)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.unet_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.unet_epochs, 1))
    crit = nn.MSELoss()

    hist = []
    t0 = time.time()
    for ep in range(cfg.unet_epochs):
        model.train()
        tot, nb = 0.0, 0
        for (xb,) in dl:
            xb = xb.to(device)
            out = model(xb)
            loss = crit(out, xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()
        hist.append(tot / max(nb, 1))
        print(f'[UNet] epoch {ep + 1}/{cfg.unet_epochs}  mse={hist[-1]:.6f}  '
              f'({time.time() - t0:.1f}s)')

    if cfg.ckpt:
        torch.save(model.state_dict(), cfg.ckpt)
        print(f'[UNet] 权重已保存: {cfg.ckpt}')
    return model, hist


@torch.no_grad()
def unet_reconstruct(model, x, device, batch=512):
    """返回重建图像 (N,28,28)，越界值裁剪到 [0,1]"""
    model.eval()
    out = []
    for i in range(0, len(x), batch):
        xb = torch.from_numpy(x[i:i + batch, None, :, :]).to(device)
        yb = model(xb).clamp(0, 1)
        out.append(yb[:, 0].cpu().numpy())
    return np.concatenate(out, 0)


def recon_metrics(x, xr):
    mse = float(np.mean((x - xr) ** 2))
    psnr = float(10 * np.log10(1.0 / max(mse, 1e-12)))
    return mse, psnr


def save_recon_figure(x, xr, y, path, n=8):
    n = min(n, len(x))
    fig, axes = plt.subplots(2, n, figsize=(1.4 * n, 3.2))
    for i in range(n):
        axes[0, i].imshow(x[i], cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'{y[i]}', fontsize=9)
        axes[0, i].axis('off')
        axes[1, i].imshow(xr[i], cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel('原图')
    axes[1, 0].set_ylabel('重建')
    fig.suptitle('UNet 重建（上：原图，下：重建）')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f'[fig] {path}')


# =============================================================================
# 4. 阶段二：中间特征提取
# =============================================================================
@torch.no_grad()
def extract_features(model, x, device, cfg=CFG, batch=512, stage='bottleneck'):
    """取 UNet 多层中间特征 -> 池化到 ksz x ksz -> 展平 (N, ksz*ksz*C)（不含归一化）

    stage='bottleneck'：只用编码器最深层
    stage='multi'    ：编码器最深两层拼接（更细的空间分辨率 + 更抽象的语义）
    """
    model.eval()
    feats = []
    for i in range(0, len(x), batch):
        xb = torch.from_numpy(x[i:i + batch, None, :, :]).to(device)
        hs = []
        h = model.inc(xb)
        for j, d in enumerate(model.downs):
            h = d(h)
            if j >= model.enc_depth - (2 if stage == 'multi' else 1):
                hs.append(h)
        parts = []
        for z in hs:
            if cfg.pool_mode == 'max':
                z = F.adaptive_max_pool2d(z, (cfg.ksz, cfg.ksz))
            else:
                z = F.adaptive_avg_pool2d(z, (cfg.ksz, cfg.ksz))
            parts.append(z)
        z = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        feats.append(z.cpu().numpy().astype(np.float32))
    f = np.concatenate(feats, 0)                                # (N,C,ksz,ksz)
    f = f.transpose(0, 2, 3, 1).reshape(len(f), -1)              # (N, ksz*ksz*C)
    return f


def normalize_features(f, mode=CFG.feat_norm, scale=CFG.feat_scale, clip=CFG.feat_clip):
    """逐样本归一化，把特征幅度对齐到 learn_v3 规则设定的阈值量级（1 / 2）

    mean：除以该样本特征的均值（非负 ReLU 特征 -> 归一化后均值为 1，
          与模板对齐时点积约为 1，正好落在规则的阈值附近，量纲最合适）
    l2  ：除以 L2 范数（幅度 <1，需要更大的 scale/alpha）
    max ：除以最大值
    none：不归一化
    """
    if mode == 'mean':
        n = f.reshape(len(f), -1).mean(axis=1, keepdims=True)
    elif mode == 'l2':
        n = np.linalg.norm(f, axis=1, keepdims=True)
    elif mode == 'max':
        n = np.abs(f).max(axis=1, keepdims=True)
    else:
        n = None
    if n is not None:
        f = f / np.maximum(n, 1e-6)
    if scale != 1.0:
        f = f * scale
    if clip and clip > 0:
        f = np.clip(f, 0, clip)
    return f.astype(np.float32)


# =============================================================================
# 5. learn_v3：前向分类学习规则（照搬 实验-分化v1.py 的 MyNet.learn_v3）
# =============================================================================
class ForwardClassifier:
    """W: (10*head, D)，每个类别 head 个子模板；在线前向更新，无反向传播

    与 实验-分化v1.py 中 MyNet.learn_v3 一致的决策逻辑：
        yp = W f,  idx = argmax(yp),  n = #{k: yp[k] > thr}
        - n==1 且 yp[idx] > conf 且预测正确  -> 不学习（已足够确信）
        - n==0 或 n==1                       -> 只更新冠军行：预测对 +alpha*f，错 -alpha*f
        - 其余（多个响应 >thr）              -> 响应 >thr 的错类别行 -alpha*f，
                                                正确类别的冠军行 +alpha*f

    这里保留原始规则的内积打分与不限制权重幅度（见实验报告：实测逐行归一化/全局归一化
    都会让打分退化成"少数模板通吃"，反而更差；内积+权重自然增长是自稳定的）。
    row_norm=True 时改为余弦打分，作为对照实验保留。
    """

    def __init__(self, dim, n_class=10, head=1, seed=0, alpha=0.01,
                 row_norm=False, thr=1.0, conf=2.0):
        self.dim = dim
        self.n_class = n_class
        self.head = head
        self.alpha = alpha
        self.row_norm = row_norm
        self.THR = thr          # 规则内部的"响应阈值"（对应源码 yp > 1）
        self.CONF = conf        # 确信阈值（对应源码 yp[idx] > 2）
        self.maxval = np.inf    # 同源码：不限制权重幅度
        rng = np.random.default_rng(seed)
        W = rng.standard_normal((n_class * head, dim)).astype(np.float32)
        if row_norm:            # 初始也保持单位模长，避免起步阶段模长不均衡
            W /= np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-9)
        else:
            W *= 0.01
        self.W = W
        self.n_update = 0       # 权重更新次数（学习活跃度统计）

    # ---- 打分 / 预测 ----
    def _scores(self, F):
        if self.row_norm:
            Wn = self.W / np.maximum(np.linalg.norm(self.W, axis=1, keepdims=True), 1e-9)
            return Wn, F.dot(Wn.T)
        return self.W, F.dot(self.W.T)

    def scores(self, f):
        return self._scores(f[None, :])[1][0]

    def predict(self, f):
        _, s = self._scores(f[None, :])
        return int(np.argmax(s[0])) // self.head

    def predict_batch(self, F):
        _, s = self._scores(F)
        return np.argmax(s, axis=1) // self.head, s

    # ---- 学习（核心规则）----
    def learn_v3(self, f, y):
        Wn, s = self._scores(f[None, :])
        yp = s[0]
        idx = int(np.argmax(yp))

        n = int(np.sum(yp > self.THR))
        if n == 1 and yp[idx] > self.CONF and idx == y:
            return 0
        if n == 0 or n == 1:
            flag = 1 if idx // self.head == y else -1
            self.W[idx] = np.clip(self.W[idx] + flag * self.alpha * f,
                                  -self.maxval, self.maxval)
            self.n_update += 1
            return 1

        cnt = 0
        for k in range(self.n_class * self.head):
            if yp[k] <= self.THR:
                continue
            if k // self.head == y:
                if k == idx:                       # 正确类别的冠军行：加强
                    self.W[k] = np.clip(self.W[k] + self.alpha * f,
                                        -self.maxval, self.maxval)
                    cnt += 1
            else:                                  # 其它已激活类别：削弱
                self.W[k] = np.clip(self.W[k] - self.alpha * f,
                                    -self.maxval, self.maxval)
                cnt += 1
        self.n_update += cnt
        return cnt

    def forget(self, decay=0.99):
        """同 实验-分化v1.py 的 forget：整体衰减权重"""
        self.W *= decay


def classify_train(clf, F, Y, epochs, prefix=''):
    """用 learn_v3 在线训练分类器，返回每 epoch 的在线训练准确率"""
    hist = []
    for ep in range(epochs):
        t0 = time.time()
        clf.n_update = 0
        correct = 0
        for i, f in enumerate(F):
            if clf.predict(f) == Y[i]:
                correct += 1
            clf.learn_v3(f, int(Y[i]))
        acc = correct / len(F)
        hist.append(acc)
        print(f'{prefix}epoch {ep + 1}/{epochs}  在线训练准确率={acc:.4f}  '
              f'权重更新={clf.n_update} 次  ({time.time() - t0:.1f}s)')
    return hist


def classify_eval(clf, F, Y):
    """评估：整体准确率 + 每类统计 + 混淆矩阵"""
    pred, _ = clf.predict_batch(F)
    acc = float(np.mean(pred == Y))
    stat = np.zeros((10, 2), dtype=np.int64)
    conf = np.zeros((10, 10), dtype=np.int64)
    for t, p in zip(Y, pred):
        stat[t, 0] += 1
        stat[t, 1] += int(t == p)
        conf[t, p] += 1
    return acc, stat, conf


# =============================================================================
# 6. 可视化
# =============================================================================
def save_weight_figure(clf, path, cfg=CFG):
    """把学到的分类模板逐类画出来（reshape 成 ksz x ksz x C）"""
    C = clf.dim // (cfg.ksz * cfg.ksz)
    n_class = clf.n_class
    fig, axes = plt.subplots(2, n_class, figsize=(1.5 * n_class, 3.4))
    for cls in range(n_class):
        w = clf.W[cls * clf.head]                       # head=1 时就是该类模板
        im = w.reshape(cfg.ksz, cfg.ksz, C)
        axes[0, cls].imshow(im.mean(axis=2), cmap='bwr')
        axes[0, cls].set_title(f'{cls}', fontsize=9)
        axes[0, cls].axis('off')
        axes[1, cls].imshow(im.reshape(cfg.ksz, -1), cmap='bwr', aspect='auto')
        axes[1, cls].axis('off')
    fig.suptitle('前向学习得到的分类权重模板（上：通道均值，下：全部通道平铺）')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f'[fig] {path}')


def save_curve_figure(curves, path, title, xlabel='epoch'):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, vals in curves.items():
        ax.plot(range(1, len(vals) + 1), vals, marker='o', label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('accuracy')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f'[fig] {path}')


def save_bad_samples(model, clf, x, y, path, cfg=CFG, max_show=10):
    """记录误分类样本：原图 | 重建 | 中间特征（通道均值） | 预测类模板 | 真值类模板"""
    F_all = normalize_features(
        extract_features(model, x, DEVICE, cfg, stage=cfg.feature_stage), cfg.feat_norm)
    pred, s = clf.predict_batch(F_all)
    bad = np.where(pred != y)[0]
    if len(bad) == 0:
        print('[log] 测试集中没有误分类样本')
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    show = bad[:max_show]
    C = F_all.shape[1] // (cfg.ksz * cfg.ksz)
    fig, axes = plt.subplots(len(show), 5, figsize=(9, 1.8 * len(show)))
    if len(show) == 1:
        axes = axes[None, :]
    for r, i in enumerate(show):
        xr = unet_reconstruct(model, x[i:i + 1], DEVICE)[0]
        fimg = F_all[i].reshape(cfg.ksz, cfg.ksz, C).mean(axis=2)
        wp = clf.W[pred[i] * clf.head].reshape(cfg.ksz, cfg.ksz, C).mean(axis=2)
        wy = clf.W[y[i] * clf.head].reshape(cfg.ksz, cfg.ksz, C).mean(axis=2)
        for c, (img, ttl) in enumerate([
                (x[i], f'真值 {y[i]}'),
                (xr, '重建'),
                (fimg, f'特征 argmax {pred[i]}'),
                (wp, f'W[预测={pred[i]}]'),
                (wy, f'W[真值={y[i]}]')]):
            axes[r, c].imshow(img, cmap='gray' if c < 2 else 'bwr')
            if r == 0:
                axes[r, c].set_title(ttl, fontsize=9)
            axes[r, c].axis('off')
        axes[r, 0].set_ylabel(f's={s[i][pred[i]]:.2f}')
    fig.suptitle('误分类样本分析（前 %d 个）' % len(show))
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f'[fig] {path}   误分类 {len(bad)}/{len(y)}')


# =============================================================================
# 7. 主流程
# =============================================================================
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description='实验5：UNet 重建 + 中间特征前向分类学习')
    p.add_argument('--unet-epochs', type=int, default=CFG.unet_epochs)
    p.add_argument('--cls-epochs', type=int, default=CFG.cls_epochs)
    p.add_argument('--batch-size', type=int, default=CFG.batch_size)
    p.add_argument('--lr', type=float, default=CFG.unet_lr)
    p.add_argument('--base-ch', type=int, default=CFG.base_ch)
    p.add_argument('--enc-depth', type=int, default=CFG.enc_depth, choices=[2, 3],
                   help='UNet 下采样层数：2 -> 瓶颈 7x7；3 -> 瓶颈 4x4')
    p.add_argument('--stage', type=str, default=CFG.feature_stage,
                   choices=['bottleneck', 'multi'], help='使用哪层中间特征')
    p.add_argument('--ksz', type=int, default=CFG.ksz)
    p.add_argument('--head', type=int, default=CFG.head)
    p.add_argument('--pool', type=str, default=CFG.pool_mode, choices=['avg', 'max'])
    p.add_argument('--feat-norm', type=str, default=CFG.feat_norm,
                   choices=['mean', 'l2', 'max', 'none'])
    p.add_argument('--feat-scale', type=float, default=CFG.feat_scale)
    p.add_argument('--feat-clip', type=float, default=CFG.feat_clip)
    p.add_argument('--row-norm', dest='row_norm', action='store_true',
                   help='模板行保持单位模长（余弦打分）；默认关闭，实测会退化')
    p.add_argument('--no-row-norm', dest='row_norm', action='store_false',
                   help='使用原始内积打分（默认）')
    p.set_defaults(row_norm=CFG.row_norm)
    p.add_argument('--thr', type=float, default=CFG.thr, help='规则响应阈值')
    p.add_argument('--conf', type=float, default=CFG.conf, help='确信阈值')
    p.add_argument('--alpha', type=float, default=CFG.alpha,
                   help='前向学习步长；不指定则自动搜索')
    p.add_argument('--alpha-list', type=float, nargs='+', default=None,
                   help='自动搜索使用的步长候选列表')
    p.add_argument('--alpha-tune-num', type=int, default=CFG.alpha_tune_num)
    p.add_argument('--forget-period', type=int, default=CFG.forget_period)
    p.add_argument('--train-num', type=int, default=CFG.cls_train_num,
                   help='前向分类训练样本数，默认全部')
    p.add_argument('--test-num', type=int, default=CFG.cls_test_num)
    p.add_argument('--load-weights', action='store_true', help='跳过 UNet 训练，直接加载权重')
    p.add_argument('--no-log-bad', action='store_true')
    p.add_argument('--show', action='store_true', help='结束后弹窗显示图片')
    p.add_argument('--smoke', action='store_true', help='冒烟测试：极小数据量快速跑通')
    return p.parse_args()


def main():
    args = parse_args()
    CFG.unet_epochs = args.unet_epochs
    CFG.cls_epochs = args.cls_epochs
    CFG.batch_size = args.batch_size
    CFG.unet_lr = args.lr
    CFG.base_ch = args.base_ch
    CFG.enc_depth = args.enc_depth
    CFG.feature_stage = args.stage
    CFG.ksz = args.ksz
    CFG.head = args.head
    CFG.pool_mode = args.pool
    CFG.feat_norm = args.feat_norm
    CFG.feat_scale = args.feat_scale
    CFG.feat_clip = args.feat_clip
    CFG.row_norm = args.row_norm
    CFG.thr = args.thr
    CFG.conf = args.conf
    if CFG.ckpt is None:
        CFG.ckpt = f'unet_mnist_e{CFG.enc_depth}_c{CFG.base_ch}.pt'
    CFG.alpha = args.alpha
    if args.alpha_list:
        CFG.alpha_list = tuple(args.alpha_list)
    CFG.alpha_tune_num = args.alpha_tune_num
    CFG.forget_period = args.forget_period
    CFG.cls_train_num = args.train_num
    CFG.cls_test_num = args.test_num
    CFG.log_bad = not args.no_log_bad
    CFG.show = args.show
    if args.smoke:
        CFG.unet_epochs, CFG.cls_epochs = 1, 1
        CFG.alpha_tune_num = 2000
        CFG.cls_train_num, CFG.cls_test_num = 2000, 1000
        CFG.recon_samples = 6

    os.makedirs(CFG.figures_dir, exist_ok=True)
    set_seed(CFG.seed)

    print('=' * 78)
    print(f'设备: {DEVICE}   UNet: base_ch={CFG.base_ch} enc_depth={CFG.enc_depth} '
          f'stage={CFG.feature_stage}')
    print(f'前向分类: ksz={CFG.ksz} head={CFG.head} pool={CFG.pool_mode} '
          f'feat_norm={CFG.feat_norm} row_norm={CFG.row_norm} thr={CFG.thr} conf={CFG.conf}')
    print('=' * 78)

    # ---------------- 数据 ----------------
    (x_train, y_train), (x_test, y_test) = load_mnist()
    print(f'训练图像 {x_train.shape}  测试图像 {x_test.shape}')

    # ---------------- 阶段一：UNet 重建（反向传播） ----------------
    print('\n' + '-' * 78)
    print('阶段一：小 UNet 重建 MNIST（torch + 反向传播，不使用标签）')
    print('-' * 78)
    if args.load_weights and os.path.exists(CFG.ckpt):
        model = build_model().to(DEVICE)
        model.load_state_dict(torch.load(CFG.ckpt, map_location=DEVICE))
        rec_hist = []
        print(f'[UNet] 已加载权重 {CFG.ckpt}')
    else:
        model, rec_hist = train_unet(x_train, DEVICE)

    xr_test = unet_reconstruct(model, x_test[:2000], DEVICE)
    mse, psnr = recon_metrics(x_test[:2000], xr_test)
    print(f'[UNet] 测试集重建 MSE={mse:.6f}  PSNR={psnr:.2f} dB')
    if rec_hist:
        save_curve_figure({'recon MSE': rec_hist},
                          os.path.join(CFG.figures_dir, 'unet_recon_loss.png'),
                          'UNet 重建损失（反向传播训练）', xlabel='epoch')
    save_recon_figure(x_test, xr_test, y_test,
                      os.path.join(CFG.figures_dir, 'unet_recon_samples.png'),
                      CFG.recon_samples)

    # ---------------- 中间特征 ----------------
    print('\n' + '-' * 78)
    print(f'阶段二：取 UNet 中间特征（{CFG.feature_stage}）刻画样本，用前向规则学习分类')
    print('-' * 78)
    t0 = time.time()
    F_train = normalize_features(
        extract_features(model, x_train, DEVICE, stage=CFG.feature_stage), CFG.feat_norm)
    F_test = normalize_features(
        extract_features(model, x_test, DEVICE, stage=CFG.feature_stage), CFG.feat_norm)
    print(f'[feat] 特征维度 {F_train.shape[1]}  '
          f'(C={F_train.shape[1] // CFG.ksz ** 2} x {CFG.ksz}x{CFG.ksz})  '
          f'提取耗时 {time.time() - t0:.1f}s')
    print(f'[feat] 取值范围 [{F_train.min():.3f}, {F_train.max():.3f}]  '
          f'均值 {F_train.mean():.4f}  逐样本 L2 范数均值 '
          f'{np.linalg.norm(F_train[:2000], axis=1).mean():.4f}')

    if CFG.cls_train_num:
        F_tr, Y_tr = F_train[:CFG.cls_train_num], y_train[:CFG.cls_train_num]
    else:
        F_tr, Y_tr = F_train, y_train
    if CFG.cls_test_num:
        F_te, Y_te = F_test[:CFG.cls_test_num], y_test[:CFG.cls_test_num]
    else:
        F_te, Y_te = F_test, y_test
    print(f'[feat] 前向分类训练样本 {len(F_tr)}  测试样本 {len(F_te)}')

    # ---------------- 参考上界：最近类均值（不是前向学习，只用来判断特征好坏） ----------------
    nc_tr = F_tr / np.maximum(np.linalg.norm(F_tr, axis=1, keepdims=True), 1e-9)
    nc_te = F_te / np.maximum(np.linalg.norm(F_te, axis=1, keepdims=True), 1e-9)
    C = np.stack([nc_tr[Y_tr == c].mean(0) for c in range(10)])
    C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-9)
    nc_acc = float(np.mean(np.argmax(nc_te.dot(C.T), 1) == Y_te))
    print(f'[ref] 最近类均值（nearest centroid，特征可分性上界参考）= {nc_acc:.4f}')

    # ---------------- alpha 自动搜索 ----------------
    alphas = [CFG.alpha] if CFG.alpha is not None else list(CFG.alpha_list)
    if CFG.alpha is None:
        n_tune = min(CFG.alpha_tune_num, len(F_tr))
        print(f'\n[alpha] 在 {len(alphas)} 个步长上试探（{n_tune} 个样本，'
              f'{CFG.alpha_tune_epochs} 轮在线学习）:')
        best_alpha, best_acc, tune_rec = None, -1.0, {}
        for a in alphas:
            clf = ForwardClassifier(F_tr.shape[1], 10, CFG.head, seed=CFG.seed, alpha=a,
                                    row_norm=CFG.row_norm, thr=CFG.thr, conf=CFG.conf)
            classify_train(clf, F_tr[:n_tune], Y_tr[:n_tune], CFG.alpha_tune_epochs,
                           prefix=f'  alpha={a:<6} ')
            acc, _, _ = classify_eval(clf, F_te, Y_te)
            tune_rec[a] = acc
            print(f'  alpha={a:<6} 测试准确率={acc:.4f}')
            if acc > best_acc:
                best_alpha, best_acc = a, acc
        CFG.alpha = best_alpha
        print(f'[alpha] 选用 alpha={CFG.alpha}（试算准确率 {best_acc:.4f}）')
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.semilogx(alphas, [tune_rec[a] for a in alphas], marker='o')
        for a in alphas:
            ax.annotate(f'{tune_rec[a]:.3f}', (a, tune_rec[a]),
                        textcoords='offset points', xytext=(0, 8), fontsize=8,
                        ha='center')
        ax.set_xlabel('alpha')
        ax.set_ylabel('测试准确率')
        ax.set_title(f'前向学习步长的试探结果（{n_tune} 个样本，'
                     f'{CFG.alpha_tune_epochs} 轮在线学习）')
        ax.grid(alpha=0.3, which='both')
        fig.tight_layout()
        fig.savefig(os.path.join(CFG.figures_dir, 'alpha_tune.png'), dpi=120)
        plt.close(fig)
        print(f'[fig] {os.path.join(CFG.figures_dir, "alpha_tune.png")}')

    # ---------------- 前向分类学习（learn_v3） ----------------
    print(f'\n[cls] learn_v3 规则，alpha={CFG.alpha}，D={F_tr.shape[1]}，'
          f'W 形状={10 * CFG.head}x{F_tr.shape[1]}，row_norm={CFG.row_norm}')
    clf = ForwardClassifier(F_tr.shape[1], 10, CFG.head, seed=CFG.seed, alpha=CFG.alpha,
                            row_norm=CFG.row_norm, thr=CFG.thr, conf=CFG.conf)
    train_acc_hist, test_acc_hist = [], []
    for ep in range(CFG.cls_epochs):
        clf.n_update = 0
        t0 = time.time()
        correct = 0
        for i, f in enumerate(F_tr):
            if clf.predict(f) == Y_tr[i]:
                correct += 1
            clf.learn_v3(f, int(Y_tr[i]))
            if CFG.forget_period and (i + 1) % CFG.forget_period == 0:
                clf.forget()
        train_acc = correct / len(F_tr)
        acc, stat, conf_mat = classify_eval(clf, F_te, Y_te)
        train_acc_hist.append(train_acc)
        test_acc_hist.append(acc)
        pred_tr, _ = clf.predict_batch(F_tr[:5000])
        dist = np.bincount(pred_tr, minlength=10)
        print(f'[cls] epoch {ep + 1}/{CFG.cls_epochs}  在线训练准确率={train_acc:.4f}  '
              f'测试准确率={acc:.4f}  权重更新={clf.n_update} 次  '
              f'|W|max={np.linalg.norm(clf.W, axis=1).max():.1f}  '
              f'预测分布={dist.tolist()}  ({time.time() - t0:.1f}s)')

    acc, stat, conf_mat = classify_eval(clf, F_te, Y_te)
    print('\n[cls] 各类别测试结果（前向学习，无反向传播）:')
    for i in range(10):
        print(f'  {i:2d}  {stat[i, 1]:5d}/{stat[i, 0]:5d}   '
              f'{stat[i, 1] / max(stat[i, 0], 1):.4f}')
    print(f'  总计 {stat[:, 1].sum()}/{stat[:, 0].sum()}  准确率={acc:.4f}')

    np.save('cls_W.npy', clf.W)
    np.save('cls_confusion.npy', conf_mat)
    save_curve_figure({'在线训练准确率': train_acc_hist, '测试准确率': test_acc_hist},
                      os.path.join(CFG.figures_dir, 'forward_cls_accuracy.png'),
                      f'前向分类学习（learn_v3, alpha={CFG.alpha}）准确率')
    save_weight_figure(clf, os.path.join(CFG.figures_dir, 'forward_cls_weights.png'))
    if CFG.log_bad:
        save_bad_samples(model, clf, x_test, y_test,
                         os.path.join(CFG.log_bad_dir, 'bad_samples.png'))

    print(f'\n完成。图片见 {CFG.figures_dir}/，分类权重 cls_W.npy，UNet 权重 {CFG.ckpt}')
    print(f'重建 MSE={mse:.6f} (PSNR {psnr:.2f} dB)   前向分类测试准确率={acc:.4f}')

    if CFG.show:
        plt.show()


if __name__ == '__main__':
    main()
