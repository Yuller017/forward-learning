#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最终方案：bbox + MAE/SimCLR 特征融合 + 优化后的前向规则

这是整个项目的最终管线（实验22 的最优组合），自包含、可直接运行：
  1. bbox：墨迹紧包围框 → 拉伸到 28×28
  2. 特征 A：MAE（三层 UNet，随机遮 50% 的 4×4 patch，重建整图）→ h2（32×14×14=6272）
  3. 特征 B：SimCLR（64/128/256 卷积编码器 + GAP → 256 维；仿射增强 + NT-Xent）
  4. 融合：两块各自 L2 归一化后拼接（6528 维）
  5. 分类：优化后的 learn_v3 规则（margin 门控 + 归一化步长 + epoch 权重平均），
     逐样本、无反向传播、每次只改两行权重
  6. 评估：测试集准确率（同时给出 MAE 单特征 / SimCLR 单特征作为对照）

预期结果（2 个初始化种子）：
  MAE h2        98.55%
  SimCLR        98.94%
  拼接 + 规则    99.26%

用法：
  python 最终方案.py                 # 完整（约 3–4 分钟，GPU）
  python 最终方案.py --quick         # 冒烟（小数据、少 epoch）
"""
import argparse
import gzip
import json
import os
import struct
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(ROOT, '..', 'data', 'MNIST', 'raw'))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- 优化后的前向规则超参（实验16 在验证集上选定） ----
MARGIN = 0.2
ALPHA = 0.2
RULE_EPOCHS = 8


# =============================================================================
# 0. 工具
# =============================================================================
class Tee:
    def __init__(self, path, stream):
        self.stream = stream
        self.file = open(path, 'w', encoding='utf-8')

    def write(self, s):
        self.stream.write(s)
        self.file.write(s)
        self.file.flush()
        return len(s)

    def flush(self):
        self.stream.flush()
        self.file.flush()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# =============================================================================
# 1. 数据：MNIST → bbox 28×28
# =============================================================================
def read_idx(filename):
    opener = gzip.open if filename.endswith('.gz') else open
    with opener(filename, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        if magic == 2051:
            n = struct.unpack('>I', f.read(4))[0]
            rows = struct.unpack('>I', f.read(4))[0]
            cols = struct.unpack('>I', f.read(4))[0]
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)
        if magic == 2049:
            n = struct.unpack('>I', f.read(4))[0]
            data = np.frombuffer(f.read(), dtype=np.uint8)
            assert data.size == n
            return data
        raise ValueError(f'未知的魔数: {magic}')


def load_mnist(data_dir=DATA_DIR):
    def p(name):
        return os.path.join(data_dir, name)

    xtr = read_idx(p('train-images-idx3-ubyte.gz')).astype(np.float32) / 255.0
    ytr = read_idx(p('train-labels-idx1-ubyte.gz')).astype(np.int64)
    xte = read_idx(p('t10k-images-idx3-ubyte.gz')).astype(np.float32) / 255.0
    yte = read_idx(p('t10k-labels-idx1-ubyte.gz')).astype(np.int64)
    return (xtr, ytr), (xte, yte)


def bbox_normalize(img, out=28):
    """墨迹紧包围框 → 拉伸到 out×out"""
    ys, xs = np.nonzero(img > 0.0)
    if len(ys) == 0:
        return np.zeros((out, out), np.float32)
    crop = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return cv2.resize(crop, (out, out), interpolation=cv2.INTER_LINEAR)


def build_bbox(images, out=28, desc=''):
    X = np.zeros((len(images), out, out), np.float32)
    for i, img in enumerate(images):
        X[i] = bbox_normalize(img, out)
    print(f'[prep] bbox {desc}: {len(images)} 张')
    return X


# =============================================================================
# 2. 特征 A：MAE（三层 UNet，遮挡重建）
# =============================================================================
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class TinyUNet3(nn.Module):
    """28→14→7→14→28；h1=16×28×28，h2=32×14×14，h3=64×7×7"""

    def __init__(self, base=16):
        super().__init__()
        self.e1 = ConvBlock(1, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.d2 = ConvBlock(base * 2 + base * 2, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.d1 = ConvBlock(base + base, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x, return_feats=False):
        h1 = self.e1(x)
        h2 = self.e2(F.max_pool2d(h1, 2))
        h3 = self.e3(F.max_pool2d(h2, 2))
        d2 = self.d2(torch.cat([self.u2(h3), h2], dim=1))
        d1 = self.d1(torch.cat([self.u1(d2), h1], dim=1))
        y = torch.sigmoid(self.out(d1))
        return (y, (h1, h2)) if return_feats else y


def train_mae(X, args, seed=0, mask_ratio=0.5):
    torch.manual_seed(seed)
    model = TinyUNet3(base=args.base).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(seed)
    Xt = torch.from_numpy(X).unsqueeze(1).to(DEVICE)
    n = len(X)
    for ep in range(args.mae_epochs):
        model.train()
        perm = rng.permutation(n)
        tot, cnt = 0.0, 0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            xb = Xt[idx]
            m = (torch.rand(len(idx), 7, 7, device=DEVICE) < mask_ratio)
            m = m.repeat_interleave(4, 1).repeat_interleave(4, 2)
            loss = F.mse_loss(model(xb * (~m).unsqueeze(1).float()), xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.detach().item() * len(idx)
            cnt += len(idx)
        print(f'[MAE] epoch {ep + 1}/{args.mae_epochs}  MSE={tot / cnt:.6f}')
    return model


@torch.no_grad()
def mae_h2(model, X, batch=512):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).unsqueeze(1).to(DEVICE)
        _, (_, h2) = model(xb, return_feats=True)
        out.append(h2.cpu().numpy())
    return np.concatenate(out, axis=0).reshape(len(X), -1)


# =============================================================================
# 3. 特征 B：SimCLR（实例判别）
# =============================================================================
class SimCLR(nn.Module):
    def __init__(self, feat=256):
        super().__init__()
        w = max(32, feat // 4)
        self.enc = nn.Sequential(
            ConvBlock(1, w), nn.MaxPool2d(2),
            ConvBlock(w, 2 * w), nn.MaxPool2d(2),
            ConvBlock(2 * w, 4 * w),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.proj = nn.Sequential(nn.Linear(4 * w, 128), nn.ReLU(),
                                  nn.Linear(128, feat))

    def forward(self, x):
        h = self.enc(x)
        return h, self.proj(h)


def affine_augment(Xt, rng):
    B = len(Xt)
    sh = rng.uniform(-0.2, 0.2, B).astype(np.float32)
    sc = rng.uniform(0.9, 1.1, B).astype(np.float32)
    th = np.deg2rad(rng.uniform(-12, 12, B).astype(np.float32))
    tx = rng.uniform(-0.15, 0.15, B).astype(np.float32)
    ty = rng.uniform(-0.15, 0.15, B).astype(np.float32)
    c, s = np.cos(th), np.sin(th)
    M = np.zeros((B, 2, 3), np.float32)
    M[:, 0, 0] = c * sc
    M[:, 0, 1] = c * sc * sh - s * sc
    M[:, 1, 0] = s * sc
    M[:, 1, 1] = s * sc * sh + c * sc
    M[:, 0, 2] = tx
    M[:, 1, 2] = ty
    grid = F.affine_grid(torch.from_numpy(M).to(Xt.device), Xt.shape,
                         align_corners=False)
    return F.grid_sample(Xt, grid, align_corners=False, padding_mode='zeros',
                         mode='bilinear')


def nt_xent(z1, z2, tau=0.5):
    z = F.normalize(torch.cat([z1, z2], 0), dim=1)
    sim = z @ z.T / tau
    n = len(z1)
    sim.masked_fill_(torch.eye(2 * n, device=z.device, dtype=torch.bool), -1e9)
    labels = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, labels)


def train_simclr(X, args, seed=0):
    torch.manual_seed(seed)
    model = SimCLR(feat=args.sim_feat).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xt = torch.from_numpy(X).unsqueeze(1).to(DEVICE)
    n = len(X)
    rng = np.random.default_rng(seed)
    for ep in range(args.sim_epochs):
        model.train()
        perm = rng.permutation(n)
        tot, cnt = 0.0, 0
        for i in range(0, n, args.sim_batch):
            idx = perm[i:i + args.sim_batch]
            xb = Xt[idx]
            _, z1 = model(affine_augment(xb, rng))
            _, z2 = model(affine_augment(xb, rng))
            loss = nt_xent(z1, z2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.detach().item() * len(idx)
            cnt += len(idx)
        print(f'[SimCLR] epoch {ep + 1}/{args.sim_epochs}  loss={tot / cnt:.4f}')
    return model


@torch.no_grad()
def simclr_feat(model, X, batch=512):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).unsqueeze(1).to(DEVICE)
        h, _ = model(xb)
        out.append(h.cpu().numpy())
    return np.concatenate(out, axis=0)


# =============================================================================
# 4. 优化后的 learn_v3 规则（逐样本、无梯度）
# =============================================================================
class MarginNet:
    """s = W f；正确类最匹配模板 j 与最错模板 w：
       若 s[j] - s[w] > margin 则不学；
       否则 W[j] += α f/‖f‖²，W[w] -= α f/‖f‖²（每次只改两行）。
       每个 epoch 末保存快照，最终权重取各快照平均（Polyak）。"""

    def __init__(self, W, margin=MARGIN):
        self.mat = np.array(W, np.float32, copy=True)
        self.margin = margin

    def learn(self, f, y, alpha):
        s = self.mat.dot(f)
        w = int(np.argmax(np.where(np.arange(10) == y, -np.inf, s)))
        if s[y] - s[w] > self.margin:
            return
        step = (alpha / (float(f.dot(f)) + 1e-8)) * f
        self.mat[y] += step
        self.mat[w] -= step

    def predict(self, F):
        return np.argmax(F @ self.mat.T, axis=1)


def train_rule(F_tr, y_tr, F_te, y_te, seeds, alpha=ALPHA, margin=MARGIN,
               epochs=RULE_EPOCHS):
    """训练 + 测试；返回 (平均准确率, 标准差, 每个种子的平均权重 W)"""
    accs, Ws = [], []
    for s in seeds:
        rng = np.random.default_rng(5000 + s)
        net = MarginNet(np.zeros((10, F_tr.shape[1]), np.float32),
                        margin=margin)
        Wavg = None
        for ep in range(epochs):
            for i in rng.permutation(len(F_tr)):
                net.learn(F_tr[i], int(y_tr[i]), alpha)
            Wavg = (net.mat.copy() if Wavg is None
                    else (Wavg * ep + net.mat) / (ep + 1))
        accs.append(float(np.mean(np.argmax(F_te @ Wavg.T, axis=1) == y_te)))
        Ws.append(Wavg)
    return float(np.mean(accs)), float(np.std(accs)), Ws


def l2n(F):
    return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)


# =============================================================================
# 5. 图
# =============================================================================
def save_confusion(conf, path, title):
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(conf, cmap='Blues')
    ax.set_xlabel('预测')
    ax.set_ylabel('真实')
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    th = conf.max() / 2
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(conf[i, j]), ha='center', va='center',
                    fontsize=8, color='white' if conf[i, j] > th else 'black')
    fig.colorbar(im, fraction=0.046, pad=0.04)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# =============================================================================
# 6. 主流程
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description='最终方案：MAE + SimCLR + 前向规则')
    p.add_argument('--mae-epochs', type=int, default=8)
    p.add_argument('--sim-epochs', type=int, default=30)
    p.add_argument('--sim-batch', type=int, default=512)
    p.add_argument('--sim-feat', type=int, default=256)
    p.add_argument('--base', type=int, default=16)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    p.add_argument('--quick', action='store_true')
    p.add_argument('--tag', default='final')
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.join(ROOT, 'runs', args.tag)
    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    sys.stdout = Tee(os.path.join(ROOT, f'run_{args.tag}.log'), sys.stdout)
    t_start = time.time()
    print('=' * 78)
    print('最终方案：bbox + MAE/SimCLR 特征融合 + 优化后的前向规则')
    print(f'device={DEVICE}  margin={MARGIN}  alpha={ALPHA}  '
          f'rule_epochs={RULE_EPOCHS}')
    print('=' * 78)

    set_seed(0)
    (x_train, y_train), (x_test, y_test) = load_mnist()
    if args.quick:
        x_train, y_train = x_train[:6000], y_train[:6000]
        x_test, y_test = x_test[:2000], y_test[:2000]
        args.mae_epochs = 2
        args.sim_epochs = 2
        args.sim_batch = 256
        args.seeds = args.seeds[:1]

    X_tr = build_bbox(x_train, 28, 'train')
    X_te = build_bbox(x_test, 28, 'test ')
    print(f'[data] train {X_tr.shape}  test {X_te.shape}')

    # ---------- 特征 ----------
    print('\n[1] MAE（遮挡重建）')
    mae = train_mae(X_tr, args)
    M_tr, M_te = mae_h2(mae, X_tr), mae_h2(mae, X_te)

    print('\n[2] SimCLR（对比学习）')
    sim = train_simclr(X_tr, args)
    S_tr, S_te = simclr_feat(sim, X_tr), simclr_feat(sim, X_te)

    # ---------- 分类 ----------
    print('\n[3] 前向规则分类')
    res = {}
    res['MAE h2'], std_m, _ = train_rule(M_tr, y_train, M_te, y_test,
                                        args.seeds)
    print(f'  MAE h2   : {res["MAE h2"]:.4f} ± {std_m:.4f}')
    res['SimCLR'], std_s, _ = train_rule(S_tr, y_train, S_te, y_test,
                                         args.seeds)
    print(f'  SimCLR   : {res["SimCLR"]:.4f} ± {std_s:.4f}')

    C_tr = np.hstack([l2n(S_tr), l2n(M_tr)]).astype(np.float32)
    C_te = np.hstack([l2n(S_te), l2n(M_te)]).astype(np.float32)
    res['FUSION'], std_f, Ws = train_rule(C_tr, y_train, C_te, y_test,
                                          args.seeds)
    print(f'  融合+规则 : {res["FUSION"]:.4f} ± {std_f:.4f}')

    # ---------- 评估细节 ----------
    W = Ws[0]
    pred = np.argmax(C_te @ W.T, axis=1)
    conf = np.zeros((10, 10), np.int64)
    np.add.at(conf, (y_test, pred), 1)
    stat = np.zeros((10, 2), np.int64)
    stat[:, 0] = np.bincount(y_test, minlength=10)
    stat[:, 1] = np.bincount(y_test[pred == y_test], minlength=10)
    per_class = [float(stat[c, 1] / stat[c, 0]) for c in range(10)]
    save_confusion(conf, os.path.join(fig_dir, 'confusion.png'),
                   f'最终方案混淆矩阵（{res["FUSION"]:.2%}）')
    np.save(os.path.join(out_dir, 'W.npy'), W)

    summary = {
        'pipeline': 'bbox 28x28 -> MAE h2 + SimCLR 256 -> per-block L2 concat '
                    '-> optimized forward rule (margin gate + normalized step '
                    '+ epoch averaging)',
        'acc': res, 'std': {'MAE': std_m, 'SimCLR': std_s, 'FUSION': std_f},
        'per_class': per_class,
        'hyper': {'margin': MARGIN, 'alpha': ALPHA, 'rule_epochs': RULE_EPOCHS,
                  'mae_epochs': args.mae_epochs,
                  'sim_epochs': args.sim_epochs,
                  'sim_batch': args.sim_batch,
                  'sim_feat': args.sim_feat},
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print('\n' + '=' * 78)
    print(f'最终测试准确率：{res["FUSION"]:.4f} ± {std_f:.4f}')
    print('每类准确率：' + '  '.join(f'{c}:{per_class[c]:.3f}'
                                    for c in range(10)))
    print(f'总耗时 {time.time() - t_start:.1f}s')
    print(f'结果目录 {out_dir}')


if __name__ == '__main__':
    main()
