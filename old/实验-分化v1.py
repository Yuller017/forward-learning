import os
import struct
import numpy as np
import cv2
import matplotlib.pyplot as plt
import gzip
import time


def read_idx(filename):
    """读取IDX格式的文件"""
    with gzip.open(filename, 'rb') as f:
        # 读取魔数和维度信息
        magic_number = struct.unpack('>I', f.read(4))[0]
        
        # 根据魔数判断数据类型
        if magic_number == 2051:  # 图像文件
            num_images = struct.unpack('>I', f.read(4))[0]
            num_rows = struct.unpack('>I', f.read(4))[0]
            num_cols = struct.unpack('>I', f.read(4))[0]
            data = np.frombuffer(f.read(), dtype=np.uint8)
            data = data.reshape(num_images, num_rows, num_cols)
        elif magic_number == 2049:  # 标签文件
            num_labels = struct.unpack('>I', f.read(4))[0]
            data = np.frombuffer(f.read(), dtype=np.uint8)
        else:
            raise ValueError(f"未知的魔数: {magic_number}")
        
    return data


# 读取数据
x_train = read_idx('data/MNIST/raw/train-images-idx3-ubyte.gz')
y_train = read_idx('data/MNIST/raw/train-labels-idx1-ubyte.gz')
x_test = read_idx('data/MNIST/raw/t10k-images-idx3-ubyte.gz')
y_test = read_idx('data/MNIST/raw/t10k-labels-idx1-ubyte.gz')

# 归一化
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

print(f"训练图像: {x_train.shape}")  # (60000, 28, 28)
print(f"训练标签: {y_train.shape}")  # (60000,)
print(f"测试图像: {x_test.shape}")   # (10000, 28, 28)
print(f"测试标签: {y_test.shape}")   # (10000,)

# 可视化一些样本
if False:
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    # axes.set(title="样例")
    for i, ax in enumerate(axes.flat):
        ax.imshow(x_train[i], cmap='gray')
        ax.set_title(f"Label: {y_train[i]}")
        ax.axis('off')
    plt.tight_layout()
    plt.show(block=True)


def test_260615():
    ksz = 4
    feats = [] # np.random.rand(1024, ksz, ksz)

    for step, it in enumerate(x_train):
        h, w = it.shape
        for i in range(h-ksz):
            for j in range(w-ksz):
                patch = it[i:i+ksz, j:j+ksz]
                patch_mean = np.mean(patch)
                patch = patch - patch_mean
                patch_absmax = np.max(np.abs(patch))
                if patch_absmax < 1e-3:
                    continue

                patch = patch / patch_absmax
                patch_s2 = np.sum(patch * patch)
                
                if len(feats) == 0:
                    feats.append(patch)
                else:
                    sim = np.zeros(len(feats))
                    for k, f in enumerate(feats):
                        sim[k] = np.sum(f * patch) / (np.sum(f * f) * patch_s2 + 1e-6)
                        # print(np.sum(f*patch), np.sum(f * f), f)
                    # print(sim)
                    idx = np.argmax(sim)
                    if sim[idx] < 0.1:
                        feats.append(patch)
        print(f'{step:06d}  {len(feats)}')
        if step > 100:
            break

    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    # axes.set(title="样例")
    for i, ax in enumerate(axes.flat):
        if i >= len(feats):
            break
        ax.imshow(feats[i], cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.show(block=True)

# class Node:
#     def __init__(self, input, classmap):
        
#     def 


def find_beg_end(val):
    for beg in range(val.size):
        if val[beg] > 0:
            break
    for end in range(val.size - 1, beg, -1):
        if val[end] > 0:
            break
    margin = 0
    return max(beg-margin, 0), min(end + 1 + margin, val.size-1)


class MyNet:
    def __init__(self):
        self.ksz = 16
        self.head = 1  # 没有用
        self.mat = np.random.rand(10 * self.head, self.ksz**2)

        # 用于记录
        self.enn = 0
        self.epoch = 0
        

    def get_feat(self, x):
        x1 = np.sum(x, axis=(1,))
        x2 = np.sum(x, axis=(0,))
        beg1, end1 = find_beg_end(x1)
        beg2, end2 = find_beg_end(x2)

        feat = cv2.resize(x[beg1:end1, beg2:end2], (self.ksz, self.ksz), interpolation=cv2.INTER_AREA)
        # cv2.imshow('', cv2.resize(feat, (64, 64), interpolation=cv2.INTER_NEAREST))
        # cv2.waitKey(1)
        f = feat.flatten()
        return f

    def learn_v1(self, x, y:int):
        f = self.get_feat(x)
        yp = self.mat.dot(f)
        idx1 = np.argmax(yp)
        val1 = yp[idx1]

        yp[idx1] = 0
        idx2 = np.argmax(yp)
        val2 = yp[idx2]

        alpha = 0.002
        maxval = np.inf # 对目标足够确信时，没必要再学习，所以权重可以不限制
        if idx1 // self.head == y:
            if val2 > 0.25 * val1:
                self.mat[idx1] = np.clip(self.mat[idx1] + alpha * f, -maxval, maxval)
        else:
            # 方案一
            self.mat[idx1] = np.clip(self.mat[idx1] - alpha * f, -maxval, maxval)

            # 方案二，不行
            # err = self.mat[idx] - self.mat[y]
            # self.mat[idx] = np.clip(self.mat[idx] + alpha * err, -2, 2)
            # idx = idx - a * (idx - y)
            #     = idx - a * idx + a * y
            #     = (1-a) * idx + a * y

            if True:
                tt = [f, self.mat[idx1], self.mat[y], abs(self.mat[idx1] - self.mat[y])]
                im = np.hstack([it.reshape(self.ksz, self.ksz) for it in tt])
                vmin = im.min()
                vmax = im.max()
                im = (im - vmin) / (vmax - vmin)
                im = cv2.resize(im, (0,0), fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
                save_path = f"log/{y}-{idx1}"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                cv2.imwrite(f"{save_path}/{self.epoch}_{self.enn:06d}_{val1:.3f}_{val2:.3f}.png", (255 * im).astype(np.uint8))
                self.enn += 1
    
    def learn_v2(self, x, y:int, alpha):
        """ head = 1 时最好
        """
        f = self.get_feat(x)
        yp = self.mat.dot(f)
        idx = np.argmax(yp)
        
        maxval = np.inf # 对目标足够确信时，没必要再学习，所以权重可以不限制。但考虑到生物生长的限制，有限制也合理
        n = np.sum(yp > 1)
        if n > 1:
            for k in range(10 * self.head):
                if yp[k] <= 1: continue
                flag = 1 if k // self.head == y else -1
                self.mat[k] = np.clip(self.mat[k] + flag * alpha * f, -maxval, maxval)
        else: #elif n == 0:
            flag = 1 if idx // self.head == y else -1
            self.mat[idx] = np.clip(self.mat[idx] + flag * alpha * f, -maxval, maxval)

    def learn_v3(self, x, y:int, alpha):
        f = self.get_feat(x)
        yp = self.mat.dot(f)
        idx = np.argmax(yp)
        
        # alpha = 0.05
        maxval = np.inf # 对目标足够确信时，没必要再学习，所以权重可以不限制
        n = np.sum(yp > 1)
        if n == 1 and yp[idx] > 2 and idx == y:
            pass
        elif n == 0 or n == 1:
            flag = 1 if idx // self.head == y else -1
            self.mat[idx] = np.clip(self.mat[idx] + flag * alpha * f, -maxval, maxval)
        else:
            for k in range(10 * self.head):
                if yp[k] <= 1: continue
                # flag = 1 if k // self.head == y else -1
                if k // self.head == y:
                    if k == idx:
                        self.mat[k] = np.clip(self.mat[k] + alpha * f, -maxval, maxval)
                    else:
                        # self.mat[k] = np.clip(self.mat[k] + 0.5 * alpha * f, -maxval, maxval)
                        pass
                else:
                    self.mat[k] = np.clip(self.mat[k] - alpha * f, -maxval, maxval)

    def learn(self, x, y:int, alpha):
        self.learn_v3(x, y, alpha)


    def forget(self):
        """ 遗忘，一是减少能量消耗，二是弱化不重要的特征
        """
        self.mat = 0.99 * self.mat

    def predict(self, x):
        f = self.get_feat(x)
        yp = self.mat.dot(f)
        idx = np.argmax(yp)
        return idx // self.head

def test():
    """ 按整体到局部的分辨策略
        先把图像都缩放到 8*8，视图分辨，然后细化
        8*8 -> 10 按之前的方法
    """
    model = MyNet()

    epochs = 10
    epoch_record = {}
    alpha = 0.05
    for epoch in range(epochs):
        print(f'epoch {epoch+1}')

        model.epoch = epoch
        model.enn = 0

        for ii, x in enumerate(x_train):
            y = y_train[ii]
            model.learn(x, y, alpha)
            if (ii + 1) % 1000 == 0:
                model.forget()
                pass
        alpha *= 0.5

        # 验证
        stat = np.zeros((10, 2), dtype=np.int32)
        for ii, x in enumerate(x_test):
            y = y_test[ii]
            stat[y,0] += 1

            idx = model.predict(x)
            if y == idx:
                stat[y,1] += 1
            
        
        for i in range(10):
            acc = stat[i,1] / stat[i,0]
            if i not in epoch_record:
                epoch_record[i] = []
            epoch_record[i].append(acc)
            print(f'{i:4d}  {stat[i,1]:4d}/{stat[i,0]:4d}    {acc}')

        print(f'total {np.sum(stat[:,1])/np.sum(stat[:,0])}')

    for i in range(10):
        plt.plot(epoch_record[i], label=f'{i}')
    plt.legend()
    plt.show()

    for hi in range(model.head):
        fig, axes = plt.subplots(2, 5, figsize=(10, 5))
        mat = model.mat[hi::model.head]
        # axes.set(title="样例")
        for i, ax in enumerate(axes.flat):
            im = mat[i].reshape((model.ksz, model.ksz))
            ax.imshow(im, cmap='gray')
            ax.set_title(f"Label: {i}")
            ax.axis('off')
        plt.tight_layout()
        plt.show(block=True)
test()