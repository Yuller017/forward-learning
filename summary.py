#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""汇总整个项目的关键结果，生成总览图（数据来自各实验的 runs/）"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ROOT = os.path.dirname(os.path.abspath(__file__))

# (名称, 测试准确率, 来源, 类别)
RESULTS = [
    ('最小框转正 + 16×16 像素', 0.8408, '实验12', 'pixel'),
    ('紧 bbox + 16×16 像素', 0.9024, '实验12', 'pixel'),
    ('VAE μ (64)', 0.9274, '实验20', 'feature'),
    ('NMF r=64', 0.9289, '实验20', 'feature'),
    ('PCA r=512', 0.9331, '实验19', 'feature'),
    ('Gabor 散射 (192)', 0.9765, '实验20', 'feature'),
    ('UNet h2 + 优化规则', 0.9810, '实验16', 'feature'),
    ('k-means 字典 K=256', 0.9839, '实验19', 'feature'),
    ('局部码 K=64', 0.9842, '实验18', 'feature'),
    ('MAE h2', 0.9855, '实验20/22', 'feature'),
    ('SimCLR+ (256)', 0.9894, '实验22', 'feature'),
    ('MAE + SimCLR 拼接', 0.9918, '实验21', 'fusion'),
    ('MAE + SimCLR+ 拼接', 0.9926, '实验22/23', 'fusion'),
]

COLORS = {'pixel': '#95a5a6', 'feature': '#3498db', 'fusion': '#2ecc71'}


def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             gridspec_kw={'width_ratios': [1.35, 1]})

    # 左：排行榜
    ax = axes[0]
    names = [r[0] for r in RESULTS]
    accs = [r[1] * 100 for r in RESULTS]
    ys = range(len(RESULTS))
    bars = ax.barh(list(ys), accs, color=[COLORS[r[3]] for r in RESULTS])
    ax.set_yticks(list(ys))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(80, 100)
    ax.set_xlabel('测试准确率 (%)')
    ax.set_title('各方法排行榜（同一前向规则）', fontsize=12)
    ax.grid(alpha=0.3, axis='x')
    for b, v in zip(bars, accs):
        ax.text(v + 0.15, b.get_y() + b.get_height() / 2, f'{v:.2f}',
                va='center', fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in COLORS]
    ax.legend(handles, ['像素级', '特征学习', '特征融合'], loc='lower right',
              fontsize=9)

    # 右：演进路线
    ax = axes[1]
    steps = [
        ('12 转正像素', 84.08),
        ('12 紧框像素', 90.24),
        ('13/16 UNet h2', 97.23),
        ('16 规则优化', 98.10),
        ('18 局部码', 98.42),
        ('20 SimCLR', 98.81),
        ('22/23 融合', 99.26),
    ]
    xs = range(len(steps))
    ax.plot(xs, [s[1] for s in steps], marker='o', lw=2, color='#2ecc71')
    ax.set_xticks(list(xs))
    ax.set_xticklabels([s[0] for s in steps], rotation=25, ha='right', fontsize=9)
    ax.set_ylim(82, 100)
    ax.set_ylabel('测试准确率 (%)')
    ax.set_title('准确率演进', fontsize=12)
    ax.grid(alpha=0.3)
    for x, (_, v) in zip(xs, steps):
        ax.text(x, v + 0.7, f'{v:.2f}', ha='center', fontsize=9)

    fig.suptitle('MNIST 前向学习：从像素模板到 99.26%', fontsize=14)
    fig.tight_layout()
    out = os.path.join(ROOT, '总览.png')
    fig.savefig(out, dpi=140)
    print(out)


if __name__ == '__main__':
    main()
