"""读 results.json 画性能曲线图

注: 必须 `import torch` 先初始化 MKL, 否则 numpy/matplotlib import 会 DLL 冲突
    (Windows + conda-forge pytorch + numpy 的已知坑)
"""

import json
from pathlib import Path

import torch  # noqa: F401 ; 必须先 import 它初始化 MKL
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent


def plot_softmax(data):
    if not data:
        return
    Ns = [r['N'] for r in data]
    gb_triton = [r['triton_gbs'] for r in data]
    gb_torch = [r['torch_gbs'] for r in data]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(Ns, gb_triton, marker='o', label='Triton fused', linewidth=2)
    ax.plot(Ns, gb_torch, marker='s', label='PyTorch eager', linewidth=2)
    ax.set_xscale('log', base=2)
    ax.set_xlabel('N (row width)')
    ax.set_ylabel('Bandwidth (GB/s)')
    ax.set_title('Softmax: Triton vs PyTorch  (M=4096, fp32, RTX 4090)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    out = ROOT / 'results' / 'softmax_curve.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    print(f"[OK] -> {out}")


def plot_layernorm(data):
    if not data:
        return
    Ns = [r['N'] for r in data]
    gb_triton = [r['triton_gbs'] for r in data]
    gb_torch = [r['torch_gbs'] for r in data]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(Ns, gb_triton, marker='o', label='Triton fused', linewidth=2)
    ax.plot(Ns, gb_torch, marker='s', label='PyTorch', linewidth=2)
    ax.set_xscale('log', base=2)
    ax.set_xlabel('N (row width)')
    ax.set_ylabel('Bandwidth (GB/s)')
    ax.set_title('LayerNorm: Triton vs PyTorch  (M=4096, fp32, RTX 4090)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    out = ROOT / 'results' / 'layernorm_curve.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    print(f"[OK] -> {out}")


def plot_sgemm(data):
    """sgemm 数据期望格式: list of {'version': str, 'gflops': float, 'pct_cublas': float}
    在 M=N=K=4096 单点上"""
    if not data:
        return
    versions = [r['version'] for r in data]
    gflops = [r['gflops'] for r in data]
    pct = [r.get('pct_cublas', 0) for r in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.bar(versions, gflops, color='steelblue')
    ax1.set_ylabel('GFLOPS')
    ax1.set_title('SGEMM GFLOPS  (M=N=K=4096, fp32, RTX 4090)')
    ax1.tick_params(axis='x', rotation=15)
    for i, v in enumerate(gflops):
        ax1.text(i, v, f'{v:.0f}', ha='center', va='bottom', fontsize=9)

    ax2.bar(versions, pct, color='coral')
    ax2.set_ylabel('% of cuBLAS')
    ax2.set_ylim(0, 110)
    ax2.set_title('% of cuBLAS')
    ax2.tick_params(axis='x', rotation=15)
    ax2.axhline(100, ls='--', c='gray', alpha=0.5)
    for i, v in enumerate(pct):
        ax2.text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)

    out = ROOT / 'results' / 'sgemm_curve.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    print(f"[OK] -> {out}")


def plot_flash_attention(fa_block):
    """画 FA1 vs naive 3-pass 的 speedup 曲线 (按 N 维)"""
    if not fa_block or not fa_block.get('runs'):
        return
    runs = fa_block['runs']
    # 用 (B*H, N) 联合作横轴标签
    labels = [f"BH={r['B']*r['H']}\nN={r['N']}" for r in runs]
    fa_ms = [r['fa1_ms'] for r in runs]
    naive_ms = [r['naive_ms'] for r in runs]
    speedup = [r['speedup'] for r in runs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    x = list(range(len(runs)))
    width = 0.38
    ax1.bar([i - width/2 for i in x], fa_ms, width, label='FA1 (ours)', color='steelblue')
    ax1.bar([i + width/2 for i in x], naive_ms, width, label='Naive 3-pass', color='coral')
    ax1.set_yscale('log')
    ax1.set_ylabel('Time (ms, log)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_title('Flash Attention 1 vs Naive 3-pass  (FP32, d=64, RTX 4090 Laptop)')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.plot(x, speedup, marker='o', linewidth=2, color='green')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel('Speedup (×)')
    ax2.set_title('FA1 speedup vs naive')
    ax2.axhline(1.0, ls='--', c='gray', alpha=0.5)
    for i, v in enumerate(speedup):
        ax2.text(i, v, f'{v:.1f}×', ha='center', va='bottom', fontsize=9)
    ax2.grid(True, alpha=0.3)

    out = ROOT / 'results' / 'flash_attention_curve.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    print(f"[OK] -> {out}")


def main():
    json_path = ROOT / 'results' / 'results.json'
    if not json_path.exists():
        print(f"找不到 {json_path}，先跑 bench/run_all.py")
        return
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    plot_softmax(data.get('softmax'))
    plot_layernorm(data.get('layernorm'))
    plot_sgemm(data.get('sgemm'))
    plot_flash_attention(data.get('flash_attention_1_fwd'))


if __name__ == '__main__':
    main()
