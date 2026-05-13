"""读 results.json 画性能曲线图"""

import json
from pathlib import Path

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


if __name__ == '__main__':
    main()
