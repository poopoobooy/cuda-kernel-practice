"""
benchmark 总入口
================================

跑：
  - Triton softmax vs PyTorch eager
  - Triton layernorm vs PyTorch
  - (可选) SGEMM 五版 vs cuBLAS —— 需要先 nvcc 编出 sgemm/bench.exe，
    这里只读 SGEMM 跑出来的 results.json 合并

输出: results/results.json
"""

import json
import os
import sys
import time
from pathlib import Path

# 让我们能 import 上层目录的模块
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from softmax.softmax_triton import _bench as bench_softmax  # noqa: E402
from layernorm.layernorm_triton import _bench as bench_layernorm  # noqa: E402


def collect_softmax():
    results = []
    print("\n=== Softmax (Triton vs PyTorch eager) ===")
    print(f"{'M':>6} {'N':>6} {'triton(ms)':>12} {'torch(ms)':>12} "
          f"{'speedup':>8} {'triton(GB/s)':>14} {'torch(GB/s)':>14}")
    for N in [128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
        r = bench_softmax(M=4096, N=N)
        print(f"{r['M']:>6} {r['N']:>6} {r['triton_ms']:>12.4f} "
              f"{r['torch_ms']:>12.4f} {r['speedup']:>8.2f} "
              f"{r['triton_gbs']:>14.1f} {r['torch_gbs']:>14.1f}")
        results.append(r)
    return results


def collect_layernorm():
    results = []
    print("\n=== LayerNorm (Triton vs PyTorch) ===")
    print(f"{'M':>6} {'N':>6} {'triton(ms)':>12} {'torch(ms)':>12} "
          f"{'speedup':>8} {'triton(GB/s)':>14} {'torch(GB/s)':>14}")
    for N in [256, 512, 1024, 2048, 4096, 8192]:
        r = bench_layernorm(M=4096, N=N)
        print(f"{r['M']:>6} {r['N']:>6} {r['triton_ms']:>12.4f} "
              f"{r['torch_ms']:>12.4f} {r['speedup']:>8.2f} "
              f"{r['triton_gbs']:>14.1f} {r['torch_gbs']:>14.1f}")
        results.append(r)
    return results


def collect_sgemm_if_available():
    """SGEMM 数据是 CUDA C++ 跑出来的，bench.exe 会写到 results/sgemm_raw.json
    这里读一下合并进总 json"""
    sgemm_path = ROOT / 'results' / 'sgemm_raw.json'
    if sgemm_path.exists():
        with open(sgemm_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    print("\n[skip] SGEMM 结果未生成 (需要先 nvcc 编 sgemm/bench.cu 跑一次)")
    return None


def main():
    assert torch.cuda.is_available(), "需要 CUDA"
    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name}")
    print(f"PyTorch: {torch.__version__}")
    try:
        import triton
        print(f"Triton: {triton.__version__}")
    except ImportError:
        print("Triton: NOT INSTALLED")
        return 1

    payload = {
        'device': device_name,
        'torch_version': torch.__version__,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'softmax': collect_softmax(),
        'layernorm': collect_layernorm(),
        'sgemm': collect_sgemm_if_available(),
    }

    out = ROOT / 'results' / 'results.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] saved -> {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
