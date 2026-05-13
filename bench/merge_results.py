"""把 CUDA C++ bench (softmax_cuda / reduction) 和 torch_ext 的输出合到 results.json

依赖前置:
  - sgemm/sgemm_bench.exe 已跑 (生成 results/sgemm_raw.json)
  - softmax/softmax_cuda.exe 跑过, 数字这里手填 (输出格式简单, 解析没必要)
  - reduction/reduction.exe 跑过
  - torch_ext/test.py 跑过, 抓到 14151 / 21493 GFLOPS 这两个数字
"""

import json
import subprocess
import re
from pathlib import Path

import torch  # noqa: F401 ; MKL 初始化解 DLL 冲突
ROOT = Path(__file__).resolve().parent.parent


def run_and_capture(exe_args, cwd=None):
    """跑一个 exe, 返回 stdout"""
    r = subprocess.run(exe_args, cwd=cwd, capture_output=True, text=True, timeout=180)
    return r.stdout + r.stderr


def parse_softmax_cuda(out: str):
    """[softmax_cuda] 0.0325 ms/iter, 516.4 GB/s -> dict"""
    m = re.search(r"\[softmax_cuda\] ([\d.]+) ms/iter, ([\d.]+) GB/s", out)
    if not m:
        return None
    return {'ms': float(m.group(1)), 'gbs': float(m.group(2))}


def parse_reduction(out: str):
    """3 行: [name] x.xx ms, x.x GB/s, sum_diff = ..."""
    rows = []
    for m in re.finditer(r"\[(\S+)\s*\]\s+([\d.]+) ms,\s+([\d.]+) GB/s,\s+sum_diff = ([\d.e+-]+)", out):
        rows.append({
            'version': m.group(1),
            'ms': float(m.group(2)),
            'gbs': float(m.group(3)),
            'sum_diff': float(m.group(4)),
        })
    return rows


def main():
    res_path = ROOT / 'results' / 'results.json'
    with open(res_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # ---- softmax_cuda ----
    print("[run] softmax_cuda.exe 4096 512")
    out = run_and_capture([str(ROOT / 'softmax' / 'softmax_cuda.exe'), '4096', '512'])
    print(out.strip())
    sc = parse_softmax_cuda(out)
    if sc:
        data['softmax_cuda'] = {'M': 4096, 'N': 512, **sc}
        print(f"  saved: {sc}")

    # ---- reduction ----
    print("\n[run] reduction.exe")
    out = run_and_capture([str(ROOT / 'reduction' / 'reduction.exe')])
    print(out.strip())
    rd = parse_reduction(out)
    if rd:
        data['reduction'] = rd
        print(f"  saved: {len(rd)} variants")

    # ---- torch_ext ----  (跑 test.py 抓最后两行 'ours' 'torch' 速度)
    print("\n[run] torch_ext/test.py")
    py = subprocess.run([
        'C:\\Users\\16229\\miniconda3\\envs\\cuda-kernel\\python.exe',
        str(ROOT / 'torch_ext' / 'test.py'),
    ], cwd=ROOT / 'torch_ext', capture_output=True, text=True, timeout=180)
    out = py.stdout
    print(out)
    m_ours  = re.search(r"ours\s*:\s+([\d.]+) ms,\s+([\d.]+) GFLOPS", out)
    m_torch = re.search(r"torch\s*:\s+([\d.]+) ms,\s+([\d.]+) GFLOPS", out)
    if m_ours and m_torch:
        data['torch_ext'] = {
            'M': 4096, 'N': 4096, 'K': 4096,
            'ours_ms':    float(m_ours.group(1)),
            'ours_gflops': float(m_ours.group(2)),
            'torch_ms':   float(m_torch.group(1)),
            'torch_gflops': float(m_torch.group(2)),
            'pct_torch_matmul': float(m_ours.group(2)) / float(m_torch.group(2)) * 100,
        }
        print(f"  saved: ours={m_ours.group(2)} GFLOPS, torch={m_torch.group(2)}")

    # ---- 写回 ----
    with open(res_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] -> {res_path}")


if __name__ == '__main__':
    main()
