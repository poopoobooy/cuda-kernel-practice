"""
Triton fused softmax kernel
================================

参考: https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html

要点（面试官会问的）:
  1. 一个 program 处理一行 (row)，行内并行 reduce 求 max / sum
  2. safe softmax: 减 max 防止 expf 溢出 (FP32 max ≈ 3.4e38，但 expf(>89) 就溢出)
  3. 单 kernel 把 max / sub / exp / sum / div 全 fuse —— 只读一次 x，只写一次 y
     PyTorch eager 会拆成 4-5 个 kernel，每个都要走一遍 HBM
  4. num_warps 调度: 块大（>2048）要更多 warps 才能 hide latency
"""

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    output_ptr,            # 输出张量首地址
    input_ptr,             # 输入张量首地址
    input_row_stride,      # 输入相邻两行的元素间距 (= n_cols, 假设连续)
    output_row_stride,     # 输出相邻两行的元素间距
    n_rows,                # 总行数 (主要给 mask 用)
    n_cols,                # 每行的列数
    BLOCK_SIZE: tl.constexpr,  # 编译期常量，每行 load 的 tile 大小（>= n_cols 即可一次性吃下整行）
    num_stages: tl.constexpr,
):
    # 每个 program 处理一行；这里用了 grid stride loop（program_id 范围 < n_rows 时也能复用）
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # ---- 1. load 一整行到寄存器/SRAM ----
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # mask: 超出 n_cols 的位置填 -inf —— 这样后面取 max 时不会被算进去
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

        # ---- 2. safe softmax ----
        # 减 max 是数值稳定性的关键: exp(x - max) 永远 <= 1，绝不会上溢
        row_minus_max = row - tl.max(row, axis=0)
        # __expf 等价物: Triton 默认就是用 fast intrinsic
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator

        # ---- 3. 写回 ----
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


def softmax_triton(x: torch.Tensor) -> torch.Tensor:
    """对最后一维做 softmax。要求 x 是 2D + 连续 + cuda 上的 fp32"""
    assert x.is_cuda and x.dtype == torch.float32 and x.dim() == 2, \
        "softmax_triton 当前实现只支持 2D fp32 cuda 张量"
    n_rows, n_cols = x.shape

    # BLOCK_SIZE 取大于等于 n_cols 的最小 2 的幂
    # 这样一个 program 一次就能 load 整行，省掉行内的二次 reduce
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # num_warps 调度规则 (Triton 官方 tutorial 给的经验)
    # 行越宽，越需要多 warp 来 hide HBM latency
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    # num_stages: 软件流水线深度，A100/Ada 上 4 是个稳的值
    num_stages = 4 if x.device.type == 'cuda' else 2

    y = torch.empty_like(x)

    # grid 大小: 让 program 数 = 一定数量（这里直接 = n_rows 也行；用 SM 数 * 2 也行）
    # 这里偷个懒 = n_rows，每行一个 program
    grid = (n_rows,)

    softmax_kernel[grid](
        y, x,
        x.stride(0), y.stride(0),
        n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    return y


# ---------------------------------------------------------------
# 自测 / benchmark
# ---------------------------------------------------------------
def _correctness_test():
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device='cuda', dtype=torch.float32)
    y_triton = softmax_triton(x)
    y_torch = torch.softmax(x, dim=-1)
    # softmax 在小数值上易有 1e-6 量级误差，atol/rtol 给宽一点
    assert torch.allclose(y_triton, y_torch, atol=1e-5, rtol=1e-4), \
        f"max diff = {(y_triton - y_torch).abs().max().item()}"
    print(f"[OK] softmax_triton vs torch.softmax  max_diff="
          f"{(y_triton - y_torch).abs().max().item():.3e}")


def _bench(M: int, N: int, n_iter: int = 100, n_warmup: int = 20):
    """返回 dict: {'triton_ms', 'torch_ms', 'speedup'}"""
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)

    # warmup
    for _ in range(n_warmup):
        softmax_triton(x)
        torch.softmax(x, dim=-1)
    torch.cuda.synchronize()

    # Triton
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        softmax_triton(x)
    end.record()
    torch.cuda.synchronize()
    triton_ms = start.elapsed_time(end) / n_iter

    # PyTorch eager
    start.record()
    for _ in range(n_iter):
        torch.softmax(x, dim=-1)
    end.record()
    torch.cuda.synchronize()
    torch_ms = start.elapsed_time(end) / n_iter

    # 带宽: softmax 是 memory bound（每个 elem 读 1 次 + 写 1 次）
    bytes_per_iter = M * N * 4 * 2  # fp32 read + write
    gb_s_triton = bytes_per_iter / (triton_ms * 1e-3) / 1e9
    gb_s_torch = bytes_per_iter / (torch_ms * 1e-3) / 1e9
    return {
        'M': M, 'N': N,
        'triton_ms': triton_ms,
        'torch_ms': torch_ms,
        'speedup': torch_ms / triton_ms,
        'triton_gbs': gb_s_triton,
        'torch_gbs': gb_s_torch,
    }


if __name__ == '__main__':
    _correctness_test()
    print(f"{'M':>6} {'N':>6} {'triton(ms)':>12} {'torch(ms)':>12} "
          f"{'speedup':>8} {'triton(GB/s)':>14} {'torch(GB/s)':>14}")
    for N in [128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
        r = _bench(M=4096, N=N)
        print(f"{r['M']:>6} {r['N']:>6} {r['triton_ms']:>12.4f} "
              f"{r['torch_ms']:>12.4f} {r['speedup']:>8.2f} "
              f"{r['triton_gbs']:>14.1f} {r['torch_gbs']:>14.1f}")
