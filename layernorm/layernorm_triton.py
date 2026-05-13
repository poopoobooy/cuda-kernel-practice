"""
Triton fused LayerNorm (forward only)
================================

要点（面试官会问的）:
  1. LayerNorm 公式:
       mean = sum(x) / N
       var  = sum((x - mean)^2) / N
       y    = (x - mean) / sqrt(var + eps) * weight + bias
  2. 数值稳定性: mean / var 必须用 FP32 累加！哪怕输入是 FP16/BF16
     否则 N>=4096 时 var 累加误差会爆掉
  3. 一行一个 program，行内 BLOCK_SIZE 一次性吃下整行 —— 行宽 <= 64K 时这样最快
  4. 保存 mean / rstd 给反向用（哪怕这里没写 backward，也存上，将来可补）
  5. fuse 一波: mean / var / norm / affine 在一个 kernel 里搞完，HBM 流量降 4-5 倍
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_fwd_kernel(
    X,          # 输入 [M, N]
    Y,          # 输出 [M, N]
    W,          # weight [N]
    B,          # bias [N]
    Mean,       # 输出 mean [M] —— 给反向用
    Rstd,       # 输出 1/sqrt(var+eps) [M] —— 给反向用
    stride,     # X 行 stride（按 N 假设连续 = N）
    N,          # 行宽
    eps,        # 防 div 0
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride

    # ---- 1. compute mean ----
    # 用 fp32 累加，不管 X 是什么 dtype
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N

    # ---- 2. compute var ----
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.0)  # mask 之外置 0，不影响 sum
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    # 把 mean / rstd 存起来（每行一个）
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)

    # ---- 3. normalize + affine ----
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)


def layernorm_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                     eps: float = 1e-5):
    """对最后一维做 LayerNorm。返回 (y, mean, rstd)"""
    assert x.is_cuda and x.is_contiguous()
    assert weight.shape == bias.shape == (x.shape[-1],)
    # 摊平成 2D
    *batch_shape, N = x.shape
    M = 1
    for d in batch_shape:
        M *= d
    x_flat = x.view(M, N)

    y = torch.empty_like(x_flat)
    mean = torch.empty((M,), device=x.device, dtype=torch.float32)
    rstd = torch.empty((M,), device=x.device, dtype=torch.float32)

    # BLOCK_SIZE: 取 2 的幂，但不要超过硬件限制（H100 / Ada 上 <= 64K 都行，但太大会爆 SRAM）
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))

    # warps 调度: 同 softmax
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    _layernorm_fwd_kernel[(M,)](
        x_flat, y, weight, bias, mean, rstd,
        x_flat.stride(0), N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y.view(*batch_shape, N), mean, rstd


# ---------------------------------------------------------------
# 自测 / benchmark
# ---------------------------------------------------------------
def _correctness_test():
    torch.manual_seed(0)
    M, N = 1151, 2049
    x = torch.randn(M, N, device='cuda', dtype=torch.float32) * 5 + 0.5
    w = torch.randn(N, device='cuda', dtype=torch.float32)
    b = torch.randn(N, device='cuda', dtype=torch.float32)
    y_triton, _, _ = layernorm_triton(x, w, b, eps=1e-5)
    y_torch = torch.nn.functional.layer_norm(x, (N,), weight=w, bias=b, eps=1e-5)
    diff = (y_triton - y_torch).abs().max().item()
    assert diff < 1e-3, f"max diff too big: {diff}"
    print(f"[OK] layernorm_triton vs torch  max_diff={diff:.3e}")


def _bench(M: int, N: int, n_iter: int = 100, n_warmup: int = 20):
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    w = torch.randn(N, device='cuda', dtype=torch.float32)
    b = torch.randn(N, device='cuda', dtype=torch.float32)

    for _ in range(n_warmup):
        layernorm_triton(x, w, b)
        torch.nn.functional.layer_norm(x, (N,), weight=w, bias=b)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        layernorm_triton(x, w, b)
    end.record()
    torch.cuda.synchronize()
    triton_ms = start.elapsed_time(end) / n_iter

    start.record()
    for _ in range(n_iter):
        torch.nn.functional.layer_norm(x, (N,), weight=w, bias=b)
    end.record()
    torch.cuda.synchronize()
    torch_ms = start.elapsed_time(end) / n_iter

    bytes_per_iter = M * N * 4 * 2
    return {
        'M': M, 'N': N,
        'triton_ms': triton_ms, 'torch_ms': torch_ms,
        'speedup': torch_ms / triton_ms,
        'triton_gbs': bytes_per_iter / (triton_ms * 1e-3) / 1e9,
        'torch_gbs':  bytes_per_iter / (torch_ms * 1e-3) / 1e9,
    }


if __name__ == '__main__':
    _correctness_test()
    print(f"{'M':>6} {'N':>6} {'triton(ms)':>12} {'torch(ms)':>12} "
          f"{'speedup':>8} {'triton(GB/s)':>14} {'torch(GB/s)':>14}")
    for N in [256, 512, 1024, 2048, 4096, 8192]:
        r = _bench(M=4096, N=N)
        print(f"{r['M']:>6} {r['N']:>6} {r['triton_ms']:>12.4f} "
              f"{r['torch_ms']:>12.4f} {r['speedup']:>8.2f} "
              f"{r['triton_gbs']:>14.1f} {r['torch_gbs']:>14.1f}")
