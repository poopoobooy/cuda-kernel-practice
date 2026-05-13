"""
Triton SGEMM (FP32, row-major)
================================

参考: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html

要点（面试官会问的）:
  1. block 的 grid 是 2D over (M/BM, N/BN), 每个 program 算 BM x BN 的 C 子块
  2. K 维 loop: 加载 As/Bs 的 BK tile, 内积累加到 acc
  3. 用 tl.dot 自动调最优指令 (sm_89 上是 FMA + register tile)
  4. autotune 在 (BM, BN, BK, GROUP_M, num_warps, num_stages) 网格里搜最快配置
  5. swizzle (group-major ordering): 提升 L2 cache 命中率, 让相邻 block 共享 A/B 行列

  跟 CUDA C++ 五版的对应:
    - Triton 内部已经做好了 coalesce / SMEM tiling / 1D-2D thread tile / register reuse
    - 我们只需要选对 BM/BN/BK 让硬件吃饱
"""

import torch
import triton
import triton.language as tl


# ---- autotune 搜空间 ----
# 裁剪到 ~8 个经验上 Ada (sm_89) FP32 表现好的配置
# 全网格 200+ 个会让首跑 autotune 卡很久 (Windows JIT 编译尤其慢)
def _autotune_configs():
    return [
        triton.Config({'BM': 128, 'BN': 128, 'BK': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 16, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BM': 128, 'BN':  64, 'BK': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BM':  64, 'BN': 128, 'BK': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BM':  64, 'BN':  64, 'BK': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BM': 128, 'BN': 256, 'BK': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BM': 256, 'BN': 128, 'BK': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    ]


@triton.autotune(configs=_autotune_configs(), key=['M', 'N', 'K'])
@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)

    # ---- group-major swizzle ----
    # 默认 program 顺序是按 (col, row) 平铺, 相邻 program 共享 A 的行不共享 B 的列
    # 改成"每 GROUP_M 行一组, 一组内先纵向走" → 相邻 program 共享更多 A/B → L2 命中飙升
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ---- 当前 block 在 C 中的起始 ----
    offs_am = (pid_m * BM + tl.arange(0, BM)) % M
    offs_bn = (pid_n * BN + tl.arange(0, BN)) % N
    offs_k = tl.arange(0, BK)

    # 用指针偏移构造 A/B 的 block 视图
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        # 边界 mask: 最后一个 K-tile 可能不满
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BK, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BK, other=0.0)
        # tl.dot: Triton 把这个映射到最优硬件指令
        # 在 Ada FP32 上是 outer-product FMA; 如果 dtype 是 fp16/bf16 会用 TensorCore
        acc += tl.dot(a, b)
        a_ptrs += BK * stride_ak
        b_ptrs += BK * stride_bk

    # ---- 写回 C, 加边界 mask ----
    offs_cm = pid_m * BM + tl.arange(0, BM)
    offs_cn = pid_n * BN + tl.arange(0, BN)
    c_ptrs = C + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def matmul_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda and A.dtype == B.dtype == torch.float32
    assert A.shape[1] == B.shape[0]
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    # grid: 1D, autotune 后 BM/BN 由 META 决定, 所以 grid 也是个 lambda
    grid = lambda META: (triton.cdiv(M, META['BM']) * triton.cdiv(N, META['BN']),)
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    return C


# ---------------------------------------------------------------
def _correctness_test():
    torch.manual_seed(0)
    for M, K, N in [(512, 512, 512), (1024, 1024, 1024), (4096, 4096, 4096)]:
        A = torch.randn(M, K, device='cuda', dtype=torch.float32)
        B = torch.randn(K, N, device='cuda', dtype=torch.float32)
        C_ours = matmul_triton(A, B)
        C_ref = A @ B
        atol = max(1e-3, 5e-5 * K)
        diff = (C_ours - C_ref).abs().max().item()
        ok = diff < atol
        print(f"  M={M} K={K} N={N}  max_err={diff:.3e}  atol={atol:.1e}  "
              f"{'OK' if ok else 'FAIL'}")
        assert ok


def _bench(M: int, N: int, K: int, n_repeat: int = 10, n_warmup: int = 5):
    """L2 flush + median 稳定 bench, 返回 dict"""
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    flush = torch.empty(64 * 1024 * 1024 // 4, device='cuda', dtype=torch.float32)

    def time_one(fn):
        ts = []
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        for _ in range(n_warmup):
            fn()
        torch.cuda.synchronize()
        for _ in range(n_repeat):
            flush.zero_()
            torch.cuda.synchronize()
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        ts.sort()
        return ts[len(ts) // 2]

    triton_ms = time_one(lambda: matmul_triton(A, B))
    torch_ms  = time_one(lambda: A @ B)
    gflops = 2.0 * M * N * K
    return {
        'M': M, 'N': N, 'K': K,
        'triton_ms': triton_ms, 'torch_ms': torch_ms,
        'triton_tflops': gflops / (triton_ms * 1e-3) / 1e12,
        'torch_tflops':  gflops / (torch_ms  * 1e-3) / 1e12,
        'pct_torch':     torch_ms / triton_ms * 100,   # 注意定义: triton 快则 >100
    }


if __name__ == '__main__':
    assert torch.cuda.is_available()
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Triton: {triton.__version__}")

    print("\n== correctness ==")
    _correctness_test()

    print("\n== benchmark (FP32) ==")
    print(f"{'M=N=K':>8} {'triton(ms)':>12} {'torch(ms)':>12} "
          f"{'triton(TFLOPS)':>16} {'torch(TFLOPS)':>15} {'triton/torch':>14}")
    for sz in [512, 1024, 2048, 4096]:
        r = _bench(sz, sz, sz)
        print(f"{sz:>8} {r['triton_ms']:>12.3f} {r['torch_ms']:>12.3f} "
              f"{r['triton_tflops']:>16.2f} {r['torch_tflops']:>15.2f} "
              f"{r['pct_torch']:>13.1f}%")
