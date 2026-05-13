"""测试自定义 SGEMM extension 的正确性 + autograd 反向

跑法 (先编译好 sgemm_ext*.pyd):
  cd torch_ext
  python setup.py build_ext --inplace
  python test.py
"""

import sys
import torch
import sgemm_ext   # 我们自己 build 出来的扩展


class SGEMMFn(torch.autograd.Function):
    """把 sgemm_ext.sgemm_forward 包成 autograd.Function

    forward: C = A @ B
    backward:
        grad_A = grad_C @ B^T
        grad_B = A^T @ grad_C
    反向直接复用 torch.matmul (内部调 cuBLAS) —— 反向用什么 kernel 不重要,
    重点是 forward 走我们的 v5
    """

    @staticmethod
    def forward(ctx, A, B):
        ctx.save_for_backward(A, B)
        return sgemm_ext.sgemm_forward(A, B)

    @staticmethod
    def backward(ctx, grad_C):
        A, B = ctx.saved_tensors
        grad_A = grad_C @ B.t()
        grad_B = A.t() @ grad_C
        return grad_A, grad_B


def sgemm(A, B):
    return SGEMMFn.apply(A, B)


def test_correctness():
    print("\n== forward correctness ==")
    for M, K, N in [(128, 128, 128), (1024, 768, 512), (4096, 4096, 4096)]:
        A = torch.randn(M, K, device='cuda', dtype=torch.float32)
        B = torch.randn(K, N, device='cuda', dtype=torch.float32)

        C_ours = sgemm(A, B)
        C_ref = A @ B

        # FP32 + N=K=4096 累加误差预期 ~1e-2 量级 (随机数 ~ N(0,1), 累加 K 次有 sqrt(K)*eps)
        atol = max(1e-3, 5e-5 * K)
        ok = torch.allclose(C_ours, C_ref, atol=atol, rtol=1e-3)
        max_err = (C_ours - C_ref).abs().max().item()
        print(f"  M={M} K={K} N={N}  max_err={max_err:.3e}  atol={atol:.1e}  "
              f"{'OK' if ok else 'FAIL'}")
        assert ok, f"forward 误差超标: {max_err}"


def test_autograd():
    print("\n== backward correctness ==")
    M, K, N = 256, 256, 256
    A = torch.randn(M, K, device='cuda', dtype=torch.float32, requires_grad=True)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32, requires_grad=True)

    # 我们的 sgemm
    C_ours = sgemm(A, B)
    loss_ours = C_ours.sum()
    loss_ours.backward()
    gA_ours, gB_ours = A.grad.clone(), B.grad.clone()
    A.grad = None; B.grad = None

    # PyTorch 参考
    C_ref = A @ B
    loss_ref = C_ref.sum()
    loss_ref.backward()
    gA_ref, gB_ref = A.grad.clone(), B.grad.clone()

    eA = (gA_ours - gA_ref).abs().max().item()
    eB = (gB_ours - gB_ref).abs().max().item()
    print(f"  grad_A max_err = {eA:.3e}")
    print(f"  grad_B max_err = {eB:.3e}")
    assert eA < 1e-3 and eB < 1e-3, "backward 误差超标"
    print("  OK")


def bench():
    print("\n== speed vs torch.matmul (4090, fp32) ==")
    M = K = N = 4096
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)

    for _ in range(5):
        sgemm(A, B); A @ B
    torch.cuda.synchronize()

    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    n_iter = 20

    s.record()
    for _ in range(n_iter): sgemm(A, B)
    e.record(); torch.cuda.synchronize()
    ours_ms = s.elapsed_time(e) / n_iter

    s.record()
    for _ in range(n_iter): A @ B
    e.record(); torch.cuda.synchronize()
    torch_ms = s.elapsed_time(e) / n_iter

    gf_ours  = 2 * M * N * K / (ours_ms * 1e-3) / 1e9
    gf_torch = 2 * M * N * K / (torch_ms * 1e-3) / 1e9
    print(f"  ours   : {ours_ms:.2f} ms,  {gf_ours:>7.1f} GFLOPS")
    print(f"  torch  : {torch_ms:.2f} ms,  {gf_torch:>7.1f} GFLOPS  "
          f"(ours / torch = {gf_ours / gf_torch * 100:.1f}%)")


if __name__ == '__main__':
    assert torch.cuda.is_available()
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    test_correctness()
    test_autograd()
    bench()
