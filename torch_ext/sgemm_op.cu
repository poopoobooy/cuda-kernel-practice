// sgemm_op.cu —— 把手写 SGEMM v5 包装成 PyTorch CUDA extension
// =================================================================
// 暴露给 Python 的接口: sgemm_forward(A, B) -> C
// Python 端用 torch.autograd.Function 写反向 (用 torch.matmul 复用 cuBLAS)
//
// 这一步主要展示:
//   1. 怎么从 torch::Tensor 取裸指针 + 校验 contiguous / cuda / fp32
//   2. 怎么 pybind11 暴露给 Python
//   3. 怎么和 PyTorch 的 autograd 链路对接

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

// ===== 直接 copy v5 的 kernel 进来 (避免链接 sgemm/05_2d_tile.cu) =====
template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_2d_tile_kernel(const float* A, const float* B, float* C,
                                     int M, int N, int K) {
    int c_row = blockIdx.y * BM;
    int c_col = blockIdx.x * BN;
    int thread_col = threadIdx.x % (BN / TN);
    int thread_row = threadIdx.x / (BN / TN);

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    constexpr int NUM_THREADS = (BM * BN) / (TM * TN);
    constexpr int A_STRIDE = NUM_THREADS / BK;
    constexpr int B_STRIDE = NUM_THREADS / BN;

    int a_inner_row = threadIdx.x / BK;
    int a_inner_col = threadIdx.x % BK;
    int b_inner_row = threadIdx.x / BN;
    int b_inner_col = threadIdx.x % BN;

    float acc[TM][TN] = {{0.f}};
    float reg_a[TM];
    float reg_b[TN];

    for (int k_tile = 0; k_tile < K; k_tile += BK) {
        #pragma unroll
        for (int off = 0; off < BM; off += A_STRIDE) {
            int gA_row = c_row + a_inner_row + off;
            int gA_col = k_tile + a_inner_col;
            As[(a_inner_row + off) * BK + a_inner_col] =
                (gA_row < M && gA_col < K) ? A[gA_row * K + gA_col] : 0.f;
        }
        #pragma unroll
        for (int off = 0; off < BK; off += B_STRIDE) {
            int gB_row = k_tile + b_inner_row + off;
            int gB_col = c_col + b_inner_col;
            Bs[(b_inner_row + off) * BN + b_inner_col] =
                (gB_row < K && gB_col < N) ? B[gB_row * N + gB_col] : 0.f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                reg_a[i] = As[(thread_row * TM + i) * BK + k];
            #pragma unroll
            for (int j = 0; j < TN; ++j)
                reg_b[j] = Bs[k * BN + thread_col * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += reg_a[i] * reg_b[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int row = c_row + thread_row * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int col = c_col + thread_col * TN + j;
            if (row < M && col < N)
                C[row * N + col] = acc[i][j];
        }
    }
}

// ===== C++ 包装: tensor 校验 + kernel launch =====
torch::Tensor sgemm_forward(torch::Tensor A, torch::Tensor B) {
    // 入参校验 (面试官会问"如果输入不连续怎么办")
    TORCH_CHECK(A.is_cuda(), "A must be on CUDA");
    TORCH_CHECK(B.is_cuda(), "B must be on CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be fp32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be fp32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A, B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "K dim mismatch");

    // 确保 contiguous —— 如果传进来的是 view/slice 出来的不连续 tensor, 这里 copy
    A = A.contiguous();
    B = B.contiguous();

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    // launch v5
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    dim3 block((BM * BN) / (TM * TN));
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    // 用 PyTorch 当前 stream, 否则会和 torch 主流串行混乱
    auto stream = at::cuda::getCurrentCUDAStream();
    sgemm_2d_tile_kernel<BM, BN, BK, TM, TN><<<grid, block, 0, stream.stream()>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, N, K);

    return C;
}

// pybind11 暴露
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sgemm_forward", &sgemm_forward,
          "Custom SGEMM forward (FP32 row-major)",
          py::arg("A"), py::arg("B"));
}
