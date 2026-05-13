// 01_naive.cu —— SGEMM 第 1 版: 每个 thread 算 C 一个元素
// =================================================================
// 性能预期: 在 RTX 4090 上 FP32 M=N=K=4096 大概 ~300-500 GFLOPS
//           (cuBLAS 是 ~30000+ GFLOPS，差 ~60x)
//
// 为啥慢? 面试时这是 ECHO 答案:
//   1. 没用 shared memory: 每个 thread 从 global memory load 2K 个 float
//      (A 的一行 K 个 + B 的一列 K 个)，K=4096 时单 thread 8K 次 global load
//   2. memory NOT coalesced (这一版做了 2D block，threadIdx.x 沿 M 维 = A 的行索引):
//      - 同一 warp 的 32 个 thread, threadIdx.x 取 0~31, 它们的 col_b = blockIdx.x*BS + ty 是同一列
//        于是 32 个 thread 同时访问 B[k, col_b] —— 同一个值，访问 B 是 broadcast (OK)
//      - 但 A[row, k] 中 row = blockIdx.y*BS + tx, 32 个 tx 取 0~31, row 是连续的, 但 k 相同
//        => 访问 A 是访问"同一列"的 32 个不同行 —— 跨 N 步 stride, 完全 NOT coalesced
//      - 这是经典反面教材，下一版 02_coalesce.cu 把 tx/ty 角色对换就能修
//
// 编译:  nvcc -O3 -arch=sm_89 -o 01_naive.exe 01_naive.cu
// 运行:  ./01_naive.exe  (M=N=K=4096 默认值, 见 main)

#include "common.cuh"

// kernel: 每个 thread 算 C[row, col] = sum_{k} A[row, k] * B[k, col]
__global__ void sgemm_naive_kernel(const float* A, const float* B, float* C,
                                   int M, int N, int K) {
    // 每个 thread 一个 C 元素;
    // 这里故意把 row 绑到 threadIdx.x (后面会发现这是错的)
    int row = blockIdx.y * blockDim.y + threadIdx.x;
    int col = blockIdx.x * blockDim.x + threadIdx.y;

    if (row >= M || col >= N) return;  // 边界保护

    float acc = 0.f;
    // 内层 K 维 loop —— 这里就是 SGEMM 的核心 2*K flops/element
    for (int k = 0; k < K; ++k) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}

// host 入口: 假设 A/B/C 都是 row-major, A[M,K] B[K,N] C[M,N]
void sgemm_naive(const float* dA, const float* dB, float* dC,
                 int M, int N, int K) {
    constexpr int BS = 32;
    dim3 block(BS, BS);  // 32*32 = 1024, 等于一个 SM 一个 block 的 thread 上限
    dim3 grid((N + BS - 1) / BS, (M + BS - 1) / BS);
    sgemm_naive_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_LAST_CUDA_ERROR();
}

// ------- main: 单独运行 + 报告性能 -------
#ifndef LIB_ONLY
#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
    int M = 4096, N = 4096, K = 4096;
    if (argc > 1) { M = N = K = std::atoi(argv[1]); }
    printf("[01_naive] M=N=K=%d\n", M);

    // host alloc + init
    std::vector<float> hA((size_t)M * K), hB((size_t)K * N), hC((size_t)M * N);
    fill_random(hA.data(), M * K, 1);
    fill_random(hB.data(), K * N, 2);

    // device alloc
    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, sizeof(float) * M * K));
    CHECK_CUDA(cudaMalloc(&dB, sizeof(float) * K * N));
    CHECK_CUDA(cudaMalloc(&dC, sizeof(float) * M * N));
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), sizeof(float)*M*K, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), sizeof(float)*K*N, cudaMemcpyHostToDevice));

    // warmup
    for (int i = 0; i < 3; ++i) sgemm_naive(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    // bench
    CudaTimer t;
    const int n_iter = 5;
    t.start();
    for (int i = 0; i < n_iter; ++i) sgemm_naive(dA, dB, dC, M, N, K);
    float ms = t.stop() / n_iter;
    double g = gflops(M, N, K, ms);
    printf("[01_naive] %.2f ms/iter, %.1f GFLOPS\n", ms, g);

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    return 0;
}
#endif
