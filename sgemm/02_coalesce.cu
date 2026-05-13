// 02_coalesce.cu —— SGEMM 第 2 版: 修复 memory coalescing
// =================================================================
// 唯一改动: thread 在 block 内的 row/col 映射换了一下
//   v1: row = threadIdx.x  →  同 warp 的 32 个 thread 对应 32 个不同 row,
//                              访问 A 时跨 K 步 stride，不 coalesce
//   v2: 把 block 改成 1D, 把 col 绑到 threadIdx.x % 32, row 绑到 threadIdx.x / 32
//       →  同 warp 32 个 thread 同 row, col 连续 0~31
//       →  访问 A[row, k] 是 broadcast (1 transaction)
//       →  访问 B[k, col] 是连续 32 个 float = 128B = 一个 cache line, COALESCED
//
// 性能预期: 4090 上 ~1500-2500 GFLOPS, 比 v1 提速 3-5x
//
// 这是最简单也最关键的优化 —— 面试官问"你做过什么 CUDA 优化"时,
// 必讲: 改 thread mapping 让 warp 访存连续，coalesce 后单次访存吞吐 4x+
//
// 注: BS 必须等于 warp size (32) 或其倍数，否则 warp 跨 row, 还是不 coalesce

#include "common.cuh"

__global__ void sgemm_coalesce_kernel(const float* A, const float* B, float* C,
                                      int M, int N, int K) {
    constexpr int BS = 32;
    // 关键: 把 row/col 都从 1D 的 threadIdx.x 解出来,
    //       而且 col 绑 mod 32 这一维 —— 这样 warp 内 32 个 thread 取连续 32 列
    int row = blockIdx.y * BS + (threadIdx.x / BS);
    int col = blockIdx.x * BS + (threadIdx.x % BS);

    if (row >= M || col >= N) return;

    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        // A[row*K + k]: warp 内 row 都一样, k 一样 → 32 thread 读同一 float → broadcast
        // B[k*N + col]: warp 内 k 一样, col = 0..31 连续 → 1 个 128B 事务 = COALESCED
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}

void sgemm_coalesce(const float* dA, const float* dB, float* dC,
                    int M, int N, int K) {
    constexpr int BS = 32;
    dim3 block(BS * BS);                                  // 1D, 1024 threads
    dim3 grid((N + BS - 1) / BS, (M + BS - 1) / BS);
    sgemm_coalesce_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_LAST_CUDA_ERROR();
}

#ifndef LIB_ONLY
#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
    int M = 4096, N = 4096, K = 4096;
    if (argc > 1) { M = N = K = std::atoi(argv[1]); }
    printf("[02_coalesce] M=N=K=%d\n", M);

    std::vector<float> hA((size_t)M * K), hB((size_t)K * N);
    fill_random(hA.data(), M * K, 1);
    fill_random(hB.data(), K * N, 2);

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, sizeof(float) * M * K));
    CHECK_CUDA(cudaMalloc(&dB, sizeof(float) * K * N));
    CHECK_CUDA(cudaMalloc(&dC, sizeof(float) * M * N));
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), sizeof(float)*M*K, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), sizeof(float)*K*N, cudaMemcpyHostToDevice));

    for (int i = 0; i < 3; ++i) sgemm_coalesce(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    CudaTimer t;
    const int n_iter = 5;
    t.start();
    for (int i = 0; i < n_iter; ++i) sgemm_coalesce(dA, dB, dC, M, N, K);
    float ms = t.stop() / n_iter;
    printf("[02_coalesce] %.2f ms/iter, %.1f GFLOPS\n", ms, gflops(M, N, K, ms));

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    return 0;
}
#endif
