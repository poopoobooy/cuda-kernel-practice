// 03_smem.cu —— SGEMM 第 3 版: shared memory tiling
// =================================================================
// 这是 SGEMM 优化里"跨级跳"的一步，从 ~2k GFLOPS 一下能到 ~7-9k GFLOPS
//
// 核心思想:
//   - 每个 block 负责算 C 的 BS x BS 子块 (BS=32)
//   - 这个 C 子块需要 A 的 BS x K 切片和 B 的 K x BS 切片
//   - 我们把 K 维分成 K/BS 个 tile, 每个 tile 加载 BS x BS 的 A 子块和 B 子块到 shared mem
//   - block 内所有 thread 从 SMEM 读 (~100x 快于 HBM)
//
// 算笔账:
//   v2: 每个 thread 读 2K 个 global float (A 一行 K + B 一列 K)
//       block 共 BS*BS = 1024 threads, 总 global load = 1024 * 2K = 2K*1024 个 float
//   v3: 每个 K-tile 全 block 共加载 BS*BS (A) + BS*BS (B) = 2*BS*BS 个 float
//       共 K/BS 个 tile, 总 global load = K/BS * 2*BS*BS = 2*K*BS 个 float
//       而 block 算 BS*BS 个 C 元素 → 每元素 global load = 2*K*BS / (BS*BS) = 2K/BS
//       相比 v2 的 2K, 减少了 BS 倍 = 32x
//
// 两次 __syncthreads():
//   1. load As/Bs → __syncthreads() → 保证 SMEM 写完才 compute
//   2. compute 完 → __syncthreads() → 保证下一轮 load 不覆盖正在被读的 SMEM
//
// 性能预期: 4090 上 ~7000-10000 GFLOPS

#include "common.cuh"

template <int BS>
__global__ void sgemm_smem_kernel(const float* A, const float* B, float* C,
                                  int M, int N, int K) {
    // 这个 block 负责的 C 子块的左上角
    int block_row = blockIdx.y * BS;
    int block_col = blockIdx.x * BS;

    // thread 在 block 内的 (row, col), 与 v2 同样的 mapping 保证 coalesce
    int t_row = threadIdx.x / BS;
    int t_col = threadIdx.x % BS;

    // 全局坐标
    int row = block_row + t_row;
    int col = block_col + t_col;

    __shared__ float As[BS][BS];
    __shared__ float Bs[BS][BS];

    float acc = 0.f;

    // 沿 K 维切 tile
    for (int k_tile = 0; k_tile < K; k_tile += BS) {
        // ---- 阶段 1: 把当前 tile 从 global → shared ----
        // 每个 thread 负责 As / Bs 各一个元素 (因为 block 大小恰好 = BS*BS)
        // As[t_row][t_col] = A[row, k_tile + t_col]
        // Bs[t_row][t_col] = B[k_tile + t_row, col]
        // 这两个 global 访问都是 coalesced (warp 内同 row, t_col 连续 0..31)
        As[t_row][t_col] = A[row * K + (k_tile + t_col)];
        Bs[t_row][t_col] = B[(k_tile + t_row) * N + col];
        __syncthreads();   // 必须等 block 内所有 thread 都 load 完

        // ---- 阶段 2: 在 SMEM 上做 BS 次乘加 ----
        // 注意 Bs[k][t_col]: 同 warp 不同 thread 有不同 t_col,
        // 但 Bs 是 SMEM, 没有 cache line / coalesce 概念,
        // 关键是看 bank conflict —— Bs[k][t_col] t_col 0..31 = 32 个 bank, OK
        // As[t_row][k]: 同 warp t_row 相同, 32 thread 取同一 SMEM 地址 = broadcast, OK
        #pragma unroll
        for (int k = 0; k < BS; ++k) {
            acc += As[t_row][k] * Bs[k][t_col];
        }
        __syncthreads();   // 等所有 thread 算完才能覆盖 SMEM
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

void sgemm_smem(const float* dA, const float* dB, float* dC,
                int M, int N, int K) {
    constexpr int BS = 32;
    dim3 block(BS * BS);                                  // 1D, 1024 threads
    dim3 grid((N + BS - 1) / BS, (M + BS - 1) / BS);
    sgemm_smem_kernel<BS><<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_LAST_CUDA_ERROR();
}

#ifndef LIB_ONLY
#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
    int M = 4096, N = 4096, K = 4096;
    if (argc > 1) { M = N = K = std::atoi(argv[1]); }
    printf("[03_smem] M=N=K=%d\n", M);

    std::vector<float> hA((size_t)M * K), hB((size_t)K * N);
    fill_random(hA.data(), M * K, 1);
    fill_random(hB.data(), K * N, 2);

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, sizeof(float) * M * K));
    CHECK_CUDA(cudaMalloc(&dB, sizeof(float) * K * N));
    CHECK_CUDA(cudaMalloc(&dC, sizeof(float) * M * N));
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), sizeof(float)*M*K, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), sizeof(float)*K*N, cudaMemcpyHostToDevice));

    for (int i = 0; i < 3; ++i) sgemm_smem(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    CudaTimer t;
    const int n_iter = 5;
    t.start();
    for (int i = 0; i < n_iter; ++i) sgemm_smem(dA, dB, dC, M, N, K);
    float ms = t.stop() / n_iter;
    printf("[03_smem] %.2f ms/iter, %.1f GFLOPS\n", ms, gflops(M, N, K, ms));

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    return 0;
}
#endif
