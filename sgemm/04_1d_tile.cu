// 04_1d_tile.cu —— SGEMM 第 4 版: 1D thread tile + register 复用
// =================================================================
// 在 v3 的 SMEM tiling 基础上更进一步:
//   每个 thread 不再只算 C 的 1 个元素, 而是算 1 列 TM=8 个元素
//   → block 用更少的 thread (512 而不是 4096) 算同样大的 C 子块
//   → 每个 thread 算 8 个 C 元素时, Bs 的某个值能在 register 里复用 8 次
//
// 几何尺寸:
//   BM x BN = 64 x 64    一个 block 算的 C 子块
//   BK = 8               K 维 tile 宽度
//   TM = 8               一个 thread 算 TM 个 C 元素 (column 方向)
//   block threads = BM*BN / TM = 4096/8 = 512
//
// 内层数学:
//   for k in 0..BK:
//       b_reg = Bs[k][thread_col]              # 1 个值, load 1 次
//       for tm in 0..TM:                       # unroll
//           acc[tm] += As[thread_row*TM + tm][k] * b_reg
//
// 性能预期: 4090 上 ~15000-18000 GFLOPS

#include "common.cuh"

template <int BM, int BN, int BK, int TM>
__global__ void sgemm_1d_tile_kernel(const float* A, const float* B, float* C,
                                     int M, int N, int K) {
    // block 负责的 C 子块的左上角
    int c_row = blockIdx.y * BM;
    int c_col = blockIdx.x * BN;

    // 每个 thread 在 block 内的 (thread_row, thread_col):
    // 注意现在 thread 在 N 维 (col) 仍是 0..BN-1, 但在 M 维只占 BM/TM = 8 个槽
    int thread_col = threadIdx.x % BN;                  // 0..63
    int thread_row = threadIdx.x / BN;                  // 0..7   (因为 BM/TM = 64/8 = 8)

    __shared__ float As[BM * BK];     // 64*8 = 512 float = 2 KB
    __shared__ float Bs[BK * BN];     // 8*64 = 512 float = 2 KB

    // 每个 thread 在 K-tile 内的 load 角色:
    // As 是 BM x BK = 64x8, 共 512 个 float, 正好 = block 的 thread 数 (512), 每 thread 1 个
    int a_inner_row = threadIdx.x / BK;     // 0..63
    int a_inner_col = threadIdx.x % BK;     // 0..7
    int b_inner_row = threadIdx.x / BN;     // 0..7
    int b_inner_col = threadIdx.x % BN;     // 0..63

    // register tile: TM 个 累加器
    float acc[TM] = {0.f};

    for (int k_tile = 0; k_tile < K; k_tile += BK) {
        // ---- load 1 个 K-tile ----
        // 关键: 这里 As 的 load 不一定 coalesced (取决于 BK), 但 BM*BK 总量小,
        //       重要的是 Bs 的 load 一定要 coalesced (b_inner_col 跨连续 thread, 触发 coalesce)
        As[a_inner_row * BK + a_inner_col] = A[(c_row + a_inner_row) * K + (k_tile + a_inner_col)];
        Bs[b_inner_row * BN + b_inner_col] = B[(k_tile + b_inner_row) * N + (c_col + b_inner_col)];
        __syncthreads();

        // ---- compute: 内层 K-loop, 每个 thread 算 TM 个 C 元素 ----
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // Bs 的一个值 (整个 thread 列共用), 取出来放 register 里, 后面复用 TM 次
            float b_reg = Bs[k * BN + thread_col];
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm) {
                acc[tm] += As[(thread_row * TM + tm) * BK + k] * b_reg;
            }
        }
        __syncthreads();
    }

    // ---- 写回 C ----
    #pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
        int row = c_row + thread_row * TM + tm;
        int col = c_col + thread_col;
        if (row < M && col < N) {
            C[row * N + col] = acc[tm];
        }
    }
}

void sgemm_1d_tile(const float* dA, const float* dB, float* dC,
                   int M, int N, int K) {
    constexpr int BM = 64, BN = 64, BK = 8, TM = 8;
    dim3 block((BM * BN) / TM);              // 512
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_1d_tile_kernel<BM, BN, BK, TM><<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_LAST_CUDA_ERROR();
}

#ifndef LIB_ONLY
#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
    int M = 4096, N = 4096, K = 4096;
    if (argc > 1) { M = N = K = std::atoi(argv[1]); }
    printf("[04_1d_tile] M=N=K=%d\n", M);

    std::vector<float> hA((size_t)M * K), hB((size_t)K * N);
    fill_random(hA.data(), M * K, 1);
    fill_random(hB.data(), K * N, 2);

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, sizeof(float) * M * K));
    CHECK_CUDA(cudaMalloc(&dB, sizeof(float) * K * N));
    CHECK_CUDA(cudaMalloc(&dC, sizeof(float) * M * N));
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), sizeof(float)*M*K, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), sizeof(float)*K*N, cudaMemcpyHostToDevice));

    for (int i = 0; i < 3; ++i) sgemm_1d_tile(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    CudaTimer t;
    const int n_iter = 5;
    t.start();
    for (int i = 0; i < n_iter; ++i) sgemm_1d_tile(dA, dB, dC, M, N, K);
    float ms = t.stop() / n_iter;
    printf("[04_1d_tile] %.2f ms/iter, %.1f GFLOPS\n", ms, gflops(M, N, K, ms));

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    return 0;
}
#endif
