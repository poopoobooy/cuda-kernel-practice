// 05_2d_tile.cu —— SGEMM 第 5 版: 2D thread tile + 外积累加
// =================================================================
// 这是手写 SGEMM 能轻松抵达的"性价比顶点", 接下来再想提升就要 ldmatrix /
// TensorCore / async copy / double buffer 这一堆重活, 不在本项目目标内
//
// 几何:
//   BM x BN = 128 x 128       一个 block 算的 C 子块
//   BK = 8                    K 维 tile 宽度
//   TM x TN = 8 x 8           一个 thread 算的 C 子块 (64 个元素)
//   block threads = BM*BN / (TM*TN) = 16384/64 = 256
//
// 每个 K iteration 的 inner loop 是"外积":
//   reg_a[0..TM] = As 一列     (TM 个值)
//   reg_b[0..TN] = Bs 一行     (TN 个值)
//   for i in TM, for j in TN:  acc[i][j] += reg_a[i] * reg_b[j]
//   → 64 个 FMA, 16 个 register load = 4 FMA / load, 算力密度大幅提升
//
// load 阶段每 thread 要搬多块数据:
//   As 总量 = BM*BK = 1024 个 float, 256 thread → 每 thread 4 个
//   Bs 总量 = BK*BN = 1024 个 float, 256 thread → 每 thread 4 个
//
// 性能预期: 4090 fp32 cuBLAS ~31-38 TFLOPS, v5 目标 ~20-25 TFLOPS (60-70% cuBLAS)

#include "common.cuh"

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_2d_tile_kernel(const float* A, const float* B, float* C,
                                     int M, int N, int K) {
    int c_row = blockIdx.y * BM;
    int c_col = blockIdx.x * BN;

    // thread 在 block 内的 (thread_row, thread_col) — 每个 thread 负责 TM x TN 个 C 元素
    // block 共 (BM/TM) x (BN/TN) = 16 x 16 个 thread
    int thread_col = threadIdx.x % (BN / TN);   // 0..15
    int thread_row = threadIdx.x / (BN / TN);   // 0..15

    __shared__ float As[BM * BK];     // 128*8 = 1024 float
    __shared__ float Bs[BK * BN];     // 8*128 = 1024 float

    // load 时每个 thread 在 As 中负责 BM*BK / 256 = 4 个元素 (连续的 1 行内的 4 个？不,
    // 实际是把 As 摊平后按 thread index 分 4 段); Bs 同理.
    // 这里写法是用 stride load: 每 thread 反复 stride 跳着 load 多个元素.
    constexpr int NUM_THREADS = (BM * BN) / (TM * TN);   // 256
    constexpr int A_STRIDE = NUM_THREADS / BK;           // 256/8 = 32 (每次 load 一次能填 As 多少行)
    constexpr int B_STRIDE = NUM_THREADS / BN;           // 256/128 = 2

    // As load: As 形状 BM x BK, threadIdx.x 解成 (row, col)
    int a_inner_row = threadIdx.x / BK;                  // 0..31
    int a_inner_col = threadIdx.x % BK;                  // 0..7
    // Bs load
    int b_inner_row = threadIdx.x / BN;                  // 0..1
    int b_inner_col = threadIdx.x % BN;                  // 0..127

    // 寄存器累加器 TM x TN
    float acc[TM][TN] = {{0.f}};
    float reg_a[TM];
    float reg_b[TN];

    for (int k_tile = 0; k_tile < K; k_tile += BK) {
        // ---- load 1 个 K-tile ----
        // As: 一次 load fills A_STRIDE rows (= 32 行), 重复 BM/A_STRIDE = 4 次 fill 满 128 行
        #pragma unroll
        for (int off = 0; off < BM; off += A_STRIDE) {
            int gA_row = c_row + a_inner_row + off;
            int gA_col = k_tile + a_inner_col;
            As[(a_inner_row + off) * BK + a_inner_col] =
                (gA_row < M && gA_col < K) ? A[gA_row * K + gA_col] : 0.f;
        }
        // Bs: 一次 load fills B_STRIDE rows (= 2 行), 重复 BK/B_STRIDE = 4 次 fill 满 8 行
        #pragma unroll
        for (int off = 0; off < BK; off += B_STRIDE) {
            int gB_row = k_tile + b_inner_row + off;
            int gB_col = c_col + b_inner_col;
            Bs[(b_inner_row + off) * BN + b_inner_col] =
                (gB_row < K && gB_col < N) ? B[gB_row * N + gB_col] : 0.f;
        }
        __syncthreads();

        // ---- compute: 外积累加 ----
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // 1. 从 As 的某一列 (即 As[*][k]) 取 TM 个值到 reg_a
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                reg_a[i] = As[(thread_row * TM + i) * BK + k];
            }
            // 2. 从 Bs 的某一行 (即 Bs[k][*]) 取 TN 个值到 reg_b
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                reg_b[j] = Bs[k * BN + thread_col * TN + j];
            }
            // 3. 外积: TM x TN 个 FMA, 只用了 TM + TN 个 SMEM load
            //    算力密度 = TM*TN / (TM+TN) = 64/16 = 4 FMA/load
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }
        __syncthreads();
    }

    // ---- 写回 C ----
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int row = c_row + thread_row * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int col = c_col + thread_col * TN + j;
            if (row < M && col < N) {
                C[row * N + col] = acc[i][j];
            }
        }
    }
}

void sgemm_2d_tile(const float* dA, const float* dB, float* dC,
                   int M, int N, int K) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    dim3 block((BM * BN) / (TM * TN));         // 256
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_2d_tile_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_LAST_CUDA_ERROR();
}

#ifndef LIB_ONLY
#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
    int M = 4096, N = 4096, K = 4096;
    if (argc > 1) { M = N = K = std::atoi(argv[1]); }
    printf("[05_2d_tile] M=N=K=%d\n", M);

    std::vector<float> hA((size_t)M * K), hB((size_t)K * N);
    fill_random(hA.data(), M * K, 1);
    fill_random(hB.data(), K * N, 2);

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, sizeof(float) * M * K));
    CHECK_CUDA(cudaMalloc(&dB, sizeof(float) * K * N));
    CHECK_CUDA(cudaMalloc(&dC, sizeof(float) * M * N));
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), sizeof(float)*M*K, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), sizeof(float)*K*N, cudaMemcpyHostToDevice));

    for (int i = 0; i < 3; ++i) sgemm_2d_tile(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    CudaTimer t;
    const int n_iter = 5;
    t.start();
    for (int i = 0; i < n_iter; ++i) sgemm_2d_tile(dA, dB, dC, M, N, K);
    float ms = t.stop() / n_iter;
    printf("[05_2d_tile] %.2f ms/iter, %.1f GFLOPS\n", ms, gflops(M, N, K, ms));

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    return 0;
}
#endif
