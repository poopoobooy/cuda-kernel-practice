// 06_wmma_fp16.cu —— SGEMM 第 6 版: FP16 + Ada 4th-gen TensorCore (WMMA)
// =================================================================
// 这一版换 dtype + 用 TensorCore. 之前 1-5 版都是 FP32 走 FMA,
// 4090 Laptop FP32 算力天花板就 30 TFLOPS, FP16 走 TC 是 165 TFLOPS (5.5x)
//
// 概念地图:
//   - mma.sync (PTX 指令): 一个 warp 做一次 16x16x16 (M x N x K) 的 D = A*B+C, 16 元素 / lane
//   - wmma:: C++ API: 把 mma.sync 包成 fragment-load-compute-store, 不用写 PTX
//   - 一个 warp 用多个 mma 拼出更大的 WM x WN 子块 (这里 WM=WN=64, 4x4 fragment)
//   - 一个 block 用多个 warp 拼出 BM x BN (这里 BM=BN=128, 2x2 warp)
//
// 几何 (跟 CUTLASS 的命名对齐, 方便面试讲):
//   BM x BN = 128 x 128       block tile
//   BK = 32                   K-tile width
//   WM x WN = 64 x 64         warp tile (一个 warp 算 64x64 元素)
//   MMA_M = MMA_N = MMA_K = 16  WMMA 原子粒度 (Ada FP16 唯一推荐尺寸)
//
//   每 warp 的 fragment 数 = (WM/MMA_M) x (WN/MMA_N) = 4 x 4 = 16
//   每 block warp 数 = (BM/WM) x (BN/WN) = 2 x 2 = 4 warp = 128 thread
//
// 性能预期: 4090 Laptop FP16 TC 峰值 ~165 TFLOPS, 目标 100+ TFLOPS (60%+)
//           跟 FP32 v5 (19.5 TFLOPS) 比是 ~5x, 跟 cuBLAS FP32 24.5 TFLOPS 比也 4x+
//
// 编译:
//   nvcc -O3 -arch=sm_89 --use_fast_math -lcublas -o 06_wmma_fp16.exe 06_wmma_fp16.cu

#include "common.cuh"
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>
#include <vector>

using namespace nvcuda;
using namespace nvcuda::wmma;

// FP16 in / FP32 out (累加用 FP32 保数值精度, 这是工业标准)
using half_t = __half;
using AccT = float;

constexpr int MMA_M = 16, MMA_N = 16, MMA_K = 16;
constexpr int WM = 64, WN = 64;
constexpr int BM = 128, BN = 128, BK = 32;
constexpr int N_WARP_M = BM / WM;       // 2
constexpr int N_WARP_N = BN / WN;       // 2
constexpr int N_WARPS = N_WARP_M * N_WARP_N;   // 4
constexpr int N_FRAG_M = WM / MMA_M;    // 4
constexpr int N_FRAG_N = WN / MMA_N;    // 4
constexpr int N_THREADS = N_WARPS * 32;        // 128

__global__ void sgemm_wmma_fp16_kernel(const half_t* A, const half_t* B, AccT* C,
                                       int M, int N, int K) {
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    // warp id & warp 在 block 内的 (warp_row, warp_col) 二维坐标
    int warp_id = threadIdx.x / 32;
    int warp_row = warp_id / N_WARP_N;        // 0..N_WARP_M-1
    int warp_col = warp_id % N_WARP_N;        // 0..N_WARP_N-1
    int lane = threadIdx.x % 32;

    // 共享内存: 一个 K-tile 的 As/Bs
    // 注意 padding +8 防 bank conflict (FP16 -> 2B, 8 个 fp16 = 16B = 一个 bank)
    __shared__ half_t As[BM][BK + 8];
    __shared__ half_t Bs[BK][BN + 8];

    // ---- accumulator fragments (FP32 累加) ----
    fragment<accumulator, MMA_M, MMA_N, MMA_K, AccT> acc[N_FRAG_M][N_FRAG_N];
    #pragma unroll
    for (int i = 0; i < N_FRAG_M; ++i)
        #pragma unroll
        for (int j = 0; j < N_FRAG_N; ++j)
            fill_fragment(acc[i][j], 0.0f);

    // ---- 加载 A/B 全局到 SMEM 的角色分配 ----
    // As 形状 BM x BK = 128 x 32 = 4096 个 half = 8192 B, N_THREADS=128, 每 thread 32 个 half
    // Bs 形状 BK x BN = 32 x 128 = 4096 个 half 同理
    // 这里用最朴素 stride load (每 thread 一次搬 8 个 half, 走 128B 对齐)
    constexpr int VEC = 8;    // 8 个 half = 16B, 单次 LD 最优带宽
    constexpr int N_VEC = (BM * BK) / VEC;          // 4096/8 = 512 个 vec
    constexpr int VEC_PER_THREAD = N_VEC / N_THREADS;  // 512/128 = 4

    for (int k_tile = 0; k_tile < K; k_tile += BK) {
        // ---- 阶段 1: load As ----
        // 把 As 的 4096 个 half 摊平后按 thread index 分段, 每 thread 处理 32 个
        #pragma unroll
        for (int it = 0; it < VEC_PER_THREAD; ++it) {
            int vec_idx = it * N_THREADS + threadIdx.x;        // 0..511
            int row = vec_idx / (BK / VEC);                     // 0..127
            int col = (vec_idx % (BK / VEC)) * VEC;             // 0,8,16,24
            int g_row = block_row + row;
            int g_col = k_tile + col;
            int4 v;
            if (g_row < M && g_col < K) {
                v = *reinterpret_cast<const int4*>(&A[g_row * K + g_col]);
            } else {
                v = make_int4(0, 0, 0, 0);
            }
            *reinterpret_cast<int4*>(&As[row][col]) = v;
        }
        // ---- 阶段 1b: load Bs ----
        #pragma unroll
        for (int it = 0; it < VEC_PER_THREAD; ++it) {
            int vec_idx = it * N_THREADS + threadIdx.x;
            int row = vec_idx / (BN / VEC);                     // 0..31
            int col = (vec_idx % (BN / VEC)) * VEC;             // 0,8,16,...120
            int g_row = k_tile + row;
            int g_col = block_col + col;
            int4 v;
            if (g_row < K && g_col < N) {
                v = *reinterpret_cast<const int4*>(&B[g_row * N + g_col]);
            } else {
                v = make_int4(0, 0, 0, 0);
            }
            *reinterpret_cast<int4*>(&Bs[row][col]) = v;
        }
        __syncthreads();

        // ---- 阶段 2: WMMA compute ----
        // 沿 BK 维分 BK/MMA_K = 32/16 = 2 个 mma 步
        #pragma unroll
        for (int k_step = 0; k_step < BK / MMA_K; ++k_step) {
            // load A fragments: 一个 warp 的 4 个 16x16 行 fragment
            fragment<matrix_a, MMA_M, MMA_N, MMA_K, half_t, row_major> a_frag[N_FRAG_M];
            fragment<matrix_b, MMA_M, MMA_N, MMA_K, half_t, row_major> b_frag[N_FRAG_N];

            #pragma unroll
            for (int i = 0; i < N_FRAG_M; ++i) {
                int row = warp_row * WM + i * MMA_M;
                load_matrix_sync(a_frag[i], &As[row][k_step * MMA_K], BK + 8);
            }
            #pragma unroll
            for (int j = 0; j < N_FRAG_N; ++j) {
                int col = warp_col * WN + j * MMA_N;
                load_matrix_sync(b_frag[j], &Bs[k_step * MMA_K][col], BN + 8);
            }

            // 16x16 外积: 4 * 4 = 16 个 mma.sync, 每个做 16x16x16 共 4096 FMA
            // 一个 warp 单 K-step 做 16*4096 = 65536 FMA
            #pragma unroll
            for (int i = 0; i < N_FRAG_M; ++i) {
                #pragma unroll
                for (int j = 0; j < N_FRAG_N; ++j) {
                    mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

    // ---- 写回 C (FP32) ----
    #pragma unroll
    for (int i = 0; i < N_FRAG_M; ++i) {
        #pragma unroll
        for (int j = 0; j < N_FRAG_N; ++j) {
            int row = block_row + warp_row * WM + i * MMA_M;
            int col = block_col + warp_col * WN + j * MMA_N;
            if (row < M && col < N) {
                store_matrix_sync(&C[row * N + col], acc[i][j], N, mem_row_major);
            }
        }
    }
}

void sgemm_wmma_fp16(const half_t* dA, const half_t* dB, AccT* dC,
                     int M, int N, int K) {
    dim3 block(N_THREADS);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_wmma_fp16_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_LAST_CUDA_ERROR();
}

// ===========================================================================
// main: bench vs cuBLAS GemmEx (FP16 in, FP32 out)
// ===========================================================================
int main(int argc, char** argv) {
    int M = 4096, N = 4096, K = 4096;
    if (argc > 1) { M = N = K = std::atoi(argv[1]); }
    printf("\n=== SGEMM WMMA FP16 -> FP32: M=N=K=%d, RTX 4090 Laptop (sm_89) ===\n", M);

    // host alloc
    std::vector<float> hA_f((size_t)M * K), hB_f((size_t)K * N);
    fill_random(hA_f.data(), M * K, 1);
    fill_random(hB_f.data(), K * N, 2);

    // 转 fp16
    std::vector<half_t> hA((size_t)M * K), hB((size_t)K * N);
    for (size_t i = 0; i < hA.size(); ++i) hA[i] = __float2half(hA_f[i]);
    for (size_t i = 0; i < hB.size(); ++i) hB[i] = __float2half(hB_f[i]);

    half_t *dA, *dB;
    float *dC, *dC_ref;
    CHECK_CUDA(cudaMalloc(&dA, sizeof(half_t) * M * K));
    CHECK_CUDA(cudaMalloc(&dB, sizeof(half_t) * K * N));
    CHECK_CUDA(cudaMalloc(&dC, sizeof(float)  * M * N));
    CHECK_CUDA(cudaMalloc(&dC_ref, sizeof(float) * M * N));
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), sizeof(half_t)*M*K, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), sizeof(half_t)*K*N, cudaMemcpyHostToDevice));

    // ---- 跑我们的 kernel ----
    for (int i = 0; i < 5; ++i) sgemm_wmma_fp16(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    // ---- 跑 cuBLAS (FP16 in, FP32 out) ----
    cublasHandle_t h;
    cublasCreate(&h);
    // 用 cublasGemmEx: A/B fp16, C fp32, 累加 fp32, 算法 DEFAULT_TENSOR_OP 强制走 TC
    const float alpha = 1.f, beta = 0.f;
    auto cublas_call = [&]() {
        // row-major -> 用 C^T = B^T A^T 技巧
        cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N,
                     N, M, K,
                     &alpha,
                     dB, CUDA_R_16F, N,
                     dA, CUDA_R_16F, K,
                     &beta,
                     dC_ref, CUDA_R_32F, N,
                     CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    };
    for (int i = 0; i < 5; ++i) cublas_call();
    CHECK_CUDA(cudaDeviceSynchronize());

    // ---- 正确性校验 ----
    std::vector<float> hC((size_t)M * N), hRef((size_t)M * N);
    CHECK_CUDA(cudaMemcpy(hC.data(),   dC,    sizeof(float)*M*N, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hRef.data(), dC_ref, sizeof(float)*M*N, cudaMemcpyDeviceToHost));
    float err = max_abs_diff(hC.data(), hRef.data(), M * N);
    // FP16 in 累加 K=4096 次, 误差预期 ~1e-1 量级 (FP16 eps=2e-3, sqrt(K)*eps ~0.13)
    printf("max_abs_diff vs cuBLAS TC = %.3e\n", err);

    // ---- bench ----
    CudaTimer t;
    const int n_iter = 20;
    t.start();
    for (int i = 0; i < n_iter; ++i) sgemm_wmma_fp16(dA, dB, dC, M, N, K);
    float ms_ours = t.stop() / n_iter;

    t.start();
    for (int i = 0; i < n_iter; ++i) cublas_call();
    float ms_cublas = t.stop() / n_iter;

    double gf_ours   = gflops(M, N, K, ms_ours);
    double gf_cublas = gflops(M, N, K, ms_cublas);
    printf("[ours WMMA]    %.3f ms, %.1f GFLOPS = %.2f TFLOPS\n",
           ms_ours,   gf_ours,   gf_ours / 1000.0);
    printf("[cuBLAS TC]    %.3f ms, %.1f GFLOPS = %.2f TFLOPS (100%% baseline)\n",
           ms_cublas, gf_cublas, gf_cublas / 1000.0);
    printf("ours / cuBLAS = %.1f%%\n", gf_ours / gf_cublas * 100.0);

    cublasDestroy(h);
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    CHECK_CUDA(cudaFree(dC_ref));
    return 0;
}
