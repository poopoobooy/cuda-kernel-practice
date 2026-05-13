// fa1_fwd.cu —— Flash Attention 1, forward 教学版 (FP32, head_dim = 64)
// =================================================================
// 这一版照搬 FlashAttention paper (Dao et al., 2022) 的 Algorithm 1,
// 用 "block-tiled + online softmax" 把标准 attention 从 O(N^2) memory
// 砍到 O(N) memory, 同时把 HBM 访问从 O(N^2) 砍到 O(N^2 * d / sqrt(SRAM)).
//
// 标准 attention 的痛点:
//   1. 算 P = softmax(Q @ K^T / sqrt(d)) 要把整张 N x N 的 score matrix 写回 HBM,
//      4096 x 4096 x 4B = 64 MB, 对 N=8192 直接 256 MB 写 HBM, 带宽全砸在这.
//   2. O(N^2) 的 memory 拖慢一切, 还限制了 batch / seq len 上限.
//
// Flash Attention 思路:
//   - 把 Q 分成 Br 行的块, 把 K/V 分成 Bc 行的块
//   - 外层循环: 遍历 Q 的块
//   - 内层循环: 对当前 Q 块, 遍历所有 K/V 块, online 维护
//     (running max m_i, running sum l_i, running output O_i)
//   - 关键 trick: 看到新的 m_new 时, OLD 的 (l_i, O_i) 都要乘 exp(m_old - m_new) 修正
//   - 最后 O_i /= l_i 才是真正的 softmax-weighted output
//
// 这就实现了 "softmax 永不 materialize 完整 N x N 的 P 矩阵" — N x N 只活在 register
//
// 维度选择 (4090 Laptop):
//   d = 64 (一个 attention head 的标准 dim)
//   BR = 64 (一个 block 处理 64 行 Q, 一 thread 一行)
//   BC = 64 (内层扫 64 行 K/V)
//   一个 block 64 threads, 4 warps
//
// Shared memory: K_smem + V_smem = 2 * BC * d * 4 = 2 * 64 * 64 * 4 = 32 KB / block
//
// 编译:
//   nvcc -O3 -arch=sm_89 --use_fast_math -o flash_attn\fa1_fwd.exe flash_attn\fa1_fwd.cu

#include "../sgemm/common.cuh"
#include <vector>
#include <cmath>
#include <algorithm>

constexpr int D_HEAD = 64;
constexpr int BR = 64;
constexpr int BC = 64;
constexpr int N_THREADS = BR;        // 一个 thread 处理一行 Q

// ===========================================================================
// Flash Attention 1 forward kernel
// 网格: grid(N/BR, B*H)  block(BR)
// 输入: Q, K, V 形状 (B*H, N, d) 行主序
// 输出: O 形状 (B*H, N, d)
// ===========================================================================
__global__ void fa1_fwd_kernel(const float* __restrict__ Q,
                               const float* __restrict__ K,
                               const float* __restrict__ V,
                               float* __restrict__ O,
                               int N, float scale) {
    int row_block = blockIdx.x;       // Q 行块下标
    int bh = blockIdx.y;              // batch * head 平铺下标
    int tid = threadIdx.x;            // 0..BR-1

    int q_row = row_block * BR + tid; // 本 thread 负责的 Q 行 (global)

    // (b,h) slice 的起始指针
    int bh_offset = bh * N * D_HEAD;
    const float* Q_bh = Q + bh_offset;
    const float* K_bh = K + bh_offset;
    const float* V_bh = V + bh_offset;
    float*       O_bh = O + bh_offset;

    // 寄存器: 我这行 Q (在整个 kernel 复用), 在线累加器 O (FP32)
    float q[D_HEAD];
    float o[D_HEAD];
    float m_i = -INFINITY;     // running max
    float l_i = 0.f;            // running sum

    // 加载我的那行 Q 到 reg
    #pragma unroll
    for (int i = 0; i < D_HEAD; ++i) {
        o[i] = 0.f;
        q[i] = (q_row < N) ? Q_bh[q_row * D_HEAD + i] : 0.f;
    }

    // 一次性把 Q 乘上 scale = 1/sqrt(d), 后面 S = q @ K^T 就直接是 scaled score
    #pragma unroll
    for (int i = 0; i < D_HEAD; ++i) q[i] *= scale;

    // SMEM 缓存 K/V 当前块
    __shared__ float K_smem[BC][D_HEAD];
    __shared__ float V_smem[BC][D_HEAD];

    // ---- 外层: 扫所有 K/V 块 ----
    for (int j_block = 0; j_block < N; j_block += BC) {
        // 协同加载 K[j_block:j_block+BC, :] 和 V[...] 到 SMEM
        // 一共 BC * d = 4096 个 float, BR=64 threads, 每 thread 64 个
        #pragma unroll
        for (int it = 0; it < (BC * D_HEAD) / BR; ++it) {
            int idx = it * BR + tid;                 // 0..BC*D_HEAD-1
            int r = idx / D_HEAD;
            int c = idx % D_HEAD;
            int g_row = j_block + r;
            bool inb = (g_row < N);
            K_smem[r][c] = inb ? K_bh[g_row * D_HEAD + c] : 0.f;
            V_smem[r][c] = inb ? V_bh[g_row * D_HEAD + c] : 0.f;
        }
        __syncthreads();

        // ---- 算 S[my_row][0..BC-1] = q @ K_smem^T ----
        // 这里 K_smem 在 SMEM, q 在寄存器, 每个 thread 算 BC 个 dot product
        // FMA count: BC * D_HEAD = 4096 per thread per inner iter
        float s[BC];
        #pragma unroll
        for (int j = 0; j < BC; ++j) {
            float acc = 0.f;
            #pragma unroll
            for (int k = 0; k < D_HEAD; ++k) {
                acc += q[k] * K_smem[j][k];
            }
            s[j] = acc;
            // 出界 mask (j_block + j 越界的位置)
            int kv_row = j_block + j;
            if (kv_row >= N) s[j] = -INFINITY;
        }

        // ---- 本块 softmax 三件套: max, exp, sum ----
        float m_local = s[0];
        #pragma unroll
        for (int j = 1; j < BC; ++j) m_local = fmaxf(m_local, s[j]);

        float l_local = 0.f;
        #pragma unroll
        for (int j = 0; j < BC; ++j) {
            // 用 __expf 走 fast intrinsic. softmax 误差影响小, 性能差距大.
            s[j] = __expf(s[j] - m_local);
            l_local += s[j];
        }

        // ---- online softmax 更新 ----
        // m_new = max(m_old, m_local)
        // alpha = exp(m_old - m_new)        OLD 的修正系数
        // beta  = exp(m_local - m_new)      新块的修正系数
        // 之所以这么搞: 我们要的 sum 是 sum_j exp(s_j - m_new), 把 OLD 那部分
        // exp(s_j - m_old) * exp(m_old - m_new) 即可对齐到新 m_new
        float m_new = fmaxf(m_i, m_local);
        float alpha = __expf(m_i - m_new);
        float beta = __expf(m_local - m_new);
        float l_new = alpha * l_i + beta * l_local;

        // ---- 更新 O_i = alpha * O_i + beta * (P_local @ V_smem) ----
        // 注意: 这里 "P_local @ V_smem" 不显式求 N x N 的 P, 直接用 s[j] 当 P
        // 每 thread 算 d 个 output 元素, 每个 = beta * sum_j s[j] * V_smem[j][k]
        #pragma unroll
        for (int k = 0; k < D_HEAD; ++k) {
            float acc = 0.f;
            #pragma unroll
            for (int j = 0; j < BC; ++j) {
                acc += s[j] * V_smem[j][k];
            }
            o[k] = alpha * o[k] + beta * acc;
        }

        m_i = m_new;
        l_i = l_new;
        __syncthreads();
    }

    // ---- 归一化: O / l_i ----
    if (q_row < N) {
        float inv_l = 1.f / l_i;
        #pragma unroll
        for (int k = 0; k < D_HEAD; ++k) {
            O_bh[q_row * D_HEAD + k] = o[k] * inv_l;
        }
    }
}

void launch_fa1_fwd(const float* dQ, const float* dK, const float* dV, float* dO,
                    int B, int H, int N, float scale) {
    dim3 grid((N + BR - 1) / BR, B * H);
    dim3 block(N_THREADS);
    fa1_fwd_kernel<<<grid, block>>>(dQ, dK, dV, dO, N, scale);
    CHECK_LAST_CUDA_ERROR();
}

// ===========================================================================
// 朴素 attention 对照 kernel (拿来验证 FA1 正确 + 性能对照)
// 显式生成 N x N 的 P, 完全 O(N^2) memory.
// ===========================================================================
__global__ void naive_attn_score(const float* Q, const float* K, float* S,
                                 int N, float scale) {
    // grid (N/32, N/32, B*H), block (32, 32)
    int bh = blockIdx.z;
    int row = blockIdx.y * 32 + threadIdx.y;
    int col = blockIdx.x * 32 + threadIdx.x;
    if (row >= N || col >= N) return;
    int bh_off = bh * N * D_HEAD;
    float acc = 0.f;
    #pragma unroll
    for (int k = 0; k < D_HEAD; ++k) {
        acc += Q[bh_off + row * D_HEAD + k] * K[bh_off + col * D_HEAD + k];
    }
    S[bh * N * N + row * N + col] = acc * scale;
}

__global__ void naive_attn_softmax(float* S, int N) {
    // 每 block 处理一行 softmax
    int bh = blockIdx.y;
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int row_off = bh * N * N + row * N;

    // 求 max (block reduce: warp reduce -> SMEM -> single-warp reduce)
    int lane = tid & 31;
    int wid  = tid >> 5;
    int n_warps = blockDim.x / 32;
    __shared__ float warp_buf[8];   // up to 8 warps (block <=256)
    __shared__ float smax;
    __shared__ float ssum;

    float m = -INFINITY;
    for (int j = tid; j < N; j += blockDim.x) m = fmaxf(m, S[row_off + j]);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    if (lane == 0) warp_buf[wid] = m;
    __syncthreads();
    if (wid == 0) {
        m = (lane < n_warps) ? warp_buf[lane] : -INFINITY;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
        if (lane == 0) smax = m;
    }
    __syncthreads();

    float s = 0.f;
    for (int j = tid; j < N; j += blockDim.x) {
        float v = __expf(S[row_off + j] - smax);
        S[row_off + j] = v;
        s += v;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    if (lane == 0) warp_buf[wid] = s;
    __syncthreads();
    if (wid == 0) {
        s = (lane < n_warps) ? warp_buf[lane] : 0.f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
        if (lane == 0) ssum = s;
    }
    __syncthreads();

    float inv = 1.f / ssum;
    for (int j = tid; j < N; j += blockDim.x) S[row_off + j] *= inv;
}

__global__ void naive_attn_pv(const float* P, const float* V, float* O,
                              int N) {
    // grid (N/32, d/32, B*H) block(32, 32), 这里 d=64 所以 d/32=2
    int bh = blockIdx.z;
    int row = blockIdx.x * 32 + threadIdx.y;
    int col = blockIdx.y * 32 + threadIdx.x;
    if (row >= N || col >= D_HEAD) return;
    int bh_off = bh * N * D_HEAD;
    float acc = 0.f;
    for (int k = 0; k < N; ++k) {
        acc += P[bh * N * N + row * N + k] * V[bh_off + k * D_HEAD + col];
    }
    O[bh_off + row * D_HEAD + col] = acc;
}

void launch_naive_attn(const float* dQ, const float* dK, const float* dV, float* dO,
                       float* dS_workspace,
                       int B, int H, int N, float scale) {
    dim3 g1((N + 31) / 32, (N + 31) / 32, B * H);
    naive_attn_score<<<g1, dim3(32, 32)>>>(dQ, dK, dS_workspace, N, scale);
    dim3 g2(N, B * H);
    naive_attn_softmax<<<g2, 128>>>(dS_workspace, N);
    dim3 g3((N + 31) / 32, (D_HEAD + 31) / 32, B * H);
    naive_attn_pv<<<g3, dim3(32, 32)>>>(dS_workspace, dV, dO, N);
    CHECK_LAST_CUDA_ERROR();
}

// ===========================================================================
// main
// ===========================================================================
int main(int argc, char** argv) {
    int B = 1, H = 8, N = 4096;
    if (argc > 1) N = std::atoi(argv[1]);
    if (argc > 2) H = std::atoi(argv[2]);
    if (argc > 3) B = std::atoi(argv[3]);
    printf("\n=== Flash Attention 1 forward (FP32) ===\n");
    printf("B=%d  H=%d  N=%d  d=%d, RTX 4090 Laptop (sm_89)\n", B, H, N, D_HEAD);

    int BH = B * H;
    size_t qkv_size = (size_t)BH * N * D_HEAD;
    float scale = 1.f / std::sqrt((float)D_HEAD);

    std::vector<float> hQ(qkv_size), hK(qkv_size), hV(qkv_size);
    fill_random(hQ.data(), (int)qkv_size, 1);
    fill_random(hK.data(), (int)qkv_size, 2);
    fill_random(hV.data(), (int)qkv_size, 3);

    float *dQ, *dK, *dV, *dO_fa, *dO_naive, *dS_ws;
    CHECK_CUDA(cudaMalloc(&dQ,       qkv_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dK,       qkv_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dV,       qkv_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dO_fa,    qkv_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dO_naive, qkv_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dS_ws, (size_t)BH * N * N * sizeof(float)));   // N x N 朴素 P 工作区
    CHECK_CUDA(cudaMemcpy(dQ, hQ.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dK, hK.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dV, hV.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice));

    // ---- 跑 FA1 ----
    launch_fa1_fwd(dQ, dK, dV, dO_fa, B, H, N, scale);
    CHECK_CUDA(cudaDeviceSynchronize());

    // ---- 跑朴素对照 ----
    launch_naive_attn(dQ, dK, dV, dO_naive, dS_ws, B, H, N, scale);
    CHECK_CUDA(cudaDeviceSynchronize());

    // ---- 校验 ----
    std::vector<float> hO_fa(qkv_size), hO_naive(qkv_size);
    CHECK_CUDA(cudaMemcpy(hO_fa.data(),    dO_fa,    qkv_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hO_naive.data(), dO_naive, qkv_size * sizeof(float), cudaMemcpyDeviceToHost));
    float err = max_abs_diff(hO_fa.data(), hO_naive.data(), (int)qkv_size);
    printf("max_abs_diff (FA1 vs naive) = %.3e\n", err);
    // FP32 + 不同 reduction 顺序, 4096 行点积加和顺序不同, 预期 ~1e-4 到 1e-5

    // ---- bench: FA1 ----
    CudaTimer t;
    const int n_iter = 20;
    for (int i = 0; i < 3; ++i) launch_fa1_fwd(dQ, dK, dV, dO_fa, B, H, N, scale);
    CHECK_CUDA(cudaDeviceSynchronize());
    t.start();
    for (int i = 0; i < n_iter; ++i) launch_fa1_fwd(dQ, dK, dV, dO_fa, B, H, N, scale);
    float ms_fa = t.stop() / n_iter;

    // ---- bench: naive ----
    for (int i = 0; i < 3; ++i) launch_naive_attn(dQ, dK, dV, dO_naive, dS_ws, B, H, N, scale);
    CHECK_CUDA(cudaDeviceSynchronize());
    t.start();
    for (int i = 0; i < n_iter; ++i) launch_naive_attn(dQ, dK, dV, dO_naive, dS_ws, B, H, N, scale);
    float ms_naive = t.stop() / n_iter;

    // FLOPS: 2 * N * N * d (QK^T) + 5*N*N (softmax) + 2 * N * N * d (PV) ≈ 4 N^2 d
    double flops_per_iter = 4.0 * (double)N * N * D_HEAD * BH;
    double tflops_fa    = flops_per_iter / (ms_fa    * 1e-3) / 1e12;
    double tflops_naive = flops_per_iter / (ms_naive * 1e-3) / 1e12;

    // Memory accessed by FA1: O(B*H*N*d) for Q+K+V+O, total 4*B*H*N*d * 4B
    double bytes_fa = 4.0 * BH * N * D_HEAD * 4.0;
    // Memory accessed by naive: 加上写 + 读 NxN 矩阵 (P) 两次, 主要 cost
    double bytes_naive = 2.0 * BH * N * N * 4.0 + 4.0 * BH * N * D_HEAD * 4.0;
    double gbs_fa = bytes_fa / (ms_fa * 1e-3) / 1e9;
    double gbs_naive = bytes_naive / (ms_naive * 1e-3) / 1e9;

    printf("\n--- bench ---\n");
    printf("[FA1   ]  %.3f ms,  %.2f TFLOPS,  est %.1f GB/s HBM\n", ms_fa,    tflops_fa,    gbs_fa);
    printf("[naive ]  %.3f ms,  %.2f TFLOPS,  est %.1f GB/s HBM\n", ms_naive, tflops_naive, gbs_naive);
    printf("FA1 speedup vs naive = %.2fx (FA1 saves %.0f MB of HBM P-matrix traffic)\n",
           ms_naive / ms_fa, BH * N * N * 4.0 / 1024.0 / 1024.0);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV);
    cudaFree(dO_fa); cudaFree(dO_naive); cudaFree(dS_ws);
    return 0;
}
