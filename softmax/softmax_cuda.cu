// softmax_cuda.cu —— 手写 CUDA safe softmax (per-row)
// =================================================================
// 核心点:
//   1. safe softmax: y[i] = exp(x[i] - max(x)) / sum(exp(x[j] - max(x)))
//      减 max 防止 expf 溢出 (FP32 expf(>89) 就 inf)
//   2. block 处理一行: warp 内 __shfl_xor_sync 做 reduction (5 step 取 max/sum)
//      跨 warp 用 SMEM 凑齐再做一次 warp reduce
//   3. 用 __expf (fast intrinsic) 而不是 expf
//
// 假设: 输入 [M, N] row-major, 每个 block 处理一行
// 限制: 这一版要求 N <= BLOCK_SIZE (= 1024). 更大的 N 要多 block 协作,
//       本 demo 不展开.
//
// 编译:  nvcc -O3 -arch=sm_89 -o softmax_cuda.exe softmax_cuda.cu

#include "../sgemm/common.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>

constexpr int WARP_SIZE = 32;

// ---- warp-level reduce: 把 warp 内 32 个 val 求 max ----
// __shfl_xor_sync(mask, val, lane_offset):
//   把 lane i 的 val 和 lane (i XOR lane_offset) 的 val 互换
// 5 步 (16, 8, 4, 2, 1) 后, warp 内每个 lane 都拿到 32 个值的 max
__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        float o = __shfl_xor_sync(0xffffffff, v, off);
        v = (v > o) ? v : o;
    }
    return v;
}
__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, off);
    }
    return v;
}

// ---- block-level reduce: 跨 warp 凑齐 ----
// 步骤: 每个 warp 内 reduce → 各 warp 的 lane0 把 partial result 写 SMEM
//      → 第 0 warp 把 SMEM 里的 num_warps 个值再做一次 warp reduce
template <bool IS_MAX>
__device__ __forceinline__ float block_reduce(float v, float* smem) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid  = threadIdx.x / WARP_SIZE;
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    // warp 内 reduce
    v = IS_MAX ? warp_reduce_max(v) : warp_reduce_sum(v);

    if (lane == 0) smem[wid] = v;
    __syncthreads();

    // 第 0 warp 把 smem 里 num_warps 个 partial 再 reduce
    if (wid == 0) {
        v = (lane < num_warps) ? smem[lane] : (IS_MAX ? -INFINITY : 0.f);
        v = IS_MAX ? warp_reduce_max(v) : warp_reduce_sum(v);
        if (lane == 0) smem[0] = v;   // 广播位置
    }
    __syncthreads();
    return smem[0];
}

// 1 block / row
__global__ void softmax_kernel(const float* X, float* Y, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* x = X + row * N;
    float* y = Y + row * N;

    __shared__ float smem[32];   // 最多 32 warps = 1024 threads/block

    // ---- step 1: max ----
    // 每 thread 处理 ceil(N / blockDim.x) 个元素 (本 demo 假设 N<=blockDim.x, 简化)
    float v = (tid < N) ? x[tid] : -INFINITY;
    float row_max = block_reduce<true>(v, smem);

    // ---- step 2: exp(x - max), 同时算 sum ----
    float e = (tid < N) ? __expf(x[tid] - row_max) : 0.f;
    float row_sum = block_reduce<false>(e, smem);

    // ---- step 3: 写回 ----
    if (tid < N) {
        y[tid] = e / row_sum;
    }
}

void softmax_cuda(const float* dX, float* dY, int M, int N) {
    // BLOCK_SIZE 取覆盖 N 的最小 32 倍数, 但 <= 1024
    int block_size = 32;
    while (block_size < N && block_size < 1024) block_size *= 2;
    softmax_kernel<<<M, block_size>>>(dX, dY, N);
    CHECK_LAST_CUDA_ERROR();
}

// ---- CPU reference (safe softmax) ----
void softmax_cpu_ref(const float* x, float* y, int M, int N) {
    for (int i = 0; i < M; ++i) {
        const float* xr = x + i * N;
        float* yr = y + i * N;
        float m = xr[0];
        for (int j = 1; j < N; ++j) if (xr[j] > m) m = xr[j];
        double s = 0.0;
        for (int j = 0; j < N; ++j) { yr[j] = std::exp(xr[j] - m); s += yr[j]; }
        for (int j = 0; j < N; ++j) yr[j] /= (float)s;
    }
}

int main(int argc, char** argv) {
    int M = 4096, N = 512;
    if (argc > 1) M = std::atoi(argv[1]);
    if (argc > 2) N = std::atoi(argv[2]);
    if (N > 1024) {
        fprintf(stderr, "本 demo 限制 N<=1024 (1 block 1 row); 你给的 N=%d\n", N);
        return 1;
    }
    printf("[softmax_cuda] M=%d N=%d\n", M, N);

    std::vector<float> hX((size_t)M * N), hY((size_t)M * N), hRef((size_t)M * N);
    fill_random(hX.data(), M * N, 7);

    float *dX, *dY;
    CHECK_CUDA(cudaMalloc(&dX, sizeof(float)*M*N));
    CHECK_CUDA(cudaMalloc(&dY, sizeof(float)*M*N));
    CHECK_CUDA(cudaMemcpy(dX, hX.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice));

    // warmup
    for (int i = 0; i < 3; ++i) softmax_cuda(dX, dY, M, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 正确性
    CHECK_CUDA(cudaMemcpy(hY.data(), dY, sizeof(float)*M*N, cudaMemcpyDeviceToHost));
    softmax_cpu_ref(hX.data(), hRef.data(), M, N);
    float err = max_abs_diff(hY.data(), hRef.data(), M * N);
    printf("[softmax_cuda] max_abs_diff vs cpu_ref = %.3e\n", err);

    // 性能
    CudaTimer t;
    const int n_iter = 100;
    t.start();
    for (int i = 0; i < n_iter; ++i) softmax_cuda(dX, dY, M, N);
    float ms = t.stop() / n_iter;
    double bytes = (double)M * N * 4 * 2;   // read + write fp32
    double gbs = bytes / (ms * 1e-3) / 1e9;
    printf("[softmax_cuda] %.4f ms/iter, %.1f GB/s\n", ms, gbs);

    CHECK_CUDA(cudaFree(dX));
    CHECK_CUDA(cudaFree(dY));
    return 0;
}
