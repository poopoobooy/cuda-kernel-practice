// reduction.cu —— 1D 数组 sum reduction 的 4 种写法对比
// =================================================================
// 目的: 演示从最 naive 的 SMEM tree reduce → warp-shuffle 写法的性能差异
// 这是面试经典题, 必会
//
// 4 种写法:
//   v0: 每个 block 内 SMEM tree reduce (经典 NVIDIA reduction tutorial 第 1 版)
//   v1: 同 v0 但 unroll 最后 warp (省 __syncthreads)
//   v2: 全用 __shfl_down_sync, 不用 SMEM (warp shuffle 比 SMEM 快, 且 0 bank conflict)
//   v3: 每 thread 串行累加多个元素 (减少 block 数量 + 摊薄启动开销)
//
// 编译: nvcc -O3 -arch=sm_89 -o reduction.exe reduction.cu

#include "../sgemm/common.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

constexpr int WARP_SIZE = 32;

// ---- v0: SMEM tree reduce ----
__global__ void reduce_v0_smem_tree(const float* in, float* out, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (gid < N) ? in[gid] : 0.f;
    __syncthreads();

    // 二叉树规约: stride 从 blockDim.x/2 折半
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ---- v1: 最后一个 warp unroll (省掉 __syncthreads, 因为 warp 内 lockstep) ----
__device__ __forceinline__ void warp_reduce_v1(volatile float* s, int tid) {
    // volatile 是必须的: 防止编译器把 SMEM 值 cache 进 register 跳过同步
    s[tid] += s[tid + 32]; s[tid] += s[tid + 16];
    s[tid] += s[tid + 8];  s[tid] += s[tid + 4];
    s[tid] += s[tid + 2];  s[tid] += s[tid + 1];
}
__global__ void reduce_v1_unroll_last_warp(const float* in, float* out, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (gid < N) ? in[gid] : 0.f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid < 32) warp_reduce_v1(sdata, tid);
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ---- v2: 纯 warp shuffle (Kepler+) ----
// __shfl_down_sync(mask, val, delta):
//   lane i 拿到 lane (i+delta) 的 val (如果 i+delta>=32 则未定义, mask 控制谁参与)
__device__ __forceinline__ float warp_reduce_shfl(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, off);
    }
    return v;
}
__global__ void reduce_v2_shfl(const float* in, float* out, int N) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    float v = (gid < N) ? in[gid] : 0.f;

    // 1. warp 内 reduce
    v = warp_reduce_shfl(v);

    // 2. 各 warp 的 lane0 把 partial 写 SMEM
    __shared__ float smem[32];
    int lane = tid % WARP_SIZE;
    int wid  = tid / WARP_SIZE;
    if (lane == 0) smem[wid] = v;
    __syncthreads();

    // 3. 第一个 warp 把 SMEM 里的值再 reduce 一次
    if (wid == 0) {
        int n_warp = blockDim.x / WARP_SIZE;
        v = (lane < n_warp) ? smem[lane] : 0.f;
        v = warp_reduce_shfl(v);
        if (lane == 0) out[blockIdx.x] = v;
    }
}

// ---- v3: 每 thread 处理 ELEMS_PER_THREAD 个元素 ----
template <int ELEMS_PER_THREAD>
__global__ void reduce_v3_grid_stride(const float* in, float* out, int N) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid_start = bid * blockDim.x * ELEMS_PER_THREAD + tid;
    int stride = blockDim.x;

    float v = 0.f;
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        int idx = gid_start + i * stride;
        if (idx < N) v += in[idx];
    }

    v = warp_reduce_shfl(v);
    __shared__ float smem[32];
    int lane = tid % WARP_SIZE;
    int wid  = tid / WARP_SIZE;
    if (lane == 0) smem[wid] = v;
    __syncthreads();
    if (wid == 0) {
        int n_warp = blockDim.x / WARP_SIZE;
        v = (lane < n_warp) ? smem[lane] : 0.f;
        v = warp_reduce_shfl(v);
        if (lane == 0) out[bid] = v;
    }
}

// 注意: 之前版本的 run_reduce 每次跑都 cudaMalloc + cudaMemcpy, 这两步加起来
//       ~1-2ms 完全主导了 kernel 本身时间 (kernel ~0.1-0.3ms),
//       导致 "warp shuffle 比 SMEM tree 慢" 的反直觉数据.
//       正确做法: dPartial 在 bench 外预分配; 计时器只圈住 kernel launch.

template <typename K>
void launch_reduce(K kernel, const float* dIn, float* dPartial,
                   int N, int block_size, int smem_bytes) {
    int n_blocks = (N + block_size - 1) / block_size;
    kernel<<<n_blocks, block_size, smem_bytes>>>(dIn, dPartial, N);
}

int main(int argc, char** argv) {
    int N = 1 << 24;   // 16M floats = 64 MB
    if (argc > 1) N = std::atoi(argv[1]);
    printf("[reduction] N=%d (%.1f MB fp32)\n", N, (double)N * 4 / 1024 / 1024);

    std::vector<float> h(N);
    fill_random(h.data(), N, 0);
    double cpu_sum = 0.0; for (float x : h) cpu_sum += x;

    float *d;
    CHECK_CUDA(cudaMalloc(&d, sizeof(float) * N));
    CHECK_CUDA(cudaMemcpy(d, h.data(), sizeof(float) * N, cudaMemcpyHostToDevice));

    constexpr int BS = 256;
    int smem = BS * sizeof(float);
    int n_blocks = (N + BS - 1) / BS;

    // 提前分配 dPartial 和 host buf, 不计入 bench 时间
    float* dPartial;
    CHECK_CUDA(cudaMalloc(&dPartial, sizeof(float) * n_blocks));
    std::vector<float> hPart(n_blocks);

    // 给最后一轮校验用 (取一次 sum)
    auto sum_partial = [&](void) -> float {
        CHECK_CUDA(cudaMemcpy(hPart.data(), dPartial, sizeof(float)*n_blocks, cudaMemcpyDeviceToHost));
        double s = 0.0;
        for (float x : hPart) s += x;
        return (float)s;
    };

    auto bench = [&](const char* name, auto kernel) {
        // warmup
        for (int i = 0; i < 5; ++i) {
            launch_reduce(kernel, d, dPartial, N, BS, smem);
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        // 校验 (跑一次, 取 partial sum 跟 cpu 对比)
        launch_reduce(kernel, d, dPartial, N, BS, smem);
        float gpu_sum = sum_partial();

        // 真正的 bench: 只圈 kernel
        CudaTimer t;
        const int n_iter = 200;       // reduction 太短, 多跑些
        t.start();
        for (int i = 0; i < n_iter; ++i) {
            launch_reduce(kernel, d, dPartial, N, BS, smem);
        }
        float ms = t.stop() / n_iter;
        double gbs = (double)N * 4 / (ms * 1e-3) / 1e9;
        printf("[%s] %.4f ms, %6.1f GB/s,  sum_diff = %.3e\n",
               name, ms, gbs, std::fabs(gpu_sum - (float)cpu_sum));
    };

    bench("v0_smem_tree        ", reduce_v0_smem_tree);
    bench("v1_unroll_last_warp ", reduce_v1_unroll_last_warp);
    bench("v2_shfl_pure_warp   ", reduce_v2_shfl);

    CHECK_CUDA(cudaFree(dPartial));

    CHECK_CUDA(cudaFree(d));
    return 0;
}
