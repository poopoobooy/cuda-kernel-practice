// common.cuh —— SGEMM 5 版共用工具
// =================================================================
// 包括: CUDA error check, CUDA Event 计时器, FP32 矩阵 max-abs-diff 校验
// 面试要点:
//   - CHECK_CUDA 这种宏是工业 CUDA 代码的标配（避免 silent failure）
//   - cudaEventRecord / cudaEventElapsedTime 是 device-side 计时，
//     比 std::chrono 准确（不会被 host-side 抖动污染）

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// ------- error check -------
// 任何 CUDA Runtime API 都用这个宏包；返回值非 cudaSuccess 直接 abort，
// 因为 CUDA 错误一旦发生再继续跑只会越错越远（context 进入 corrupted 状态）
#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err));                                  \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// ------- kernel 启动后的错误检查 -------
// kernel 启动是异步的，错误要靠 cudaGetLastError + 一次 sync 抓出来
#define CHECK_LAST_CUDA_ERROR()                                                \
    do {                                                                       \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "Kernel launch error %s:%d: %s\n", __FILE__,       \
                    __LINE__, cudaGetErrorString(err));                        \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
        CHECK_CUDA(cudaDeviceSynchronize());                                   \
    } while (0)

// ------- CUDA Event 计时器 -------
// 用法:
//   CudaTimer t; t.start(); kernel<<<...>>>(...); float ms = t.stop();
// 注意: 不要 sync 太多次（每次 stop 内部已经 sync）
struct CudaTimer {
    cudaEvent_t s, e;
    CudaTimer()  { cudaEventCreate(&s); cudaEventCreate(&e); }
    ~CudaTimer() { cudaEventDestroy(s); cudaEventDestroy(e); }

    void start() { cudaEventRecord(s); }
    float stop() {
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, s, e);
        return ms;
    }
};

// ------- 矩阵正确性校验 -------
// 返回 max abs diff，调用方决定阈值（FP32 SGEMM 一般 1e-2 量级是合理的，
// 因为 N=4096 累加会引入 ~N * eps 的累计误差）
inline float max_abs_diff(const float* a, const float* b, int n) {
    float m = 0.f;
    for (int i = 0; i < n; ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

// ------- 随机初始化 (host) -------
// 用固定 seed 保证多次跑的输入完全一致 —— 性能 benchmark 必须的
inline void fill_random(float* p, int n, unsigned seed = 42) {
    // 简单的 LCG，足够当 benchmark 输入用，无需高质量随机
    unsigned state = seed;
    for (int i = 0; i < n; ++i) {
        state = state * 1664525u + 1013904223u;
        p[i] = (float)((int)(state & 0xffff) - 32768) / 32768.f;
    }
}

// ------- GFLOPS 计算 -------
// SGEMM 浮点数: 每个 C[i,j] 做 K 次乘加 = 2*K flops；总 = 2 * M * N * K
inline double gflops(int M, int N, int K, float ms) {
    return 2.0 * M * N * K / (ms * 1e-3) / 1e9;
}
