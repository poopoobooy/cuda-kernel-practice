// bench.cu —— SGEMM 5 版 vs cuBLAS 性能对比 + 正确性校验
// =================================================================
// 编译方式 (一行搞定, 因为 bench.cu 把 5 个 .cu 都 include 进来):
//   nvcc -O3 -arch=sm_89 -DLIB_ONLY -lcublas -o sgemm_bench.exe bench.cu
//
// 用法:
//   ./sgemm_bench.exe          # M=N=K=4096 (默认)
//   ./sgemm_bench.exe 2048     # M=N=K=2048
//
// 输出:
//   - stdout: 5 版 + cuBLAS 的 GFLOPS / 速比 / 最大误差
//   - results/sgemm_raw.json: 给 Python plot 用的结构化数据

#include "common.cuh"
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

// 把 5 个版本的 host 入口都"导入"进来 (不要 main)
#define LIB_ONLY
#include "01_naive.cu"
#include "02_coalesce.cu"
#include "03_smem.cu"
#include "04_1d_tile.cu"
#include "05_2d_tile.cu"
#undef LIB_ONLY

// cuBLAS 包一层 (注意 cuBLAS 是 column-major 默认, 我们 A/B/C 是 row-major)
// 技巧: C = A * B (row-major)
//       等价于  C^T = B^T * A^T  (column-major 视角)
// cuBLAS Sgemm(opA=N, opB=N, m, n, k, alpha, B, ldb=N, A, lda=K, beta, C, ldc=N)
// 这样不用真转置, 直接颠倒参数顺序即可
void sgemm_cublas(cublasHandle_t h, const float* dA, const float* dB, float* dC,
                  int M, int N, int K) {
    const float alpha = 1.f, beta = 0.f;
    cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                dB, N,
                dA, K,
                &beta,
                dC, N);
}

struct Result {
    std::string name;
    double gflops;
    float ms;
    float pct_cublas;
    float max_err;
};

// 跑一版, 含 warmup + n_iter 次 + 校验
template <typename F>
Result run_one(const char* name, F fn,
               const float* dA, const float* dB, float* dC,
               int M, int N, int K,
               const std::vector<float>& ref) {
    // warmup
    for (int i = 0; i < 3; ++i) fn(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 校验
    std::vector<float> hC((size_t)M * N);
    CHECK_CUDA(cudaMemcpy(hC.data(), dC, sizeof(float)*M*N, cudaMemcpyDeviceToHost));
    float err = max_abs_diff(hC.data(), ref.data(), M * N);

    // bench
    CudaTimer t;
    const int n_iter = 5;
    t.start();
    for (int i = 0; i < n_iter; ++i) fn(dA, dB, dC, M, N, K);
    float ms = t.stop() / n_iter;

    return {name, gflops(M, N, K, ms), ms, 0.f, err};
}

int main(int argc, char** argv) {
    int M = 4096, N = 4096, K = 4096;
    if (argc > 1) { M = N = K = std::atoi(argv[1]); }
    printf("\n=== SGEMM bench: M=N=K=%d, fp32, RTX 4090 (sm_89) ===\n\n", M);

    // host 数据 + 拷到 device
    std::vector<float> hA((size_t)M * K), hB((size_t)K * N);
    fill_random(hA.data(), M * K, 1);
    fill_random(hB.data(), K * N, 2);

    float *dA, *dB, *dC, *dC_ref;
    CHECK_CUDA(cudaMalloc(&dA, sizeof(float) * M * K));
    CHECK_CUDA(cudaMalloc(&dB, sizeof(float) * K * N));
    CHECK_CUDA(cudaMalloc(&dC, sizeof(float) * M * N));
    CHECK_CUDA(cudaMalloc(&dC_ref, sizeof(float) * M * N));
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), sizeof(float)*M*K, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), sizeof(float)*K*N, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    cublasCreate(&handle);

    // ---- 先跑 cuBLAS 当 reference ----
    for (int i = 0; i < 3; ++i) sgemm_cublas(handle, dA, dB, dC_ref, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> ref((size_t)M * N);
    CHECK_CUDA(cudaMemcpy(ref.data(), dC_ref, sizeof(float)*M*N, cudaMemcpyDeviceToHost));

    CudaTimer t;
    const int n_iter = 10;
    t.start();
    for (int i = 0; i < n_iter; ++i) sgemm_cublas(handle, dA, dB, dC_ref, M, N, K);
    float ms_cublas = t.stop() / n_iter;
    double gf_cublas = gflops(M, N, K, ms_cublas);
    printf("[cuBLAS] %.3f ms, %.1f GFLOPS  (100%% baseline)\n", ms_cublas, gf_cublas);

    // ---- 5 版逐个跑 ----
    std::vector<Result> results;
    results.push_back(run_one("01_naive",    sgemm_naive,    dA, dB, dC, M, N, K, ref));
    results.push_back(run_one("02_coalesce", sgemm_coalesce, dA, dB, dC, M, N, K, ref));
    results.push_back(run_one("03_smem",     sgemm_smem,     dA, dB, dC, M, N, K, ref));
    results.push_back(run_one("04_1d_tile",  sgemm_1d_tile,  dA, dB, dC, M, N, K, ref));
    results.push_back(run_one("05_2d_tile",  sgemm_2d_tile,  dA, dB, dC, M, N, K, ref));

    for (auto& r : results) {
        r.pct_cublas = (float)(r.gflops / gf_cublas * 100.0);
        printf("[%s] %.3f ms, %.1f GFLOPS (%.1f%% cuBLAS), max_err=%.3e\n",
               r.name.c_str(), r.ms, r.gflops, r.pct_cublas, r.max_err);
    }

    // ---- 写 JSON ----
    // 简陋手写, 避免引入 nlohmann/json 额外依赖
    FILE* fp = fopen("../results/sgemm_raw.json", "w");
    if (!fp) fp = fopen("results/sgemm_raw.json", "w");  // 兼容不同的 cwd
    if (!fp) fp = fopen("sgemm_raw.json", "w");
    if (fp) {
        fprintf(fp, "[\n");
        fprintf(fp, "  {\"version\":\"cuBLAS\",\"gflops\":%.2f,\"ms\":%.4f,\"pct_cublas\":100.0,\"max_err\":0.0},\n",
                gf_cublas, ms_cublas);
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            fprintf(fp, "  {\"version\":\"%s\",\"gflops\":%.2f,\"ms\":%.4f,\"pct_cublas\":%.2f,\"max_err\":%.3e}%s\n",
                    r.name.c_str(), r.gflops, r.ms, r.pct_cublas, r.max_err,
                    i + 1 == results.size() ? "" : ",");
        }
        fprintf(fp, "]\n");
        fclose(fp);
        printf("\n[OK] saved sgemm_raw.json\n");
    } else {
        fprintf(stderr, "WARN: 无法写 sgemm_raw.json\n");
    }

    cublasDestroy(handle);
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    CHECK_CUDA(cudaFree(dC_ref));
    return 0;
}
