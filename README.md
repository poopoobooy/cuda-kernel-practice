# cuda-kernel-practice

> **个人学习项目，2026-05 完成**。为算子开发方向校招准备，目标是把 CUDA / Triton 算子最关键的几个 idiom 上手敲一遍、跑一遍、画一遍。**不是长期工程经验，是 48 小时密集突击的产物**。

## TL;DR

- 在 RTX 4090 (sm_89, FP32) 上，把 SGEMM 从 naive 一路调到 cuBLAS 的 **<填入百分比>** %
- Triton fused softmax / layernorm 对比 PyTorch eager 的加速比 + 带宽利用率曲线见下方
- PyTorch C++/CUDA Extension 把手写 SGEMM 包成 `torch.autograd.Function`，正反向 `torch.allclose` 验证通过

![sgemm](results/sgemm_curve.png)
![softmax](results/softmax_curve.png)
![layernorm](results/layernorm_curve.png)

## 环境

| 项目 | 配置 |
|---|---|
| OS | Windows 11 |
| GPU | NVIDIA RTX 4090 16GB（sm_89, Ada Lovelace） |
| Driver | 595.79（CUDA 13.2） |
| Python | 3.11.15 (conda env `cuda-kernel`, conda-forge channel) |
| PyTorch | 2.12.0 + cu126 |
| Triton | `triton-windows` |
| CUDA Toolkit | 12.6 (nvcc) |
| 编译器 | MSVC v143 (Visual Studio 2022 Build Tools) |

## 目录结构

```
cuda-kernel-practice/
├── sgemm/              # SGEMM 五版迭代 + cuBLAS 对比
│   ├── common.cuh      #   公共工具: CHECK_CUDA, CudaTimer, 校验
│   ├── 01_naive.cu     #   v1 每 thread 一个 C 元素, 非 coalesce
│   ├── 02_coalesce.cu  #   v2 改 thread mapping → coalesced load
│   ├── 03_smem.cu      #   v3 BS=32 shared memory tiling
│   ├── 04_1d_tile.cu   #   v4 每 thread TM=8 个 C 元素 + register tile
│   ├── 05_2d_tile.cu   #   v5 TM=TN=8 2D thread tile + 外积累加
│   └── bench.cu        #   5 版 + cuBLAS 一起跑, 出 sgemm_raw.json
├── softmax/
│   ├── softmax_cuda.cu     # warp shuffle reduction + safe softmax
│   └── softmax_triton.py   # Triton fused 版
├── layernorm/
│   └── layernorm_triton.py # Triton fused, fp32 累加保精度
├── reduction/
│   └── reduction.cu        # SMEM tree vs warp shuffle 对比 demo
├── torch_ext/
│   ├── sgemm_op.cu         # v5 包装成 torch.autograd.Function
│   ├── setup.py
│   └── test.py             # forward + backward allclose 校验
├── bench/
│   ├── run_all.py          # Triton 部分 benchmark → results.json
│   └── plot.py             # results.json → 性能曲线 png
└── results/                # benchmark 产物 (json + png), 已 commit
```

## 快速复现

```powershell
# 1. 装环境 (Windows + Miniconda + 4090)
conda create -n cuda-kernel -c conda-forge --override-channels python=3.11 pip -y
conda activate cuda-kernel
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# 2. 跑 Triton 部分 (无需 nvcc)
python bench/run_all.py
python bench/plot.py

# 3. 跑 CUDA C++ 部分 (需要 nvcc + MSVC)
cd sgemm
nvcc -O3 -arch=sm_89 -DLIB_ONLY -lcublas -o sgemm_bench.exe bench.cu
.\sgemm_bench.exe

cd ..\reduction
nvcc -O3 -arch=sm_89 -o reduction.exe reduction.cu
.\reduction.exe

cd ..\softmax
nvcc -O3 -arch=sm_89 -o softmax_cuda.exe softmax_cuda.cu
.\softmax_cuda.exe

# 4. 跑 PyTorch extension
cd ..\torch_ext
python setup.py build_ext --inplace
python test.py
```

## 性能数据 (RTX 4090, FP32, M=N=K=4096)

> 数据由 `sgemm/bench.cu` 自动生成 `results/sgemm_raw.json`，下表是手填的关键数字。

| Kernel | GFLOPS | ms | % cuBLAS | max_err vs cuBLAS |
|---|---:|---:|---:|---:|
| 01 naive            |  | | | |
| 02 coalesce         |  | | | |
| 03 smem tiling      |  | | | |
| 04 1D thread tile   |  | | | |
| 05 2D thread tile   |  | | | |
| cuBLAS (reference)  |  | | 100 % | — |

Softmax / LayerNorm 数据见 `results/results.json`，曲线图见 `results/softmax_curve.png` / `results/layernorm_curve.png`。

## 每一版的关键 idea (面试 talk track)

### SGEMM v1 → v2: coalesce
- v1 同 warp 32 个 thread 取 A 时是跨 K 步 stride，每次 32 个分立事务 → 带宽利用率极低
- v2 把 block 改 1D，`threadIdx.x % 32` 绑列，warp 内 32 thread 取 B 是 1 个 128B 连续事务
- 一般会带来 3-5× 提速

### SGEMM v2 → v3: shared memory tiling
- 每元素的 global load 从 `2K` 降到 `2K/BS`（BS=32 → 32 倍降低）
- 关键: 两次 `__syncthreads()` 保证 SMEM 不被踩
- 注意 SMEM 的 bank conflict —— `Bs[k][t_col]` t_col 0..31 = 32 banks，刚好不冲突

### SGEMM v3 → v4: 1D thread tile (register reuse)
- 每个 thread 不再算 1 个 C 元素而是算 TM=8 个（同列）
- inner loop 里 `Bs[k][t_col]` 取一次到 register 后被 TM 个 FMA 复用 → 减少 SMEM 访问
- 减少 block 内 thread 数（4096 → 512），减少同步开销

### SGEMM v4 → v5: 2D thread tile (外积累加)
- 每个 thread 算 TM×TN = 8×8 = 64 个 C 元素
- inner loop 外积写法: `acc[i][j] += reg_a[i] * reg_b[j]`，64 FMA / 16 reg load = 4 FMA/load
- 算力密度大幅提升，能逼近 cuBLAS 50%+ 性能（要再上需要 ldmatrix + TensorCore，本项目不展开）

### Softmax (CUDA)
- safe softmax: 减 max 防 exp 溢出
- warp-level reduction: `__shfl_xor_sync` 5 步完成 32 个值的 max / sum
- cross-warp: 各 warp 的 lane0 写 SMEM → 第 0 warp 再 reduce 一次
- 用 `__expf` 而不是 `expf`（fast intrinsic）

### Softmax / LayerNorm (Triton)
- Triton 是 block-level 编程，一个 program 处理一行
- `tl.max` / `tl.sum` 自动展开成最优 reduction
- `num_warps` 按 BLOCK 大小调（< 2048 用 4，< 4096 用 8，否则 16）
- LayerNorm 必须用 fp32 累加 mean/var，否则 N>=4096 会爆精度

### Reduction
- SMEM tree v.s. warp shuffle: warp shuffle 不走 SMEM，无 bank conflict，~1.5-2× 快
- 最后一个 warp 可以 unroll 并去掉 `__syncthreads`（warp 内 lockstep 保证）

### PyTorch Extension
- `torch.utils.cpp_extension.CUDAExtension` 自动处理 nvcc / MSVC 链接
- `at::cuda::getCurrentCUDAStream()` 让自定义 kernel 走 PyTorch 的当前 stream，避免和 torch 的内部 op 串行混乱
- backward: `grad_A = grad_C @ B.T`，`grad_B = A.T @ grad_C`（复用 `torch.matmul` 跑 cuBLAS）

## 踩坑记录

<!-- 自己填: 装环境 / 调 kernel 时遇到的真实坑, 越具体越好 -->
- ⚠️ Windows + Python 3.13 + Triton 组合不稳，最后用 Python 3.11
- ⚠️ PyTorch 官方 wheel 从国内拉 2.6GB，VPN 限速被卡，最后用 curl 断点续传 + 直连 Cloudflare R2 解决
- ⚠️ <填: 比如 v3 SMEM bank conflict 怎么发现的、Triton kernel 调试怎么看 IR、…>

## 我学到了什么

<!-- 这一段必须自己写，面试官就靠这个判断"你是真做了还是抄了" -->
- TODO（自己填）

## References

- [Simon Boehm: How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM)
- [NVIDIA: Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
- [Triton 官方 tutorial: fused softmax / layernorm](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [PyTorch C++ Extension docs](https://pytorch.org/tutorials/advanced/cpp_extension.html)
