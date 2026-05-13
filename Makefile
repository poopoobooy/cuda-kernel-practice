# Makefile (nmake 友好, Windows)
# ==========================================================
# 用法 (装好 nvcc + MSVC, x64 Native Tools Command Prompt 里):
#   nmake -f Makefile         # 等价 nmake all
#   nmake -f Makefile clean
#
# 也能用 GNU make (msys2 / cygwin / WSL); 注意 / 改 \ 之类
# ==========================================================

# 4090 是 sm_89; --use_fast_math 把 expf / div 都换成 fast intrinsic
NVCC      = nvcc
NVCC_FLAGS = -O3 -arch=sm_89 --use_fast_math
LIBS_CUBLAS = -lcublas

all: sgemm_bench softmax_cuda reduction

sgemm_bench:
	$(NVCC) $(NVCC_FLAGS) -DLIB_ONLY $(LIBS_CUBLAS) -o sgemm\sgemm_bench.exe sgemm\bench.cu

softmax_cuda:
	$(NVCC) $(NVCC_FLAGS) -o softmax\softmax_cuda.exe softmax\softmax_cuda.cu

reduction:
	$(NVCC) $(NVCC_FLAGS) -o reduction\reduction.exe reduction\reduction.cu

# 单独跑 SGEMM 5 版 (debug 用, 跑完看每一版独立的 GFLOPS)
sgemm_each:
	$(NVCC) $(NVCC_FLAGS) -o sgemm\01_naive.exe    sgemm\01_naive.cu
	$(NVCC) $(NVCC_FLAGS) -o sgemm\02_coalesce.exe sgemm\02_coalesce.cu
	$(NVCC) $(NVCC_FLAGS) -o sgemm\03_smem.exe     sgemm\03_smem.cu
	$(NVCC) $(NVCC_FLAGS) -o sgemm\04_1d_tile.exe  sgemm\04_1d_tile.cu
	$(NVCC) $(NVCC_FLAGS) -o sgemm\05_2d_tile.exe  sgemm\05_2d_tile.cu

torch_ext:
	cd torch_ext && python setup.py build_ext --inplace

run_bench: all
	cd sgemm && sgemm_bench.exe
	cd softmax && softmax_cuda.exe
	cd reduction && reduction.exe
	python bench\run_all.py
	python bench\plot.py

clean:
	@if exist sgemm\*.exe del /Q sgemm\*.exe
	@if exist softmax\*.exe del /Q softmax\*.exe
	@if exist reduction\*.exe del /Q reduction\*.exe
	@if exist torch_ext\build rmdir /S /Q torch_ext\build
	@if exist torch_ext\*.pyd del /Q torch_ext\*.pyd

.PHONY: all clean sgemm_bench softmax_cuda reduction sgemm_each torch_ext run_bench
