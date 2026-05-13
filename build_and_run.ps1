# build_and_run.ps1 —— 一键编译 + 跑全部 CUDA C++ kernels
# ========================================================
# 用法: powershell -ExecutionPolicy Bypass -File .\build_and_run.ps1
#
# 这个脚本做了 3 件事:
#   1. source vcvarsall.bat x64 (配置 MSVC 环境)
#   2. 把 nvcc 加进 PATH
#   3. 用一个 cmd 调用串起来编 + 跑 所有 kernel

$ErrorActionPreference = "Stop"

# ---- 路径 ----
$VCVARS = "E:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
$CUDA_BIN = "C:\Users\16229\miniconda3\envs\cuda-kernel\Library\bin"
$NVCC = "$CUDA_BIN\nvcc.exe"

if (-not (Test-Path $VCVARS)) { throw "vcvarsall.bat not found at $VCVARS" }
if (-not (Test-Path $NVCC))   { throw "nvcc not found at $NVCC" }

# ---- 编译命令 ----
# 关键 flags:
#   -O3              host 端最大优化
#   -arch=sm_89      Ada (4090) GPU 编译目标
#   --use_fast_math  __expf / fdividef 走 fast intrinsic (softmax 必备)
$NVCC_FLAGS = "-O3 -arch=sm_89 --use_fast_math"

# 列表: 文件 -> 输出 exe
$builds = @(
    @{src='sgemm\01_naive.cu';    out='sgemm\01_naive.exe'},
    @{src='sgemm\02_coalesce.cu'; out='sgemm\02_coalesce.exe'},
    @{src='sgemm\03_smem.cu';     out='sgemm\03_smem.exe'},
    @{src='sgemm\04_1d_tile.cu';  out='sgemm\04_1d_tile.exe'},
    @{src='sgemm\05_2d_tile.cu';  out='sgemm\05_2d_tile.exe'},
    @{src='sgemm\bench.cu';       out='sgemm\sgemm_bench.exe';   extra='-DLIB_ONLY -lcublas'},
    @{src='softmax\softmax_cuda.cu'; out='softmax\softmax_cuda.exe'},
    @{src='reduction\reduction.cu';  out='reduction\reduction.exe'}
)

Write-Host "==== build phase ====" -ForegroundColor Cyan
foreach ($b in $builds) {
    $extra = if ($b.extra) { $b.extra } else { "" }
    $cmd = "call `"$VCVARS`" x64 >nul && `"$NVCC`" $NVCC_FLAGS $extra -o `"$($b.out)`" `"$($b.src)`""
    Write-Host "[build] $($b.src) -> $($b.out)" -ForegroundColor Yellow
    cmd /c $cmd
    if ($LASTEXITCODE -ne 0) { throw "build failed: $($b.src)" }
}

Write-Host "`n==== run phase ====" -ForegroundColor Cyan

# 跑 SGEMM 总 bench (5 版 + cuBLAS 对比, 写 sgemm_raw.json)
Write-Host "`n[run] sgemm bench" -ForegroundColor Yellow
& .\sgemm\sgemm_bench.exe

# 跑 softmax CUDA
Write-Host "`n[run] softmax cuda" -ForegroundColor Yellow
& .\softmax\softmax_cuda.exe 4096 512

# 跑 reduction
Write-Host "`n[run] reduction" -ForegroundColor Yellow
& .\reduction\reduction.exe

Write-Host "`n==== done ====" -ForegroundColor Green
