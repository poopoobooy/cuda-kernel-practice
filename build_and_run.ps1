# build_and_run.ps1 -- one-shot: compile + run all CUDA C++ kernels
# ================================================================
# Usage:  powershell -ExecutionPolicy Bypass -File .\build_and_run.ps1
#
# Steps:
#   1. source vcvarsall.bat x64 (MSVC env)
#   2. put nvcc on PATH via $env:PATH
#   3. invoke nvcc through cmd /c so the MSVC env survives

$ErrorActionPreference = "Stop"

$VCVARS   = "E:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
$CUDA_BIN = "C:\Users\16229\miniconda3\envs\cuda-kernel\Library\bin"
$NVCC     = Join-Path $CUDA_BIN "nvcc.exe"

if (-not (Test-Path $VCVARS)) { throw "vcvarsall.bat not found at $VCVARS" }
if (-not (Test-Path $NVCC))   { throw "nvcc not found at $NVCC" }

# flags:
#   -O3              host opt
#   -arch=sm_89      Ada (4090)
#   --use_fast_math  __expf / fdividef / __fsqrt etc.
$NVCC_FLAGS = "-O3 -arch=sm_89 --use_fast_math"

function Invoke-NvccBuild {
    param(
        [string]$Src,
        [string]$Out,
        [string]$Extra = ""
    )
    $q = [char]34
    $cmd = "call $q$VCVARS$q x64 >nul && $q$NVCC$q $NVCC_FLAGS $Extra -o $q$Out$q $q$Src$q"
    Write-Host "[build] $Src -> $Out" -ForegroundColor Yellow
    cmd /c $cmd
    if ($LASTEXITCODE -ne 0) { throw "build failed: $Src" }
}

Write-Host "==== build phase ====" -ForegroundColor Cyan
Invoke-NvccBuild "sgemm\01_naive.cu"        "sgemm\01_naive.exe"
Invoke-NvccBuild "sgemm\02_coalesce.cu"     "sgemm\02_coalesce.exe"
Invoke-NvccBuild "sgemm\03_smem.cu"         "sgemm\03_smem.exe"
Invoke-NvccBuild "sgemm\04_1d_tile.cu"      "sgemm\04_1d_tile.exe"
Invoke-NvccBuild "sgemm\05_2d_tile.cu"      "sgemm\05_2d_tile.exe"
Invoke-NvccBuild "sgemm\06_wmma_fp16.cu"    "sgemm\06_wmma_fp16.exe"   "-lcublas"
Invoke-NvccBuild "sgemm\bench.cu"           "sgemm\sgemm_bench.exe"    "-DLIB_ONLY -lcublas"
Invoke-NvccBuild "softmax\softmax_cuda.cu"  "softmax\softmax_cuda.exe"
Invoke-NvccBuild "reduction\reduction.cu"   "reduction\reduction.exe"
Invoke-NvccBuild "flash_attn\fa1_fwd.cu"    "flash_attn\fa1_fwd.exe"

Write-Host "`n==== run phase ====" -ForegroundColor Cyan

Write-Host "`n[run] sgemm bench (5 FP32 versions + cuBLAS)" -ForegroundColor Yellow
& .\sgemm\sgemm_bench.exe

Write-Host "`n[run] sgemm v6 WMMA fp16  N = 2048, 4096, 8192" -ForegroundColor Yellow
& .\sgemm\06_wmma_fp16.exe 2048
& .\sgemm\06_wmma_fp16.exe 4096
& .\sgemm\06_wmma_fp16.exe 8192

Write-Host "`n[run] softmax cuda" -ForegroundColor Yellow
& .\softmax\softmax_cuda.exe 4096 512

Write-Host "`n[run] reduction" -ForegroundColor Yellow
& .\reduction\reduction.exe

Write-Host "`n[run] flash_attn fa1_fwd" -ForegroundColor Yellow
& .\flash_attn\fa1_fwd.exe 2048 8 1
& .\flash_attn\fa1_fwd.exe 4096 8 1
& .\flash_attn\fa1_fwd.exe 8192 4 1

Write-Host "`n==== done ====" -ForegroundColor Green
