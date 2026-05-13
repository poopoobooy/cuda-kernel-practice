@echo off
REM build_ext.bat —— 在 MSVC + nvcc + conda 环境下编 PyTorch C++ extension
REM 用法 (从项目根目录):  torch_ext\build_ext.bat
REM (或在 torch_ext 目录里:  build_ext.bat)

setlocal

REM 1. 配置 MSVC 环境 (会修改 PATH/INCLUDE/LIB)
call "E:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

REM 2. 把 conda env 的 bin (含 nvcc + cublas) 加到 PATH 前面
set PATH=C:\Users\16229\miniconda3\envs\cuda-kernel\Library\bin;%PATH%

REM 3. 告诉 setuptools 直接用上面 source 进来的 MSVC 环境, 别自己探测
set DISTUTILS_USE_SDK=1

REM 4. cd 到本 .bat 所在目录 (torch_ext)
cd /d "%~dp0"

REM 5. 编
"C:\Users\16229\miniconda3\envs\cuda-kernel\python.exe" setup.py build_ext --inplace

endlocal
