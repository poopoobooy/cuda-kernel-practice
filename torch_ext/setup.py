"""PyTorch CUDA extension 构建脚本

用法 (Windows, 在 torch_ext 目录里):
  build_ext.bat     # 自动 source vcvarsall.bat 后调 python setup.py

坑记:
  - conda-forge 装的 pytorch 把 .lib (c10.lib/torch.lib/...) 放在 ${CONDA_PREFIX}/Library/lib
  - PyTorch 的 cpp_extension 默认只找 site-packages/torch/lib, 找不到就 LNK1181 c10.lib
  - 这里显式补 library_dirs 修复
"""

import os
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _conda_lib_dir():
    """如果是 conda env, 返回 Library/lib (Windows) 或 lib (Linux); 否则 None"""
    if sys.platform == 'win32':
        prefix = os.environ.get('CONDA_PREFIX') or sys.prefix
        cand = os.path.join(prefix, 'Library', 'lib')
        return cand if os.path.isdir(cand) else None
    return None


extra_lib_dirs = []
cand = _conda_lib_dir()
if cand:
    extra_lib_dirs.append(cand)
    print(f"[setup.py] adding library_dirs: {cand}")


setup(
    name='sgemm_ext',
    ext_modules=[
        CUDAExtension(
            name='sgemm_ext',
            sources=['sgemm_op.cu'],
            library_dirs=extra_lib_dirs,
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '-arch=sm_89', '--use_fast_math'],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
