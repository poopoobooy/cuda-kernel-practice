"""PyTorch CUDA extension 构建脚本

用法:
  cd torch_ext
  python setup.py build_ext --inplace     # 编译, 产物 sgemm_ext*.pyd
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='sgemm_ext',
    ext_modules=[
        CUDAExtension(
            name='sgemm_ext',
            sources=['sgemm_op.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '-arch=sm_89', '--use_fast_math'],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
