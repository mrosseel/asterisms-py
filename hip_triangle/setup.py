from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# HIP uses the same extension mechanism as CUDA in PyTorch
# ROCm provides hipcc which handles the compilation
setup(
    name='triangle_hip',
    ext_modules=[
        CUDAExtension(
            name='triangle_hip',
            sources=[
                'hip_binding.cpp',
                'triangle_kernel.hip',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-lineinfo',
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
