# setup.py

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='rmsnorm_cuda',
    ext_modules=[
        CUDAExtension(
            name='rmsnorm_cuda',
            sources=['rmsnorm.cpp', 'rmsnorm_kernel.cu']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
