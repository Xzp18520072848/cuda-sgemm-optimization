from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='add_cuda',
    ext_modules=[
        CUDAExtension('add_cuda', [
            'add_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)