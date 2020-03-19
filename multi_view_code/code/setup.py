from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='renderer',
    ext_modules=[
        CUDAExtension('renderer', [
            'renderer.cpp',
            'renderer_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
})
