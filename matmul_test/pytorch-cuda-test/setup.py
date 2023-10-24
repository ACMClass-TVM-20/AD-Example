from setuptools import setup, Extension
from torch.utils import cpp_extension

nvcc_flags = [
    "-std=c++17",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-use_fast_math",
]


setup(
    name="my_extension",
    ext_modules=[
        cpp_extension.CUDAExtension(
            "my_extension",
            ["my_extension.cpp", "my_cuda.cu"],
            extra_compile_args={
                "nvcc": nvcc_flags,
            },
        ),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
