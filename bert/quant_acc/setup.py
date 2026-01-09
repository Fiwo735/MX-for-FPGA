import torch
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")

setup(
    name="ord_matmul",
    ext_modules=[
        CUDAExtension(
            "ordmm",
            ["ordmm.cu"],
            depends=[
                "ordmm_chunk_bcast.cuh",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
            extra_link_args=[
                f"-Wl,-rpath,{torch_lib_path}"
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
