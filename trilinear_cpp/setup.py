from setuptools import setup
import os, torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

CUDA_HOME = os.environ.get("CUDA_HOME", os.environ.get("CONDA_PREFIX", ""))
TORCH_LIB = os.path.join(os.path.dirname(torch.__file__), "lib")
ABI = int(getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", 0))

def make_cuda_ext():
    include_dirs = [os.path.join(CUDA_HOME, "include"),
                    os.path.join(CUDA_HOME, "targets", "x86_64-linux", "include")]
    library_dirs = [os.path.join(CUDA_HOME, "lib64"),
                    os.path.join(CUDA_HOME, "lib"),
                    os.path.join(CUDA_HOME, "targets", "x86_64-linux", "lib")]
    return CUDAExtension(
        name="trilinear",
        sources=["src/trilinear_cuda.cpp", "src/trilinear_kernel.cu"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        extra_compile_args={
            "cxx":  [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}", "-O3", "-std=c++17",
                     f"-Wl,-rpath,{TORCH_LIB}"],
            "nvcc": ["-O3", "-std=c++17", "--expt-relaxed-constexpr"],
        },
        runtime_library_dirs=[TORCH_LIB],   # Linux-only; safe here
    )

if torch.cuda.is_available():
    setup(name="trilinear", ext_modules=[make_cuda_ext()], cmdclass={"build_ext": BuildExtension})
else:
    setup(name="trilinear",
          ext_modules=[CppExtension("trilinear", ["src/trilinear.cpp"],
                                    extra_compile_args=[f"-D_GLIBCXX_USE_CXX11_ABI={ABI}", "-O3", "-std=c++17",
                                                        f"-Wl,-rpath,{TORCH_LIB}"],
                                    runtime_library_dirs=[TORCH_LIB])],
          cmdclass={"build_ext": BuildExtension})