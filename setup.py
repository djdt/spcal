from setuptools import find_packages, setup
from pybind11.setup_helpers import Pybind11Extension

# compiler = new_compiler().compiler_type

# if compiler == "msvc":
#     extra_compile_args = ["/openmp:llvm"]
#     extra_link_args = ["/openmp:llvm"]
# else:
#     extra_compile_args = ["-fopenmp"]
#     extra_link_args = ["-fopenmp"]
#
# spcalext = Extension(
#     "spcal.lib.spcalext",
#     sources=["src/sort.c", "src/spcalext.c"],
#     include_dirs=["include", numpy.get_include()],
#     define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
#     extra_compile_args=extra_compile_args,
#     extra_link_args=extra_link_args,
# )

spcalext = Pybind11Extension(
        "spcal.lib.spcalext",
        sources=["src/spcalext.cpp"],
        cxx_std=23,
)

setup(
    packages=find_packages(include=["spcal", "spcal.*"]),
    ext_modules=[spcalext],
)
