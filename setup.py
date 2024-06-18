import numpy
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import new_compiler

compiler = new_compiler().compiler_type

if compiler == "msvc":
    extra_compile_args = ["/openmp"]
    extra_link_args = ["/openmp"]
else:
    extra_compile_args = ["-fopenmp"]
    extra_link_args = ["-fopenmp"]

spcalext = Extension(
    "spcal.lib.spcalext",
    sources=["src/sort.c", "src/spcalext.c"],
    include_dirs=["include", numpy.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_2_0_API_VERSION")],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(
    packages=find_packages(include=["spcal", "spcal.*"]),
    ext_modules=[spcalext],
)
