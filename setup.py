import numpy
from setuptools import Extension, find_packages, setup

spcalext = Extension(
    "spcal.lib.spcalext",
    sources=["src/sort.c", "src/spcalext.c"],
    include_dirs=["include", numpy.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    extra_compile_args=["-fopenmp"],
    extra_link_args=["-fopenmp"],
)

setup(
    packages=find_packages(include=["spcal", "spcal.*"]),
    ext_modules=[spcalext],
)
