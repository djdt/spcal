from pybind11.setup_helpers import Pybind11Extension
from setuptools import find_packages, setup

spcalext = Pybind11Extension(
    "spcal.lib.spcalext",
    sources=["src/spcalext.cpp"],
    cxx_std=23,
)

setup(
    packages=find_packages(include=["spcal", "spcal.*"]),
    ext_modules=[spcalext],
)
