from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup
from setuptools.command.build_ext import new_compiler

compiler = new_compiler().compiler_type

if compiler == "msvc":
    cxx_std = "latest"
    extra_link_args = []
else:
    cxx_std = "23"
    extra_link_args = ["-ltbb"]

spcalext = Pybind11Extension(
    "spcal.lib.spcalext",
    sources=["src/spcalext.cpp"],
    cxx_std=cxx_std,
    extra_link_args=extra_link_args,
)

setup(
    packages=find_packages(include=["spcal", "spcal.*"]),
    cmd_class={"build_ext": build_ext},
    ext_modules=[spcalext],
)
