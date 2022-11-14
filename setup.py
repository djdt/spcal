from pathlib import Path

import numpy
from setuptools import Extension, find_packages, setup

with open("README.md") as fp:
    long_description = fp.read()

with Path("spcal", "__init__.py").open() as fp:
    version = None
    for line in fp:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"')
    if version is None:
        raise ValueError("Could not read version from __init__.py")

spcalext = Extension(
    "spcal.lib.spcalext",
    sources=["src/spcalext.c"],
    include_dirs=[numpy.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)

setup(
    name="spcal",
    version=version,
    description="Processing of spICP-MS data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="T. Lockwood",
    author_email="thomas.lockwood@uts.edu.au",
    url="https://github.com/djdt/spcal",
    project_urls={
        "Source": "https://gtihub.com/djdt/spcal",
    },
    packages=find_packages(include=["spcal", "spcal.*"]),
    install_requires=["numpy>=1.22", "PySide6", "pyqtgraph", "bottleneck"],
    extras_require={"tests": ["pytest"]},
    entry_points={"console_scripts": ["spcal=spcal.__main__:main"]},
    ext_modules=[spcalext],
)
