from pathlib import Path
from setuptools import setup, find_packages

with open("README.md") as fp:
    long_description = fp.read()

with Path("spcal", "__init__.py").open() as fp:
    version = None
    for line in fp:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"')
    if version is None:
        raise ValueError("Could not read version from __init__.py")

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
    install_requires=["numpy", "PySide6", "bottleneck"],
    extras_require={"tests": ["pytest"]},
    entry_points={"console_scripts": ["spcal=spcal.__main__:main"]},
)
