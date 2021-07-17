from pathlib import Path
from setuptools import setup, find_packages

with open("README.md") as fp:
    long_description = fp.read()

with Path("nanopart", "__init__.py").open() as fp:
    version = None
    for line in fp:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"')
    if version is None:
        raise ValueError("Could not read version from __init__.py")

setup(
    name="nanopart",
    version=version,
    description="Processing of spICP-MS data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="T. Lockwood",
    author_email="thomas.lockwood@uts.edu.au",
    url="https://github.com/djdt/nanopart",
    project_urls={
        "Source": "https://gtihub.com/djdt/nanopart",
    },
    packages=find_packages(include=["nanopart", "nanpart.*"]),
    install_requires=["numpy", "PySide2", "bottleneck"],
    extras_require=["pytest"],  # for testing
    entry_points={"console_scripts": ["nanopart=nanopart.__main__:main"]},
)
