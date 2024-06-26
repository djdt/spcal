# vim: set ft=python:
import argparse
import importlib.metadata
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true")
opts = parser.parse_args()

a = Analysis(
    [Path("spcal", "__main__.py")],
    binaries=[("C:\\Windows\\system32\\libomp*.dll", ".")],
    datas=[
        ("spcal/resources/app.ico", "spcal/resources"),
        ("spcal/resources/npdb.npz", "spcal/resources"),
    ],
    hiddenimports=["bottleneck"],
)
pyz = PYZ(a.pure)

name = f"spcal_{importlib.metadata.version('spcal')}"

if opts.debug:
    exe = EXE(pyz, a.scripts, exclude_binaries=True, name=name)
    coll = COLLECT(exe, a.binaries, a.datas, name=name + "_debug")
else:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        name=name,
        debug=False,
        upx_exclude=["Qt*.dll", "PySide*.pyd"],
        console=False,
        icon="app.ico",
    )
