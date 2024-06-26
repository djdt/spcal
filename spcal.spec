# vim: set ft=python:
import importlib.metadata
from pathlib import Path

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
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    name=f"spcal_{importlib.metadata.version('spcal')}",
    debug=False,
    upx_exclude=["Qt*.dll", "PySide*.pyd"],
    console=False,
    icon="app.ico",
)
