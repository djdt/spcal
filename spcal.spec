# vim: set ft=python:
from pathlib import Path

version = "notfound"
with Path("spcal", "__init__.py").open() as fp:
    for line in fp:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"')

block_cipher = None

a = Analysis(
    [Path("spcal", "__main__.py")],
    binaries=None,
    datas=["spcal/resources/npdb.npz"],
    hiddenimports=["bottleneck"],
    hookspath=None,
    runtime_hooks=None,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    exclude_binaries=False,
    name=f"spcal_{version}",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx_exclude=["Qt*.dll", "PySide*.pyd"],
    console=False,
    # icon="app.ico",
)
