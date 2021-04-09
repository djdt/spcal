# vim: set ft=python:
from pathlib import Path

with Path("nanopart", "__init__.py").open() as fp:
    for line in fp:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"')

block_cipher = None

a = Analysis(
    [Path("nanopart", "__main__.py")],
    binaries=None,
    datas=[],
    hiddenimports=[],
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
    name="nanopart" + "_" + version,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    # icon="app.ico",
)
