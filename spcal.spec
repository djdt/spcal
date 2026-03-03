# vim: set ft=python:
import argparse
import importlib.metadata
import os
import logging
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--bundle", action="store_true")
parser.add_argument("--debug", action="store_true")
opts = parser.parse_args()

logging.info(f"name={os.name}")
for k, v in os.environ.items():
    logging.info(f"{k}={v}")

if os.name == "nt":
    binarys = [os.environ["LIB"] + "tbb*.dll"]
else:
    binarys = []

a = Analysis(
    [Path("spcal", "__main__.py")],
    binarys=binarys,
    datas=[
        ("spcal/resources/app.ico", "spcal/resources"),
        ("spcal/resources/npdb.npz", "spcal/resources"),
        ("spcal/resources/cpln_quantiles.npz", "spcal/resources"),
    ],
    hiddenimports=["bottleneck"],
)
pyz = PYZ(a.pure)

name = f"spcal_{importlib.metadata.version('spcal')}"

if opts.debug:
    exe = EXE(pyz, a.scripts, exclude_binaries=True, name=name)
    coll = COLLECT(exe, a.binaries, a.datas, name=name + "_debug")
elif opts.bundle:
    exe = EXE(
        pyz,
        a.scripts,
        exclude_binaries=True,
        name=name,
        debug=False,
        upx_exclude=["Qt*.dll", "PySide*.pyd"],
        console=False,
        icon="app.ico",
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        name=name,
        upx_exclude=["Qt*.dll", "PySide*.pyd"],
    )
    app = BUNDLE(
        coll,
        name="spcal.app",
        icon="app.ico",
        bundle_identifier="com.spcal.spcal",
        version=importlib.metadata.version("spcal"),
        info_plist={
            "NSPrincipalClass": "NSApplication",
            "NSAppleScriptEnabled": False,
            "CFBundleDocumentTypes": [
                {
                    "CFBundleTypeExtensions": [".txt", ".csv", ".text"],
                    "CFBundleTypeName": "SP Text Data",
                    "CFBundleTypeRole": "Viewer",
                    "LSItemContentTypes": ["public.text"],
                    "LSHandlerRank": "Default",
                },
                {
                    "CFBundleTypeExtensions": [".info"],
                    "CFBundleTypeName": "Nu Instruments Data",
                    "CFBundleTypeRole": "Viewer",
                    "LSItemContentTypes": ["public.data"],
                    "LSHandlerRank": "Default",
                },
                {
                    "CFBundleTypeExtensions": [".h5"],
                    "CFBundleTypeName": "TOFWERK Data",
                    "CFBundleTypeRole": "Viewer",
                    "LSItemContentTypes": ["public.data"],
                    "LSHandlerRank": "Default",
                },
            ],
        },
    )
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
