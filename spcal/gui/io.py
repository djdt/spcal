from pathlib import Path
from typing import List

from PySide6 import QtCore, QtWidgets

from spcal.gui.dialogs._import import (
    ImportDialog,
    NuImportDialog,
    TofwerkImportDialog,
    _ImportDialogBase,
)
from spcal.io.nu import is_nu_directory
from spcal.io.text import is_text_file
from spcal.io.tofwerk import is_tofwerk_file

np_file_filters = (
    "NP Data Files (*.csv *.info *.h5);;"
    "CSV Documents(*.csv *.txt *.text);;"
    "Nu Instruments(*.info);;"
    "TOFWERK HDF5(*.h5);;"
    "All files(*)"
)


def getOpenNanoparticleFile(
    parent: QtWidgets.QWidget, title: str = "Open File"
) -> Path | None:
    dir = QtCore.QSettings().value("RecentFiles/1/path", None)
    dir = str(Path(dir).parent) if dir is not None else ""

    path, _ = QtWidgets.QFileDialog.getOpenFileName(parent, title, dir, np_file_filters)
    if path == "":
        return None

    path = Path(path)
    if path.suffix == ".info":  # Cast down for nu
        path = path.parent
    return path


def getOpenNanoparticleFiles(
    parent: QtWidgets.QWidget, title: str = "Open Files"
) -> List[Path]:
    dir = QtCore.QSettings().value("RecentFiles/1/path", None)
    dir = str(Path(dir).parent) if dir is not None else ""

    files, _ = QtWidgets.QFileDialog.getOpenFileNames(
        parent, title, dir, np_file_filters
    )

    paths = []
    for file in files:
        path = Path(file)
        if path.suffix == ".info":  # Cast down for nu
            path = path.parent
        paths.append(path)
    return paths


def getImportDialogForPath(parent: QtWidgets.QWidget, path: Path) -> _ImportDialogBase:
    if path.is_dir():
        if is_nu_directory(path):
            return NuImportDialog(path, parent)
        else:
            raise FileNotFoundError("getImportDialogForPath: invalid directory.")
    elif is_tofwerk_file(path):
        return TofwerkImportDialog(path, parent)
    elif is_text_file(path):
        return ImportDialog(path, parent)
    else:
        raise FileNotFoundError("getImportDialogForPath: invalid file.")