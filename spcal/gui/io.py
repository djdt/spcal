from pathlib import Path

from PySide6 import QtCore, QtWidgets

from spcal.gui.dialogs._import import (
    NuImportDialog,
    TextImportDialog,
    TofwerkImportDialog,
    _ImportDialogBase,
)
from spcal.io.nu import is_nu_directory, is_nu_run_info_file
from spcal.io.text import is_text_file
from spcal.io.tofwerk import is_tofwerk_file

np_file_filters = (
    "NP Data Files (*.csv *.info *.h5);;"
    "CSV Documents(*.csv *.txt *.text);;"
    "Nu Instruments(*.info);;"
    "TOFWERK HDF5(*.h5);;"
    "All files(*)"
)


def is_spcal_path(path: str | Path) -> bool:
    path = Path(path)

    if is_nu_run_info_file(path):
        path = path.parent

    return any((is_text_file(path), is_nu_directory(path), is_tofwerk_file(path)))


def get_open_spcal_path(
    parent: QtWidgets.QWidget, title: str = "Open File"
) -> Path | None:
    dir = QtCore.QSettings().value("RecentFiles/1/Path", None)
    dir = str(Path(dir).parent) if dir is not None else ""

    path, _ = QtWidgets.QFileDialog.getOpenFileName(parent, title, dir, np_file_filters)
    if path == "":
        return None

    path = Path(path)
    if is_nu_run_info_file(path):
        path = path.parent
    return path


def get_open_spcal_paths(
    parent: QtWidgets.QWidget, title: str = "Open Files"
) -> list[Path]:
    dir = QtCore.QSettings().value("RecentFiles/1/Path", None)
    dir = str(Path(dir).parent) if dir is not None else ""

    files, _ = QtWidgets.QFileDialog.getOpenFileNames(
        parent, title, dir, np_file_filters
    )

    paths = []
    for file in files:
        path = Path(file)
        if is_nu_run_info_file(path):  # use directory
            path = path.parent
        paths.append(path)
    return paths


def get_import_dialog_for_path(
    parent: QtWidgets.QWidget,
    path: Path,
    import_options: dict | None = None,
) -> _ImportDialogBase:
    if path.is_dir():
        if is_nu_directory(path):
            dlg = NuImportDialog(path, parent)
        else:
            raise FileNotFoundError("getImportDialogForPath: invalid directory.")
    elif is_nu_run_info_file(path):
        dlg = NuImportDialog(path.parent, parent)
    elif is_tofwerk_file(path):
        dlg = TofwerkImportDialog(path, parent)
    elif is_text_file(path):
        dlg = TextImportDialog(path, parent)
    else:
        raise FileNotFoundError("getImportDialogForPath: invalid file.")

    if import_options is not None:
        try:
            dlg.setImportOptions(import_options, path=False, dwelltime=False)
        except (ValueError, KeyError):
            pass
    return dlg
