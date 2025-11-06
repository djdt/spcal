from pathlib import Path

from PySide6 import QtCore, QtWidgets

from spcal.datafile import SPCalDataFile
from spcal.gui.dialogs.io import (
    ImportDialogBase,
    NuImportDialog,
    TextImportDialog,
    TofwerkImportDialog,
)
from spcal.io.nu import is_nu_directory, is_nu_run_info_file
from spcal.io.text import is_text_file
from spcal.io.tofwerk import is_tofwerk_file
from spcal.processing import SPCalProcessingMethod

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
    dir = str(QtCore.QSettings().value("RecentFiles/1/Path", ""))
    if dir != "":
        dir = str(Path(dir).parent)

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
    dir = str(QtCore.QSettings().value("RecentFiles/1/Path", ""))
    if dir != "":
        dir = str(Path(dir).parent)

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
    data_file: SPCalDataFile | None = None,
    screening_method: SPCalProcessingMethod | None = None,
) -> ImportDialogBase:
    if path.is_dir():
        if is_nu_directory(path):
            dlg = NuImportDialog(path, data_file, parent=parent)
        else:
            raise FileNotFoundError("getImportDialogForPath: invalid directory.")
    elif is_nu_run_info_file(path):
        dlg = NuImportDialog(path.parent, data_file, parent=parent)
    elif is_tofwerk_file(path):
        dlg = TofwerkImportDialog(path, data_file, parent=parent)
    elif is_text_file(path):
        dlg = TextImportDialog(path, data_file, parent=parent)
    else:
        raise FileNotFoundError("getImportDialogForPath: invalid file.")

    # if import_options is not None:
    #     try:
    #         dlg.setImportOptions(
    #             import_options, path=False, dwelltime=dlg.dwelltime.value() is None
    #         )
    #     except (ValueError, KeyError):
    #         pass
    # if screening_options is not None:
    #     dlg.setScreeningOptions(screening_options)
    return dlg
