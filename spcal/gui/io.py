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


def most_recent_spcal_path() -> Path | None:
    settings = QtCore.QSettings()
    settings.beginReadArray("RecentFiles")
    settings.setArrayIndex(0)
    path = str(settings.value("Path", ""))
    settings.endArray()
    if path == "":
        return None
    else:
        return Path(path)


def get_save_spcal_path(
    parent: QtWidgets.QWidget,
    filters: list[tuple[str, str]],
    path: str | Path | None = None,
) -> Path | None:
    if path is None:
        recent = most_recent_spcal_path()
        if recent is not None:
            path = str(recent.parent)
        else:
            path = ""
    else:
        path = str(path)

    path, filter = QtWidgets.QFileDialog.getSaveFileName(
        parent,
        "Save File",
        path,
        ";;".join(f"{name} (*{ext})" for name, ext in filters),
    )
    if path == "":
        return None
    filter_ext = filter[filter.rfind(".") : -1]

    path = Path(path)
    if not any(path.stem.lower() == ext for _, ext in filters):
        path = path.with_suffix(path.suffix + filter_ext)
    return path


def get_open_spcal_path(
    parent: QtWidgets.QWidget, title: str = "Open File"
) -> Path | None:
    recent = most_recent_spcal_path()
    if recent is not None:
        dir = str(recent.parent)
    else:
        dir = ""

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
    recent = most_recent_spcal_path()
    if recent is not None:
        dir = str(recent.parent)
    else:
        dir = ""

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


def saveSession():
    pass


def loadSession():
    pass
