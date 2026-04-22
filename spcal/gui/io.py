from spcal.io.session import decode_json_datafile
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from spcal.datafile import SPCalDataFile

from spcal.io.nu import is_nu_directory, is_nu_run_info_file
from spcal.io.text import is_text_file
from spcal.io.tofwerk import is_tofwerk_file
from spcal.processing.method import SPCalProcessingMethod

from spcal.gui.dialogs.io.base import ImportDialogBase
from spcal.gui.dialogs.io.nu import NuImportDialog
from spcal.gui.dialogs.io.text import TextImportDialog
from spcal.gui.dialogs.io.tofwerk import TofwerkImportDialog

NP_FILE_FILTER = "NP Data Files (*.csv *.info *.h5)"
TEXT_FILE_FILTER = "CSV Documents(*.csv *.txt *.text)"
NU_FILE_FILTER = "Nu Instruments(*.info)"
TOFWERK_FILE_FILTER = "TOFWERK HDF5(*.h5)"
NP_FILE_FILTERS = (
    ";;".join([NP_FILE_FILTER, TEXT_FILE_FILTER, NU_FILE_FILTER, TOFWERK_FILE_FILTER])
    + ";;All files(*)"
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
    if not any(path.suffix.lower() == ext for _, ext in filters):
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

    path, _ = QtWidgets.QFileDialog.getOpenFileName(parent, title, dir, NP_FILE_FILTERS)
    if path == "":
        return None

    path = Path(path)
    if is_nu_run_info_file(path):
        path = path.parent
    return path


def get_open_spcal_paths(
    parent: QtWidgets.QWidget,
    title: str = "Open Files",
    selected_filter: str = NP_FILE_FILTER,
) -> list[Path]:
    recent = most_recent_spcal_path()
    if recent is not None:
        dir = str(recent.parent)
    else:
        dir = ""

    files, _ = QtWidgets.QFileDialog.getOpenFileNames(
        parent, title, dir, NP_FILE_FILTERS, selected_filter
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
            dlg = NuImportDialog(path, data_file, screening_method, parent=parent)
        else:
            raise FileNotFoundError("getImportDialogForPath: invalid directory.")
    elif is_nu_run_info_file(path):
        dlg = NuImportDialog(path.parent, data_file, screening_method, parent=parent)
    elif is_tofwerk_file(path):
        dlg = TofwerkImportDialog(path, data_file, screening_method, parent=parent)
    elif is_text_file(path):
        dlg = TextImportDialog(path, data_file, parent=parent)
    else:
        raise FileNotFoundError("getImportDialogForPath: invalid file.")

    dlg.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
    return dlg


class SessionImportWorker(QtCore.QObject):
    started = QtCore.Signal(int)
    progress = QtCore.Signal(int, str)
    finished = QtCore.Signal()
    datafileImported = QtCore.Signal(object)

    def __init__(
        self, datafiles: list[tuple[dict, Path]], parent: QtCore.QObject | None = None
    ):
        super().__init__(parent)
        self.datafiles = datafiles

    def read(self):
        self.started.emit(len(self.datafiles))
        for i, (datafile, path) in enumerate(self.datafiles):
            if self.thread().isInterruptionRequested():
                return

            self.progress.emit(i, path.name)
            df = decode_json_datafile(datafile, path)
            if not self.thread().isInterruptionRequested():
                # prevent load if interrupted, but thread not complete
                self.datafileImported.emit(df)

        self.finished.emit()
