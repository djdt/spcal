from spcal.gui.dialogs.io.tofwerk import TofwerkImportDialog
from spcal.gui.dialogs.io.nu import NuImportDialog
from spcal.gui.dialogs.io.text import TextImportDialog
from pytestqt.qtbot import QtBot
from PySide6 import QtWidgets, QtCore
from pathlib import Path
from spcal.gui.io import (
    is_spcal_session_path,
    get_import_dialog_for_path,
    most_recent_spcal_path,
    SessionImportWorker,
)


def test_is_spcal_session_path():
    assert is_spcal_session_path(Path("/fake/session.spcal.json"))
    assert not is_spcal_session_path(Path("/fake/session.json"))
    assert not is_spcal_session_path(Path("/fake/spcal.json"))
    assert not is_spcal_session_path(Path("/fake/session"))
    assert not is_spcal_session_path(Path("/fake/session.csv"))


def test_get_import_dialog_for_path(qtbot: QtBot, test_data_path: Path):
    window = QtWidgets.QMainWindow()
    qtbot.addWidget(window)

    dlg = get_import_dialog_for_path(
        window, test_data_path.joinpath("text/agilent_au50nm.csv")
    )
    assert isinstance(dlg, TextImportDialog)

    dlg = get_import_dialog_for_path(window, test_data_path.joinpath("nu/normal"))
    assert isinstance(dlg, NuImportDialog)

    dlg = get_import_dialog_for_path(
        window, test_data_path.joinpath("nu/normal/run.info")
    )
    assert isinstance(dlg, NuImportDialog)

    dlg = get_import_dialog_for_path(
        window, test_data_path.joinpath("tofwerk/tofwerk_au_50nm.h5")
    )
    assert isinstance(dlg, TofwerkImportDialog)


def test_most_recent_spcal_path():
    settings = QtCore.QSettings()
    settings.remove("RecentFiles")

    path = most_recent_spcal_path()
    assert path is None

    settings.beginWriteArray("RecentFiles")
    settings.setArrayIndex(0)
    settings.setValue("Path", "/most/recent.path")
    settings.endArray()

    path = most_recent_spcal_path()
    assert path == Path("/most/recent.path")


def test_session_import_worker(qtbot: QtBot, test_data_path: Path):
    file_dict = {
        "format": "text",
        "selected isotopes": ["Au197"],
        "isotope table": {"197Au": "Au197"},
        "delimiter": ",",
        "skip row": 4,
        "cps": False,
        "override event time": None,
        "drop fields": ["Time_[Sec]"],
    }
    worker = SessionImportWorker(
        [(file_dict, test_data_path.joinpath("text/agilent_au50nm.csv"))]
    )

    with qtbot.waitSignals(
        [worker.started, worker.datafileImported, worker.finished], timeout=101
    ):
        worker.read()
