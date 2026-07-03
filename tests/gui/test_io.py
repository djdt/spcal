from spcal.gui.dialogs.io.tofwerk import TofwerkImportDialog
from spcal.gui.dialogs.io.nu import NuImportDialog
from spcal.gui.dialogs.io.text import TextImportDialog
from pytestqt.qtbot import QtBot
from PySide6 import QtWidgets
from pathlib import Path
from spcal.gui.io import is_spcal_session_path, get_import_dialog_for_path


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

    dlg = get_import_dialog_for_path(window, test_data_path.joinpath("nu"))
    assert isinstance(dlg, NuImportDialog)

    dlg = get_import_dialog_for_path(
        window, test_data_path.joinpath("tofwerk/tofwerk_au_50nm.h5")
    )
    assert isinstance(dlg, TofwerkImportDialog)
