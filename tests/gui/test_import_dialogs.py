from pathlib import Path

import numpy as np
from pytestqt.qtbot import QtBot

from spcal.gui.dialogs._import import ImportDialog, NuImportDialog, TofwerkImportDialog


def test_import_dialog_text_nu(qtbot: QtBot):
    def check_data(data: np.ndarray, options: dict):
        if data.dtype.names != ("Ag", "Au"):
            return False
        if data.size != 999:
            return False
        if options["old names"] != [
            "106.905 - seg Full mass spectrum att 1",
            "196.967 - seg Full mass spectrum att 1",
        ]:
            return False
        if options["dwelltime"] != 1e-5:
            return False
        return True

    path = Path(__file__).parent.parent.joinpath("data/text/nu_export_auag.csv")
    dlg = ImportDialog(path)
    with qtbot.wait_exposed(dlg):
        dlg.open()

    # Defaults loaded from file
    assert np.isclose(dlg.dwelltime.baseValue(), 4.9e-05)  # type: ignore
    assert dlg.combo_intensity_units.currentText() == "Counts"
    assert dlg.delimiter() == ","
    assert dlg.spinbox_first_line.value() == 1
    assert dlg.ignoreColumns() == [0]

    dlg.dwelltime.setBaseValue(None)
    assert not dlg.isComplete()
    dlg.dwelltime.setBaseValue(1e-5)
    assert dlg.isComplete()

    # Change some params
    dlg.le_ignore_columns.setText("1;3;")
    assert dlg.ignoreColumns() == [0, 2]
    dlg.table.item(0, 1).setText("Ag")
    dlg.table.item(0, 3).setText("Au")

    with qtbot.wait_signal(dlg.dataImported, check_params_cb=check_data, timeout=100):
        dlg.accept()


def test_import_dialog_text_tofwerk(qtbot: QtBot):
    def check_data(data: np.ndarray, options: dict):
        if data.dtype.names != ("[197Au]+ (cts)",):
            return False
        if data.size != 999:
            return False
        return True

    path = Path(__file__).parent.parent.joinpath("data/text/tofwerk_export_au.csv")
    dlg = ImportDialog(path)
    with qtbot.wait_exposed(dlg):
        dlg.open()

    # Defaults loaded from file
    assert np.isclose(dlg.dwelltime.baseValue(), 1e-3)  # type: ignore
    assert dlg.combo_intensity_units.currentText() == "Counts"
    assert dlg.delimiter() == ","
    assert dlg.spinbox_first_line.value() == 1
    assert dlg.ignoreColumns() == [0, 1]

    with qtbot.wait_signal(dlg.dataImported, check_params_cb=check_data, timeout=100):
        dlg.accept()


def test_import_dialog_nu(qtbot: QtBot):
    def check_data(data: np.ndarray, options: dict):
        if data.dtype.names != ("[197Au]+ (cts)",):
            return False
        if data.size != 999:
            return False
        return True

    path = Path(__file__).parent.parent.joinpath("data/nu/run.info")
    pass


def test_import_dialog_tofwerk(qtbot: QtBot):
    def check_data(data: np.ndarray, options: dict):
        if data.dtype.names != ("[197Au]+ (cts)",):
            return False
        if data.size != 999:
            return False
        return True

    path = Path(__file__).parent.parent.joinpath("data/tofwerk/tofwerk_au_50nm.h5")
    pass
