from pathlib import Path

import numpy as np
from PySide6 import QtCore
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
    qtbot.add_widget(dlg)
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
        if data.dtype.names != ("Ag107", "Au197"):
            return False
        if data.size != 30:
            return False
        if options["dwelltime"] != 2.8e-5:
            return False
        return True

    path = Path(__file__).parent.parent.joinpath("data/nu")
    dlg = NuImportDialog(path)
    qtbot.add_widget(dlg)
    with qtbot.wait_exposed(dlg):
        dlg.open()

    assert np.isclose(dlg.dwelltime.baseValue(), 2.8e-5)  # type: ignore
    for symbol in ["H", "Ne", "At", "Hs", "Ac", "Am", "Lr"]:
        assert not dlg.table.buttons[symbol].isEnabled()
    for symbol in ["Na", "Au", "Po", "La", "Lu", "Th", "Pu"]:
        assert dlg.table.buttons[symbol].isEnabled()

    assert not dlg.isComplete()

    qtbot.mouseClick(
        dlg.table.buttons["Au"], QtCore.Qt.LeftButton, QtCore.Qt.NoModifier
    )
    qtbot.mouseClick(
        dlg.table.buttons["Ag"], QtCore.Qt.LeftButton, QtCore.Qt.NoModifier
    )

    assert dlg.isComplete()

    with qtbot.wait_signal(dlg.dataImported, check_params_cb=check_data, timeout=100):
        dlg.accept()


def test_import_dialog_tofwerk(qtbot: QtBot):
    def check_data(data: np.ndarray, options: dict):
        if data.dtype.names != ("OH+", "[107Ag]+", "[197Au]+"):
            return False
        if data.size != 200:
            return False
        if options["dwelltime"] != 1e-3:
            return False
        if options["other peaks"] != ["OH+"]:
            return False
        return True

    path = Path(__file__).parent.parent.joinpath("data/tofwerk/tofwerk_au_50nm.h5")
    dlg = TofwerkImportDialog(path)
    qtbot.add_widget(dlg)
    with qtbot.wait_exposed(dlg):
        dlg.open()

    assert np.isclose(dlg.dwelltime.baseValue(), 1e-3)  # type: ignore
    for symbol in [
        "H",
        "He",
        "F",
        "Ne",
        "Tc",
        "Po",
        "At",
        "Fr",
        "Hs",
        "Pm",
        "Ac",
        "Pa",
        "Np",
        "Lr",
    ]:
        assert not dlg.table.buttons[symbol].isEnabled()
    for symbol in [
        "Li",
        "O",
        "Na",
        "Mo",
        "Ru",
        "Bi",
        "Rn",
        "La",
        "Nd",
        "Sm",
        "Lu",
        "Th",
        "U",
    ]:
        assert dlg.table.buttons[symbol].isEnabled()

    assert not dlg.isComplete()

    qtbot.mouseClick(
        dlg.table.buttons["Au"], QtCore.Qt.LeftButton, QtCore.Qt.NoModifier
    )
    qtbot.mouseClick(
        dlg.table.buttons["Ag"], QtCore.Qt.LeftButton, QtCore.Qt.NoModifier
    )
    dlg.combo_other_peaks.model().item(1).setCheckState(QtCore.Qt.CheckState.Checked)

    assert dlg.combo_other_peaks.model().item(0).text().startswith("1")
    assert dlg.isComplete()

    with qtbot.wait_signal(dlg.dataImported, check_params_cb=check_data, timeout=100):
        dlg.accept()
