from pathlib import Path

import numpy as np
from PySide6 import QtCore
from pytestqt.qtbot import QtBot

from spcal.datafile import SPCalNuDataFile, SPCalTOFWERKDataFile, SPCalTextDataFile
from spcal.gui.dialogs.io import (
    NuImportDialog,
    TextImportDialog,
    TofwerkImportDialog,
)
from spcal.isotope import ISOTOPE_TABLE
from spcal.processing.method import SPCalProcessingMethod


def test_import_dialog_text_nu(test_data_path: Path, qtbot: QtBot):
    def check_data(data_file: SPCalTextDataFile):
        if not str(data_file.selected_isotopes[0]) == "107Ag":
            return False
        if not str(data_file.selected_isotopes[1]) == "197Au":
            return False
        if not data_file.num_events == 999:
            return False
        if not list(data_file.isotope_table.values()) == [
            "106.905_-_seg_Full_mass_spectrum_att_1",
            "196.967_-_seg_Full_mass_spectrum_att_1",
        ]:
            return False
        if not data_file.event_time != 1e-5:
            return False
        return True

    path = test_data_path.joinpath("text/nu_export_auag.csv")
    dlg = TextImportDialog(path)
    qtbot.add_widget(dlg)
    with qtbot.wait_exposed(dlg):
        dlg.open()

    # Defaults loaded from file
    assert np.isclose(dlg.event_time.baseValue(), 4.852e-05)  # type: ignore
    assert dlg.combo_intensity_units.currentText() == "Counts"
    assert dlg.delimiter() == ","
    assert dlg.spinbox_first_line.value() == 1
    assert dlg.useColumns() == [1, 2, 3]

    # Disable Ag109
    pos = QtCore.QPoint(
        dlg.table_header.sectionPosition(2) + dlg.table_header.sectionSize(2) // 2,
        dlg.table_header.sizeHint().height() // 2,
    )
    with qtbot.wait_signal(dlg.table_header.checkStateChanged, timeout=2000):
        qtbot.mouseClick(
            dlg.table_header.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=pos
        )
    assert dlg.useColumns() == [1, 3]

    assert not dlg.isComplete()
    # Names are bad, correct
    dlg.table.item(0, 1).setText("107Ag")  # type: ignore , not None
    dlg.table.item(0, 3).setText("197Au")  # type: ignore , not None

    assert dlg.isComplete()
    dlg.event_time.setBaseValue(None)
    dlg.override_event_time.setChecked(True)
    assert not dlg.isComplete()
    dlg.event_time.setBaseValue(1e-5)
    assert dlg.isComplete()

    with qtbot.wait_signal(dlg.dataImported, check_params_cb=check_data, timeout=100):
        dlg.accept()


def test_import_dialog_text_tofwerk(test_data_path: Path, qtbot: QtBot):
    def check_data(data_file: SPCalTOFWERKDataFile):
        if not len(data_file.isotopes) == 1:
            return False
        if not str(data_file.isotopes[0]) == "197Au":
            return False
        if data_file.num_events != 999:
            return False
        return True

    path = test_data_path.joinpath("text/tofwerk_export_au.csv")
    dlg = TextImportDialog(path)
    with qtbot.wait_exposed(dlg):
        dlg.open()

    # Defaults loaded from file
    assert np.isclose(dlg.event_time.baseValue(), 1e-3, atol=1e-4)  # type: ignore
    assert dlg.combo_intensity_units.currentText() == "Counts"
    assert dlg.delimiter() == ","
    assert dlg.spinbox_first_line.value() == 1
    assert dlg.useColumns() == [2]

    with qtbot.wait_signal(dlg.dataImported, check_params_cb=check_data, timeout=100):
        dlg.accept()


def test_import_dialog_text_thermo_new_icap(test_data_path: Path, qtbot: QtBot):
    path = test_data_path.joinpath("text/thermo_icap_export.csv")
    dlg = TextImportDialog(path)
    with qtbot.wait_exposed(dlg):
        dlg.open()

    assert np.isclose(dlg.event_time.baseValue(), 50e-6)  # type: ignore

    def check_data(data_file: SPCalTOFWERKDataFile):
        if not len(data_file.isotopes) == 1:
            return False
        if not str(data_file.isotopes[0]) == "80Se":
            return False
        return True

    with qtbot.wait_signal(dlg.dataImported, check_params_cb=check_data, timeout=100):
        dlg.accept()


def test_import_dialog_nu(test_data_path: Path, qtbot: QtBot):
    def check_data(data_file: SPCalNuDataFile):
        if not len(data_file.isotopes) == 188:
            return False
        if not str(data_file.selected_isotopes[0]) == "107Ag":
            return False
        if not str(data_file.selected_isotopes[1]) == "197Au":
            return False
        if data_file.num_events != 40:
            return False
        if not np.isclose(data_file.event_time, 9.824e-5):
            return False

        return True

    path = test_data_path.joinpath("nu")
    dlg = NuImportDialog(path)
    qtbot.add_widget(dlg)
    with qtbot.wait_exposed(dlg):
        dlg.open()

    assert dlg.cycle_number.minimum() == 0  # Auto
    assert dlg.cycle_number.maximum() == 1
    assert dlg.segment_number.minimum() == 0  # Auto
    assert dlg.segment_number.maximum() == 1

    for symbol in ["H", "Na", "Ar", "K", "As", "Tc", "Po", "Pm", "Ac"]:
        assert not dlg.table.buttons[symbol].isEnabled()
    for symbol in ["Se", "Mo", "Ru", "Bi", "La", "Nd", "Sm", "Lu"]:
        assert dlg.table.buttons[symbol].isEnabled()

    assert not dlg.isComplete()

    qtbot.mouseClick(dlg.table.buttons["Au"], QtCore.Qt.MouseButton.LeftButton)
    qtbot.mouseClick(dlg.table.buttons["Ag"], QtCore.Qt.MouseButton.LeftButton)

    assert dlg.isComplete()

    with qtbot.wait_signal(dlg.dataImported, check_params_cb=check_data, timeout=100):
        dlg.accept()


def test_import_dialog_nu_screening(
    test_data_path: Path, default_method: SPCalProcessingMethod, qtbot: QtBot
):
    path = test_data_path.joinpath("nu")
    dlg = NuImportDialog(path)
    qtbot.add_widget(dlg)
    with qtbot.wait_exposed(dlg):
        dlg.open()

    dlg.screening_method = default_method
    dlg.screenDataFile(100, 1000, True)

    # todo: get some better test data
    assert dlg.table.selectedIsotopes() == []


def test_import_dialog_tofwerk(test_data_path: Path, qtbot: QtBot):
    def check_data(data_file: SPCalTOFWERKDataFile):
        if not str(data_file.selected_isotopes[0]) == "107Ag":
            return False
        if not str(data_file.selected_isotopes[1]) == "197Au":
            return False
        if data_file.num_events != 4895:
            return False
        if data_file.event_time != 0.0184:  # wrong but ok for test
            return False
        return True

    path = test_data_path.joinpath("tofwerk/tofwerk_testdata.h5")
    dlg = TofwerkImportDialog(path)
    qtbot.add_widget(dlg)
    with qtbot.wait_exposed(dlg):
        dlg.open()

    for symbol in [
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

    qtbot.mouseClick(dlg.table.buttons["Au"], QtCore.Qt.MouseButton.LeftButton)
    qtbot.mouseClick(dlg.table.buttons["Ag"], QtCore.Qt.MouseButton.LeftButton)

    with qtbot.wait_signal(dlg.dataImported, check_params_cb=check_data, timeout=100):
        dlg.accept()


def test_import_dialog_tofwerk_screening(
    test_data_path: Path, default_method: SPCalProcessingMethod, qtbot: QtBot
):
    path = test_data_path.joinpath("tofwerk/tofwerk_testdata.h5")
    dlg = TofwerkImportDialog(path)
    qtbot.add_widget(dlg)
    with qtbot.wait_exposed(dlg):
        dlg.open()

    dlg.screening_method = default_method
    dlg.screenDataFile(100, 1000, True)

    assert dlg.table.selectedIsotopes() == [ISOTOPE_TABLE[("Ru", 101)]]
