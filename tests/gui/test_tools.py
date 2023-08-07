import numpy as np
from PySide6 import QtCore
from pytestqt.qtbot import QtBot

from spcal.gui.dialogs.tools import MassFractionCalculatorDialog, ParticleDatabaseDialog


def test_mass_fraction_calculator(qtbot: QtBot):
    dlg = MassFractionCalculatorDialog()
    qtbot.add_widget(dlg)
    with qtbot.wait_exposed(dlg):
        dlg.show()

    assert not dlg.isComplete()

    qtbot.keyClick(dlg.lineedit_formula, "A")
    assert not dlg.isComplete()  # A
    qtbot.keyClick(dlg.lineedit_formula, "G")
    assert not dlg.isComplete()  # AG
    qtbot.keyClick(dlg.lineedit_formula, QtCore.Qt.Key.Key_Backspace)
    qtbot.keyClick(dlg.lineedit_formula, "g")
    assert dlg.lineedit_formula.text() == "Ag"
    assert dlg.isComplete()  # Ag
    assert dlg.label_mw.text() == "MW = 107.9 g/mol"
    assert np.isclose(dlg.ratios["Ag"], 1.0)
    qtbot.keyClick(dlg.lineedit_formula, "C")
    qtbot.keyClick(dlg.lineedit_formula, "l")
    assert dlg.isComplete()  # AgCl
    assert dlg.label_mw.text() == "MW = 143.3 g/mol"
    assert np.isclose(dlg.ratios["Ag"], 0.7526, atol=1e-4)
    assert np.isclose(dlg.ratios["Cl"], 0.2474, atol=1e-4)
    qtbot.keyClick(dlg.lineedit_formula, "C")
    qtbot.keyClick(dlg.lineedit_formula, "l")
    assert dlg.isComplete()  # AgCl2
    assert dlg.label_mw.text() == "MW = 178.8 g/mol"
    assert np.isclose(dlg.ratios["Ag"], 0.6034, atol=1e-4)
    assert np.isclose(dlg.ratios["Cl"], 0.3966, atol=1e-4)

    def check_ratios(ratios: list):
        if len(ratios) != 2:
            return False
        if ratios[0][0] != "Ag":
            return False
        if not np.isclose(ratios[0][1], 0.6034, atol=1e-4):
            return False
        if ratios[1][0] != "Cl":
            return False
        if not np.isclose(ratios[1][1], 0.3966, atol=1e-4):
            return False
        return True

    with qtbot.wait_signals(
        [dlg.ratiosSelected, dlg.molarMassSelected],
        timeout=100,
        check_params_cbs=[check_ratios, lambda m: np.isclose(m, 178.8, atol=0.1)],
    ):
        dlg.accept()


def test_particle_database(qtbot: QtBot):
    dlg = ParticleDatabaseDialog()
    qtbot.add_widget(dlg)
    with qtbot.wait_exposed(dlg):
        dlg.show()

    dlg.lineedit_search.setText("AgCl")

    assert dlg.proxy.data(dlg.proxy.index(0, 0)) == "AgCl"
    assert dlg.proxy.data(dlg.proxy.index(0, 1)) == "silver (i) chloride"
    assert dlg.proxy.data(dlg.proxy.index(0, 2)) == "7783-90-6"
    assert dlg.proxy.data(dlg.proxy.index(0, 3)) == "5.56"

    assert dlg.proxy.data(dlg.proxy.index(1, 0)) == "AgClO3"
    assert dlg.proxy.data(dlg.proxy.index(2, 0)) == "AgClO4"

    qtbot.mouseClick(
        dlg.table,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        dlg.table.visualRect(dlg.proxy.buddy(dlg.proxy.index(1, 0))).center(),
    )
    dlg.table.selectRow(1)

    with qtbot.wait_signals(
        [dlg.densitySelected, dlg.formulaSelected],
        timeout=1000,
        check_params_cbs=[lambda d: d == 4.43, lambda f: f == "AgClO3"],
    ):
        dlg.accept()
