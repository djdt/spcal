from PySide6 import QtCore
from pytestqt.qtbot import QtBot

from spcal.gui.dialogs.calculator import CalculatorDialog
from spcal.isotope import ISOTOPE_TABLE, SPCalIsotopeExpression


def test_calculator_dialog(qtbot: QtBot):
    isotopes = [
        ISOTOPE_TABLE[("B", 10)],
        ISOTOPE_TABLE[("Ag", 107)],
        ISOTOPE_TABLE[("Au", 197)],
    ]

    dlg = CalculatorDialog(isotopes, [])
    qtbot.addWidget(dlg)
    with qtbot.waitExposed(dlg):
        dlg.show()

    assert dlg.combo_isotope.currentText() == "Isotopes"
    assert dlg.combo_isotope.count() == 3 + 1
    assert dlg.formula.toPlainText() == ""
    assert not dlg.isComplete()

    dlg.combo_isotope.activated.emit(1)
    assert dlg.formula.toPlainText() == "10B"
    assert dlg.isComplete()

    dlg.combo_isotope.activated.emit(2)
    assert dlg.formula.toPlainText() == "10B107Ag"
    assert not dlg.isComplete()

    dlg.formula.setText("10B + 107Ag")
    assert dlg.isComplete()
    dlg.button_add.pressed.emit()

    assert dlg.expressions.count() == 1
    assert dlg.formula.toPlainText() == ""

    dlg.formula.setText("107Ag / 197Au")

    def check_expressions(exprs: list[SPCalIsotopeExpression]) -> bool:
        if not len(exprs) == 2:
            return False
        if not exprs[0].name == "(+ 10B 107Ag)":
            return False
        if not exprs[1].name == "(/ 107Ag 197Au)":
            return False
        return True

    with qtbot.waitSignal(
        dlg.expressionsChanged, check_params_cb=check_expressions, timeout=100
    ):
        dlg.accept()


def test_calculator_dialog_existing_expr(qtbot: QtBot):
    isotopes = [
        ISOTOPE_TABLE[("Ag", 107)],
        ISOTOPE_TABLE[("Ag", 109)],
    ]
    exprs = [SPCalIsotopeExpression("sum Ag", ("+", isotopes[0], isotopes[1]))]

    dlg = CalculatorDialog(isotopes, exprs)
    qtbot.addWidget(dlg)
    with qtbot.waitExposed(dlg):
        dlg.show()

    assert dlg.expressions.count() == 1
    assert dlg.expressions.item(0).text() == "sum Ag: + 107Ag 109Ag"
    qtbot.mouseClick(
        dlg.expressions.viewport(),
        QtCore.Qt.MouseButton.LeftButton,
        pos=dlg.expressions.visualItemRect(dlg.expressions.item(0)).center(),
    )
    qtbot.keyClick(dlg.expressions.viewport(), QtCore.Qt.Key.Key_Delete)
    assert dlg.expressions.count() == 0

    def check_expressions(exprs: list[SPCalIsotopeExpression]) -> bool:
        return len(exprs) == 0

    with qtbot.waitSignal(
        dlg.expressionsChanged, check_params_cb=check_expressions, timeout=100
    ):
        dlg.accept()
