from pathlib import Path
from PySide6 import QtCore
from pytestqt.qtbot import QtBot

from spcal.datafile import SPCalNuDataFile
from spcal.gui.dialogs.calculator import CalculatorDialog
from spcal.gui.dialogs.graphoptions import (
    CompositionsOptionsDialog,
    HistogramOptionsDialog,
    SpectraOptionsDialog,
)
from spcal.gui.dialogs.selectisotope import ScreeningOptionsDialog, SelectIsotopesDialog


from spcal.isotope import ISOTOPE_TABLE, SPCalIsotopeExpression
from spcal.processing.method import SPCalProcessingMethod


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


def test_graph_composition_options_dialog(qtbot: QtBot):
    dlg = CompositionsOptionsDialog(minimum_size=100.0, mode="pie")
    qtbot.add_widget(dlg)
    with qtbot.wait_exposed(dlg):
        dlg.show()

    assert dlg.combo_mode.currentText() == "Pie"
    assert dlg.lineedit_size.text() == "100.0"

    # No change, no signal
    with qtbot.assert_not_emitted(dlg.optionsChanged, wait=100):
        dlg.apply()

    dlg.combo_mode.setCurrentText("Bar")
    dlg.lineedit_size.setText("1%")

    with qtbot.wait_signal(
        dlg.optionsChanged,
        timeout=100,
        check_params_cb=lambda size, mode: size == "1%" and mode == "bar",
    ):
        dlg.apply()

    # Reset to default values
    dlg.reset()

    with qtbot.wait_signal(
        dlg.optionsChanged,
        timeout=100,
        check_params_cb=lambda size, mode: size == "5%" and mode == "pie",
    ):
        dlg.apply()

    dlg.accept()


def test_graph_histogram_options_dialog(qtbot: QtBot):
    dlg = HistogramOptionsDialog(
        bin_widths={
            "signal": None,
            "mass": None,
            "size": None,
            "volume": None,
        },
        percentile=98.0,
        draw_filtered=False,
    )
    qtbot.add_widget(dlg)
    with qtbot.wait_exposed(dlg):
        dlg.show()

    # No change, no signal
    with qtbot.assert_not_emitted(dlg.optionsChanged, wait=100):
        dlg.apply()

    dlg.width_signal.setBaseValue(100.0)
    dlg.width_mass.setBaseValue(1e-19)
    dlg.width_size.setBaseValue(1e-9)
    dlg.width_volume.setBaseValue(1e-6)
    dlg.spinbox_percentile.setValue(90)
    dlg.check_draw_filtered.setChecked(True)

    def check_bin_widths(widths: dict) -> bool:
        if widths["signal"] != 100.0:
            return False
        if widths["mass"] != 1e-19:
            return False
        if widths["size"] != 1e-9:
            return False
        if widths["volume"] != 1e-6:
            return False
        return True

    with qtbot.wait_signal(
        dlg.optionsChanged,
        timeout=100,
        check_params_cb=lambda widths, perc, draw: check_bin_widths(widths)
        and perc == 90.0
        and draw,
    ):
        dlg.apply()

    # Reset to default values
    dlg.reset()

    with qtbot.wait_signal(
        dlg.optionsChanged,
        timeout=100,
        check_params_cb=lambda widths, perc, draw: all(
            x is None for x in widths.values()
        )
        and perc == 98.0
        and not draw,
    ):
        dlg.apply()

    dlg.accept()


# def test_graph_scatter_options_dialog(qtbot: QtBot):
#     dlg = ScatterOptionsDialog(weighting="none", draw_filtered=False)
#     qtbot.add_widget(dlg)
#     with qtbot.wait_exposed(dlg):
#         dlg.show()
#
#     # No change, no signal
#     with qtbot.assert_not_emitted(dlg.weightingChanged, wait=100):
#         dlg.apply()
#
#     dlg.combo_weighting.setCurrentText("1/x")
#     dlg.check_draw_filtered.setChecked(True)
#
#     with qtbot.wait_signals(
#         [dlg.weightingChanged, dlg.drawFilteredChanged],
#         timeout=100,
#         check_params_cbs=[lambda w: w == "1/x", lambda b: b],
#     ):
#         dlg.apply()
#
#     # Reset to default values
#     dlg.reset()
#
#     with qtbot.wait_signals(
#         [dlg.weightingChanged, dlg.drawFilteredChanged],
#         timeout=100,
#         check_params_cbs=[lambda w: w == "none", lambda b: not b],
#     ):
#         dlg.apply()
#     dlg.accept()


def test_graph_spectra_options_dialog(qtbot: QtBot):
    dlg = SpectraOptionsDialog(True)
    qtbot.add_widget(dlg)
    with qtbot.wait_exposed(dlg):
        dlg.show()

    with qtbot.assert_not_emitted(dlg.optionsChanged, wait=100):
        dlg.apply()

    dlg.check_subtract_background.click()

    with qtbot.wait_signal(
        dlg.optionsChanged, timeout=100, check_params_cb=lambda sub: not sub
    ):
        dlg.apply()

    dlg.reset()

    assert dlg.check_subtract_background.isChecked()

    dlg.accept()


def test_select_isotope_dialog(
    test_data_path: Path, default_method: SPCalProcessingMethod, qtbot: QtBot
):
    df = SPCalNuDataFile.load(test_data_path.joinpath("nu"))
    df.selected_isotopes = [
        ISOTOPE_TABLE[("Ag", 107)],
        ISOTOPE_TABLE[("Ag", 109)],
        ISOTOPE_TABLE[("Au", 197)],
    ]
    dlg = SelectIsotopesDialog(df, default_method)
    qtbot.addWidget(dlg)

    with qtbot.waitExposed(dlg):
        dlg.show()

    assert dlg.table.selectedIsotopes() == df.selected_isotopes

    assert len(dlg.table.enabledIsotopes()) == 188

    dlg.screenDataFile(1000000, 1000000, False)
    assert dlg.table.selectedIsotopes() == df.selected_isotopes

    dlg.screenDataFile(1000, 1000000, True)
    assert len(dlg.table.selectedIsotopes()) == 21

    with qtbot.waitSignal(dlg.accepted, timeout=100):
        dlg.accept()

    assert len(df.selected_isotopes) == 21


def test_select_isotope_screening_dialog(qtbot: QtBot):
    dlg = ScreeningOptionsDialog(100, 1000, False)
    qtbot.addWidget(dlg)

    with qtbot.waitExposed(dlg):
        dlg.show()

    assert dlg.screening_ppm.value() == 100
    assert dlg.screening_size.value() == 1000
    assert not dlg.checkbox_replace_isotopes.isChecked()

    dlg.screening_ppm.setValue(200)
    dlg.screening_size.setValue(2000)
    dlg.checkbox_replace_isotopes.setCheckState(QtCore.Qt.CheckState.Unchecked)

    with qtbot.waitSignal(
        dlg.screeningOptionsSelected,
        check_params_cb=lambda ppm, size, replace: ppm == 200
        and size == 2000
        and not replace,
        timeout=100,
    ):
        dlg.accept()
