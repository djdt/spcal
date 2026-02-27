from typing import Callable
import numpy as np
from pathlib import Path
from PySide6 import QtCore, QtWidgets
from pytestqt.qtbot import QtBot

from spcal.datafile import SPCalTOFWERKDataFile
from spcal.gui.dialogs.calculator import CalculatorDialog
from spcal.gui.dialogs.filter import (
    BooleanItemWidget,
    FilterDialog,
    FilterItemWidget,
    ClusterFilterItemWidget,
)
from spcal.gui.dialogs.graphoptions import (
    CompositionsOptionsDialog,
    HistogramOptionsDialog,
    SpectraOptionsDialog,
)
from spcal.gui.dialogs.advancedoptions import AdvancedPoissonDialog
from spcal.gui.dialogs.peakproperties import PeakPropertiesDialog
from spcal.gui.dialogs.processingoptions import ProcessingOptionsDialog
from spcal.gui.dialogs.export import ExportDialog
from spcal.gui.dialogs.response import ResponseDialog
from spcal.gui.dialogs.selectisotope import ScreeningOptionsDialog, SelectIsotopesDialog
from spcal.gui.dialogs.singleion import SingleIonDialog
from spcal.gui.dialogs.tools import (
    MassFractionCalculatorDialog,
    ParticleDatabaseDialog,
    TransportEfficiencyDialog,
)

from spcal.isotope import ISOTOPE_TABLE, SPCalIsotopeExpression
from spcal.processing.filter import SPCalClusterFilter, SPCalValueFilter
from spcal.processing.method import SPCalProcessingMethod
from spcal.processing.options import SPCalIsotopeOptions, SPCalProcessingOptions


def test_advanced_poisson_dialog(qtbot: QtBot):
    dlg = AdvancedPoissonDialog("currie", 2.1, 0.6, 1.2, 1.1)
    qtbot.addWidget(dlg)
    with qtbot.waitExposed(dlg):
        dlg.show()

    assert dlg.poisson_formula.currentText() == "Currie"

    assert dlg.currie.epsilon.value() == 0.6
    assert dlg.currie.eta.value() == 2.1

    assert dlg.formula_a.t_blank.value() == 1.1
    assert dlg.formula_a.t_sample.value() == 1.2
    assert dlg.formula_c.t_blank.value() == 1.1
    assert dlg.formula_c.t_sample.value() == 1.2
    assert dlg.stapleton.t_blank.value() == 1.1
    assert dlg.stapleton.t_sample.value() == 1.2

    assert dlg.isComplete()

    dlg.currie.eta.setValue(None)

    assert not dlg.isComplete()

    dlg.poisson_formula.setCurrentText("Formula C")

    assert dlg.isComplete()

    dlg.formula_c.t_blank.setValue(2.0)

    with qtbot.waitSignal(
        dlg.optionsSelected,
        check_params_cb=lambda f, o1, o2: f == "formula c" and o1 == 1.2 and o2 == 2.0,
        timeout=100,
    ):
        dlg.accept()


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

def test_export_dialog(qtbot: QtBot):
    # dlg = ExportDialog([], {}, {})
    # qtbot.addWidget(dlg)
    # with qtbot.wait_exposed(dlg):
    #     dlg.show()
    raise NotImplementedError



def test_filter_dialog_empty(qtbot: QtBot):
    isotopes = [
        ISOTOPE_TABLE[("Ag", 107)],
        ISOTOPE_TABLE[("Ag", 109)],
        ISOTOPE_TABLE[("Au", 197)],
    ]
    dlg = FilterDialog(isotopes, [[]], [[]], number_clusters=10)  # type: ignore
    qtbot.addWidget(dlg)
    with qtbot.wait_exposed(dlg):
        dlg.show()

    # Will all collapse
    dlg.action_add.trigger()
    dlg.action_or.trigger()
    dlg.action_add.trigger()
    dlg.action_add.trigger()
    dlg.action_cluster_add.trigger()

    filter = dlg.list.itemWidget(dlg.list.item(0))
    assert isinstance(filter, FilterItemWidget)
    filter.value.setBaseValue(1.0)

    assert filter.isotopes.count() == 3

    filter = dlg.cluster_list.itemWidget(dlg.cluster_list.item(0))
    assert isinstance(filter, ClusterFilterItemWidget)
    filter.index.setValue(1)

    assert filter.index.maximum() == 10

    def check_filters(
        filters: list[list[SPCalValueFilter]],
        cluster_filters: list[list[SPCalClusterFilter]],
    ) -> bool:
        if len(filters) != 2:
            return False
        if len(filters[0]) != 1:
            return False
        if len(filters[1]) != 2:
            return False
        if filters[0][0].value != 1.0:
            return False
        if filters[1][0].value != 0.0:
            return False
        if len(cluster_filters) != 1:
            return False
        if len(cluster_filters[0]) != 1:
            return False
        if cluster_filters[0][0].index != 1:
            return False
        return True

    with qtbot.wait_signal(
        dlg.filtersChanged, check_params_cb=check_filters, timeout=100
    ):
        dlg.accept()


def test_filter_dialog_filters(qtbot: QtBot):
    isotopes = [
        ISOTOPE_TABLE[("Ag", 107)],
        ISOTOPE_TABLE[("Ag", 109)],
        ISOTOPE_TABLE[("Au", 197)],
    ]
    filters = [
        [
            SPCalValueFilter(isotopes[0], "signal", np.greater, 100.0),
            SPCalValueFilter(isotopes[1], "size", np.less, 200.0),
        ],
        [
            SPCalValueFilter(isotopes[2], "mass", np.less_equal, 10.0),
        ],
    ]
    dlg = FilterDialog(isotopes, filters, [[SPCalClusterFilter("mass", 7)]], 10)  # type: ignore
    qtbot.addWidget(dlg)
    with qtbot.wait_exposed(dlg):
        dlg.show()

    # Check loaded correctly
    for i in range(dlg.list.count()):
        w = dlg.list.itemWidget(dlg.list.item(i))
        if i == 0:
            assert isinstance(w, FilterItemWidget)
            assert w.isotopes.currentText() == "107Ag"
            assert w.key.currentText() == "Intensity"
            assert w.operation.currentText() == ">"
            assert w.value.unit() == "cts"
            assert w.value.baseValue() == 100.0
        elif i == 1:
            assert isinstance(w, FilterItemWidget)
            assert w.isotopes.currentText() == "109Ag"
            assert w.key.currentText() == "Size"
            assert w.operation.currentText() == "<"
            assert w.value.unit() == "m"
            assert w.value.baseValue() == 200.0
        elif i == 2:
            assert isinstance(w, BooleanItemWidget)
        elif i == 3:
            assert isinstance(w, FilterItemWidget)

    w = dlg.cluster_list.itemWidget(dlg.cluster_list.item(0))
    assert isinstance(w, ClusterFilterItemWidget)
    assert w.index.value() == 7

    def check_filters(
        filters: list[list[SPCalValueFilter]],
        cluster_filters: list[list[SPCalClusterFilter]],
    ) -> bool:
        if len(filters) != 2:
            return False
        if len(filters[0]) != 2:
            return False
        if filters[0][0].value != 100.0:
            return False
        if filters[0][1].value != 200.0:
            return False
        if filters[1][0].value != 10.0:
            return False
        if len(cluster_filters) != 1 or len(cluster_filters[0]) != 1:
            return False
        if cluster_filters[0][0].key != "mass" or cluster_filters[0][0].index != 7:
            return False
        return True

    with qtbot.wait_signal(
        dlg.filtersChanged, check_params_cb=check_filters, timeout=100
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
        check_params_cb=lambda widths, perc, draw: (
            check_bin_widths(widths) and perc == 90.0 and draw
        ),
    ):
        dlg.apply()

    # Reset to default values
    dlg.reset()

    with qtbot.wait_signal(
        dlg.optionsChanged,
        timeout=100,
        check_params_cb=lambda widths, perc, draw: (
            all(x is None for x in widths.values()) and perc == 98.0 and not draw
        ),
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


def test_peak_properties_dialog(
    qtbot: QtBot, default_method: SPCalProcessingMethod, random_result_generator
):
    results = {
        iso: random_result_generator(default_method, number=100, isotope=iso)
        for iso in [ISOTOPE_TABLE[("Fe", 56)], ISOTOPE_TABLE[("Cu", 63)]]
    }

    dlg = PeakPropertiesDialog(results, ISOTOPE_TABLE[("Cu", 63)])  # type: ignore
    qtbot.addWidget(dlg)

    with qtbot.waitExposed(dlg):
        dlg.show()

    assert dlg.combo_isotope.currentText() == "63Cu"

    for iso, result in results.items():
        dlg.combo_isotope.setCurrentIsotope(iso)
        widths = result.times[result.regions[:, 1]] - result.times[result.regions[:, 0]]
        assert np.isclose(
            float(dlg.model.item(0, 2).text()), np.mean(widths / 1e-6), atol=0.01
        )


def test_processing_options_dialog(qtbot: QtBot):
    options = SPCalProcessingOptions()
    dlg = ProcessingOptionsDialog(options)
    qtbot.addWidget(dlg)

    with qtbot.waitExposed(dlg):
        dlg.show()

    with qtbot.assertNotEmitted(dlg.optionsChanged):
        dlg.accept()

    dlg.options.points_required.setValue(2)

    with qtbot.waitSignal(dlg.optionsChanged, timeout=100):
        dlg.accept()


def test_select_isotope_dialog(
    test_data_path: Path, default_method: SPCalProcessingMethod, qtbot: QtBot
):
    df = SPCalTOFWERKDataFile.load(
        test_data_path.joinpath("tofwerk/tofwerk_testdata.h5")
    )
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

    assert len(dlg.table.enabledIsotopes()) == len(df.isotopes)

    dlg.screenDataFile(1000000, 1000000, False)
    assert dlg.table.selectedIsotopes() == df.selected_isotopes

    dlg.screenDataFile(1000, 1000000, False)
    assert len(dlg.table.selectedIsotopes()) == 4

    dlg.screenDataFile(1000, 1000000, True)
    assert len(dlg.table.selectedIsotopes()) == 1

    with qtbot.waitSignal(dlg.accepted, timeout=100):
        dlg.accept()

    assert len(df.selected_isotopes) == 1
    assert df.selected_isotopes[0] == ISOTOPE_TABLE[("Ru", 101)]


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
        check_params_cb=lambda ppm, size, replace: (
            ppm == 200 and size == 2000 and not replace
        ),
        timeout=100,
    ):
        dlg.accept()


def test_response_dialog(qtbot: QtBot, random_datafile_gen: Callable):
    dlg = ResponseDialog()
    qtbot.addWidget(dlg)

    with qtbot.waitExposed(dlg):
        dlg.show()

    dlg.reset()

    df = random_datafile_gen(
        isotopes=[
            ISOTOPE_TABLE[("Fe", 56)],
            ISOTOPE_TABLE[("Cu", 63)],
            ISOTOPE_TABLE[("Zn", 66)],
        ],
        number=0,
        seed=79826,
    )

    dlg.addDataFile(df)
    assert dlg.model_concs.columnCount() == 3
    assert dlg.model_concs.rowCount() == 1
    assert dlg.model_intensity.columnCount() == 3
    assert dlg.model_intensity.rowCount() == 1
    dlg.model_concs.setData(
        dlg.model_concs.index(0, 0), 1.0, QtCore.Qt.ItemDataRole.EditRole
    )

    df = random_datafile_gen(
        lam=10.0,
        isotopes=[
            ISOTOPE_TABLE[("Fe", 56)],
            ISOTOPE_TABLE[("Cu", 63)],
            ISOTOPE_TABLE[("Zn", 66)],
        ],
        number=0,
        seed=79827,
    )

    dlg.addDataFile(df)
    assert dlg.model_concs.rowCount() == 2
    assert dlg.model_intensity.rowCount() == 2

    dlg.model_concs.setData(
        dlg.model_concs.index(1, 0), 10.0, QtCore.Qt.ItemDataRole.EditRole
    )
    dlg.model_concs.setData(
        dlg.model_concs.index(1, 1), 10.0, QtCore.Qt.ItemDataRole.EditRole
    )

    dlg.combo_unit.setCurrentText("mg/L")

    def check_response(responses: dict):
        if len(responses) != 2:
            return False
        if not np.isclose(responses[ISOTOPE_TABLE[("Fe", 56)]], 1e6, rtol=0.05):
            return False
        if not np.isclose(responses[ISOTOPE_TABLE[("Cu", 63)]], 1e6, rtol=0.05):
            return False
        return True

    #
    with qtbot.wait_signal(
        dlg.responsesSelected, timeout=100, check_params_cb=check_response
    ):
        dlg.accept()


def test_response_dialog_save(
    tmp_path: Path, qtbot: QtBot, random_datafile_gen: Callable
):
    dlg = ResponseDialog()
    qtbot.add_widget(dlg)

    df = random_datafile_gen(
        isotopes=[
            ISOTOPE_TABLE[("Fe", 56)],
            ISOTOPE_TABLE[("Cu", 63)],
            ISOTOPE_TABLE[("Zn", 66)],
        ]
    )

    dlg.addDataFile(df)

    df2 = random_datafile_gen(
        isotopes=[
            ISOTOPE_TABLE[("Fe", 56)],
            ISOTOPE_TABLE[("Cu", 63)],
            ISOTOPE_TABLE[("Zn", 66)],
        ]
    )

    dlg.addDataFile(df2)

    dlg.model_concs.setData(
        dlg.model_concs.index(0, 0), 0.0, QtCore.Qt.ItemDataRole.EditRole
    )
    dlg.model_concs.setData(
        dlg.model_concs.index(0, 1), 1.0, QtCore.Qt.ItemDataRole.EditRole
    )
    dlg.model_concs.setData(
        dlg.model_concs.index(0, 2), 2.0, QtCore.Qt.ItemDataRole.EditRole
    )

    dlg.model_concs.setData(
        dlg.model_concs.index(1, 0), 1.0, QtCore.Qt.ItemDataRole.EditRole
    )
    dlg.model_concs.setData(
        dlg.model_concs.index(1, 2), 2.0, QtCore.Qt.ItemDataRole.EditRole
    )

    path = tmp_path.joinpath("test_response_dialog.csv")
    dlg.saveToFile(path)

    # todo: test output, may change


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

    with qtbot.wait_signal(
        dlg.densitySelected, timeout=100, check_params_cb=lambda d: d == 4430.0
    ):
        dlg.accept()


def test_transport_efficiency_dialog(
    qtbot: QtBot,
    random_datafile_gen,
    default_method: SPCalProcessingMethod,
):
    default_method.instrument_options.uptake = 1.0
    default_method.isotope_options[ISOTOPE_TABLE[("Au", 197)]] = SPCalIsotopeOptions(
        1.0, 1.0, None, diameter=10.0
    )
    df = random_datafile_gen(isotopes=[ISOTOPE_TABLE[("Au", 197)]])
    result = default_method.processDataFile(df)

    dlg = TransportEfficiencyDialog(
        df, ISOTOPE_TABLE[("Au", 197)], result[ISOTOPE_TABLE[("Au", 197)]]
    )

    qtbot.addWidget(dlg)
    with qtbot.waitExposed(dlg):
        dlg.show()

    assert dlg.diameter.baseValue() == 10.0
    assert dlg.density.baseValue() == 1.0
    assert dlg.concentration.baseValue() is None
    assert dlg.response.baseValue() == 1.0
    assert dlg.mass_fraction.value() is None

    assert not dlg.isComplete()

    with qtbot.waitSignal(dlg.efficencyChanged, timeout=100):
        dlg.mass_fraction.setValue(1.0)

    assert dlg.isComplete()
    assert dlg.efficiency.value() is not None
    assert dlg.mass_response.value() is not None

    with qtbot.waitSignals(
        [dlg.efficiencySelected, dlg.massResponseSelected, dlg.isotopeOptionsChanged],
        check_params_cbs=[
            lambda eff: eff is not None,
            lambda mr: mr is not None,
            lambda _, opt: opt.mass_fraction == 1.0,
        ],
        timeout=100,
    ):
        dlg.accept()


def test_single_ion_dialog(test_data_path: Path, qtbot: QtBot):
    params = np.empty(100, dtype=[("mass", float), ("mu", float), ("sigma", float)])
    params["mass"] = np.random.uniform(-0.001, 0.001, size=100) + np.arange(30, 130)
    params["mu"] = np.random.uniform(6.0, 8.0, size=100)
    params["sigma"] = np.linspace(0.5, 0.7, 100)

    dlg = SingleIonDialog(params)
    qtbot.addWidget(dlg)

    with qtbot.waitExposed(dlg):
        dlg.show()

    # View only
    assert not dlg.isComplete()
    assert not dlg.controls_box.isEnabled()

    # test nu loads
    dlg.loadSingleIonData(test_data_path.joinpath("nu"))

    dlg.loadSingleIonData(test_data_path.joinpath("tofwerk/tofwerk_testdata.h5"))
    assert dlg.mus.size > 0
    assert dlg.sigmas.size > 0
    q1, q3 = np.nanpercentile(dlg.sigmas, [25, 75])
    assert q1 > 0.3 and q3 < 0.6
    assert dlg.masses.size == 309

    assert np.count_nonzero(dlg.valid) == 260
    dlg.max_sigma_difference.setValue(0.2)
    assert np.count_nonzero(dlg.valid) == 285

    with qtbot.waitSignal(dlg.parametersExtracted, timeout=100):
        dlg.accept()

    with qtbot.waitSignal(dlg.resetRequested, timeout=100):
        dlg.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Reset).click()
