from PySide6 import QtCore
from pytestqt.qtbot import QtBot

from spcal.gui.dialogs.graphoptions import (
    CompositionsOptionsDialog,
    HistogramOptionsDialog,
    ScatterOptionsDialog,
)


def test_composition_options_dialog(qtbot: QtBot):
    dlg = CompositionsOptionsDialog(distance=1.0, minimum_size=100.0)
    qtbot.add_widget(dlg)
    with qtbot.wait_exposed(dlg):
        dlg.show()

    assert dlg.lineedit_distance.text() == "100.0"
    assert dlg.lineedit_size.text() == "100.0"

    # No change, no signal
    with qtbot.assert_not_emitted(dlg.distanceChanged, wait=100):
        dlg.apply()

    dlg.lineedit_distance.setText("10.0")
    dlg.lineedit_size.setText("1%")

    with qtbot.wait_signals(
        [dlg.distanceChanged, dlg.minimumSizeChanged],
        timeout=100,
        check_params_cbs=[lambda d: d == 0.1, lambda s: s == "1%"],
    ):
        dlg.apply()

    # Reset to default values
    dlg.reset()

    with qtbot.wait_signals(
        [dlg.distanceChanged, dlg.minimumSizeChanged],
        timeout=100,
        check_params_cbs=[lambda d: d == 0.03, lambda s: s == "5%"],
    ):
        dlg.apply()
    dlg.accept()


def test_histogram_options_dialog(qtbot: QtBot):
    dlg = HistogramOptionsDialog(
        fit="log normal",
        bin_widths={
            "signal": None,
            "mass": None,
            "size": None,
            "volume": None,
        },
        percentile=95,
        draw_filtered=False,
    )
    qtbot.add_widget(dlg)
    with qtbot.wait_exposed(dlg):
        dlg.show()

    # No change, no signal
    with qtbot.assert_not_emitted(dlg.fitChanged, wait=100):
        dlg.apply()

    qtbot.mouseClick(dlg.radio_fit_norm, QtCore.Qt.MouseButton.LeftButton)
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

    with qtbot.wait_signals(
        [
            dlg.fitChanged,
            dlg.binWidthsChanged,
            dlg.percentileChanged,
            dlg.drawFilteredChanged,
        ],
        timeout=100,
        check_params_cbs=[
            lambda f: f == "normal",
            check_bin_widths,
            lambda p: p == 90,
            lambda b: b,
        ],
    ):
        dlg.apply()

    # Reset to default values
    dlg.reset()

    with qtbot.wait_signals(
        [dlg.fitChanged, dlg.binWidthsChanged, dlg.drawFilteredChanged],
        timeout=100,
        check_params_cbs=[
            lambda f: f == "log normal",
            lambda d: all(x is None for x in d.values()),
            lambda b: not b,
        ],
    ):
        dlg.apply()
    dlg.accept()


def test_scatter_options_dialog(qtbot: QtBot):
    dlg = ScatterOptionsDialog(weighting="none", draw_filtered=False)
    qtbot.add_widget(dlg)
    with qtbot.wait_exposed(dlg):
        dlg.show()

    # No change, no signal
    with qtbot.assert_not_emitted(dlg.weightingChanged, wait=100):
        dlg.apply()

    dlg.combo_weighting.setCurrentText("1/x")
    dlg.check_draw_filtered.setChecked(True)

    with qtbot.wait_signals(
        [dlg.weightingChanged, dlg.drawFilteredChanged],
        timeout=100,
        check_params_cbs=[lambda w: w == "1/x", lambda b: b],
    ):
        dlg.apply()

    # Reset to default values
    dlg.reset()

    with qtbot.wait_signals(
        [dlg.weightingChanged, dlg.drawFilteredChanged],
        timeout=100,
        check_params_cbs=[lambda w: w == "none", lambda b: not b],
    ):
        dlg.apply()
    dlg.accept()
