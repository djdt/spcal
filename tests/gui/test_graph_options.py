from pytestqt.qtbot import QtBot

from spcal.gui.dialogs.graphoptions import (
    CompositionsOptionsDialog,
    HistogramOptionsDialog,
    SpectraOptionsDialog,
)


def test_composition_options_dialog(qtbot: QtBot):
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


def test_histogram_options_dialog(qtbot: QtBot):
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


# def test_scatter_options_dialog(qtbot: QtBot):
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


def test_sepctra_options_dialog(qtbot: QtBot):
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
