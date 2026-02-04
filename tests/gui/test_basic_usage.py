from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtCore
from pytestqt.qtbot import QtBot

from spcal.datafile import SPCalTextDataFile
from spcal.gui.mainwindow import SPCalMainWindow


def _click_through_mainwindow(qtbot: QtBot, win: SPCalMainWindow):
    assert win.instrument_options.options_widget.uptake.value() is None
    qtbot.keyClick(
        win.instrument_options.options_widget.uptake._value, QtCore.Qt.Key.Key_1
    )
    assert win.instrument_options.options_widget.uptake.value() == 1.0

    assert win.instrument_options.options_widget.efficiency.value() is None
    qtbot.keyClick(
        win.instrument_options.options_widget.efficiency, QtCore.Qt.Key.Key_1
    )
    assert win.instrument_options.options_widget.efficiency.value() == 1.0
    # limit options
    assert not win.limit_options.options_widget.check_window.isChecked()
    assert not win.limit_options.options_widget.window_size.isEnabled()
    assert win.limit_options.options_widget.window_size.value() == 1000
    qtbot.mouseClick(
        win.limit_options.options_widget.check_window, QtCore.Qt.MouseButton.LeftButton
    )
    assert win.limit_options.options_widget.check_window.isChecked()
    assert win.limit_options.options_widget.window_size.isEnabled()
    qtbot.keyClick(win.limit_options.options_widget.window_size, QtCore.Qt.Key.Key_End)
    qtbot.keyClick(
        win.limit_options.options_widget.window_size, QtCore.Qt.Key.Key_Backspace
    )
    assert win.limit_options.options_widget.window_size.value() == 100

    for i in range(1, win.limit_options.options_widget.limit_method.count()):
        win.limit_options.options_widget.limit_method.setCurrentIndex(i)
    win.limit_options.options_widget.limit_method.setCurrentIndex(0)
    assert not win.limit_options.options_widget.check_iterative.isChecked()
    qtbot.mouseClick(
        win.limit_options.options_widget.check_iterative,
        QtCore.Qt.MouseButton.LeftButton,
    )
    assert win.limit_options.options_widget.check_iterative.isChecked()

    qtbot.keyClicks(win.limit_options.options_widget.gaussian.alpha, "1e-3")
    assert np.isclose(win.limit_options.options_widget.gaussian.alpha.value(), 1e-3)  # type: ignore
    assert np.isclose(win.limit_options.options_widget.gaussian.sigma.value(), 3.0902)  # type: ignore

    qtbot.keyClicks(win.limit_options.options_widget.poisson.alpha, "1e-4")
    assert np.isclose(win.limit_options.options_widget.poisson.alpha.value(), 1e-4)  # type: ignore

    qtbot.keyClicks(win.limit_options.options_widget.compound.alpha, "1e-4")
    assert np.isclose(win.limit_options.options_widget.compound.alpha.value(), 1e-4)  # type: ignore
    qtbot.keyClick(
        win.limit_options.options_widget.compound.lognormal_sigma, QtCore.Qt.Key.Key_1
    )
    assert np.isclose(
        win.limit_options.options_widget.compound.lognormal_sigma.value(),  # type: ignore
        1.0,
    )

    for action in win.toolbar_view.view_actions.values():
        qtbot.mouseClick(
            win.toolbar_view.widgetForAction(action), QtCore.Qt.MouseButton.LeftButton
        )


@pytest.mark.parametrize(
    "test_locales",
    [
        QtCore.QLocale.Language.English,
        QtCore.QLocale.Language.Spanish,
        QtCore.QLocale.Language.German,
    ],
    indirect=True,
)
def test_gui_no_data(qtbot: QtBot, test_locales):
    win = SPCalMainWindow()
    qtbot.addWidget(win)
    with qtbot.waitExposed(win):
        win.show()

    _click_through_mainwindow(qtbot, win)


def test_gui_single_quad_data(qtbot: QtBot, test_locales, test_data_path):
    win = SPCalMainWindow()
    qtbot.addWidget(win)
    with qtbot.waitExposed(win):
        win.show()

    df = SPCalTextDataFile.load(
        test_data_path.joinpath("text/agilent_au50nm.csv"), skip_rows=4
    )
    df.selected_isotopes = df.isotopes

    win.files.addDataFile(df)

    _click_through_mainwindow(qtbot, win)
