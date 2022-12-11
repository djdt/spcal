from pathlib import Path

import numpy as np
from PySide6 import QtCore
from pytestqt.qtbot import QtBot

from spcal.gui.inputs import InputWidget, ReferenceWidget, SampleWidget
from spcal.gui.main import SPCalWindow
from spcal.gui.options import OptionsWidget
from spcal.gui.results import ResultsWidget


def click_though_options(qtbot: QtBot, options: OptionsWidget):
    qtbot.keyClick(options.uptake.lineedit, QtCore.Qt.Key_1)
    assert options.uptake.value() == 1
    qtbot.keyClick(options.efficiency, QtCore.Qt.Key_1)
    assert options.efficiency.text() == "1"

    assert not options.window_size.isEnabled()
    with qtbot.wait_signal(options.check_use_window.toggled, timeout=50):
        options.check_use_window.click()
    assert options.check_use_window.isChecked()
    assert options.window_size.isEnabled()
    qtbot.keyClick(options.window_size, QtCore.Qt.Key_Backspace)
    assert options.window_size.text() == "99"

    qtbot.keyClick(options.error_rate_alpha, QtCore.Qt.Key_1)
    qtbot.keyClick(options.error_rate_beta, QtCore.Qt.Key_2)
    qtbot.keyClick(options.sigma, QtCore.Qt.Key_3)
    assert options.error_rate_alpha.text() == "0.051"
    assert options.error_rate_beta.text() == "0.052"
    assert options.sigma.text() == "5.03"

    assert not options.manual.isEnabled()
    options.method.setCurrentIndex(5)
    assert options.manual.isEnabled()
    qtbot.keyClick(options.manual, QtCore.Qt.Key_1)
    assert options.manual.text() == "10.01"

    qtbot.keyClick(options.celldiameter.lineedit, QtCore.Qt.Key_1)
    assert options.celldiameter.value() == 1


def click_though_input(qtbot: QtBot, input: InputWidget):
    for i, io in enumerate(input.io):
        input.io.combo_name.setCurrentIndex(i)
        qtbot.keyClick(io.density.lineedit, QtCore.Qt.Key_1)
        assert io.density.value() == 1
        qtbot.keyClick(io.molarmass.lineedit, QtCore.Qt.Key_1)
        assert io.molarmass.value() == 1
        qtbot.keyClick(io.response.lineedit, QtCore.Qt.Key_1)
        assert io.response.value() == 1
        qtbot.keyClick(io.massfraction, QtCore.Qt.Key_Backspace)
        assert io.massfraction.text() == "1."

        if isinstance(input, ReferenceWidget):
            qtbot.keyClick(io.concentration.lineedit, QtCore.Qt.Key_1)
            assert io.concentration.value() == 1
            qtbot.keyClick(io.diameter.lineedit, QtCore.Qt.Key_1)
            assert io.diameter.value() == 1
            io.check_use_efficiency_for_all.click()
            assert io.check_use_efficiency_for_all.isChecked()

    qtbot.mouseClick(
        input.graph_toolbar,
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
        input.graph_toolbar.actionGeometry(input.action_graph_overlay).center(),
    )
    qtbot.mouseClick(
        input.graph_toolbar,
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
        input.graph_toolbar.actionGeometry(input.action_graph_stacked).center(),
    )
    qtbot.mouseClick(
        input.graph_toolbar,
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
        input.graph_toolbar.actionGeometry(input.action_graph_zoomout).center(),
    )


def click_though_results(qtbot: QtBot, results: ResultsWidget):
    for i, io in enumerate(results.io):
        results.io.combo_name.setCurrentIndex(i)

    qtbot.mouseClick(
        results.graph_toolbar,
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
        results.graph_toolbar.actionGeometry(results.action_graph_histogram).center(),
    )
    qtbot.mouseClick(
        results.graph_toolbar,
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
        results.graph_toolbar.actionGeometry(
            results.action_graph_histogram_stacked
        ).center(),
    )
    qtbot.mouseClick(
        results.graph_toolbar,
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
        results.graph_toolbar.actionGeometry(
            results.action_graph_compositions
        ).center(),
    )
    qtbot.mouseClick(
        results.graph_toolbar,
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
        results.graph_toolbar.actionGeometry(results.action_graph_scatter).center(),
    )
    results.check_scatter_logx.click()
    results.check_scatter_logy.click()
    qtbot.mouseClick(
        results.graph_toolbar,
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
        results.graph_toolbar.actionGeometry(results.action_graph_zoomout).center(),
    )


def test_spcal_no_data(qtbot: QtBot):
    window = SPCalWindow()
    qtbot.add_widget(window)
    with qtbot.wait_exposed(window):
        window.show()

    # Default enabled
    assert window.tabs.isTabEnabled(window.tabs.indexOf(window.options))
    assert window.tabs.isTabEnabled(window.tabs.indexOf(window.sample))
    assert not window.tabs.isTabEnabled(window.tabs.indexOf(window.reference))
    assert not window.tabs.isTabEnabled(window.tabs.indexOf(window.results))

    click_though_options(qtbot, window.options)

    # Switch to sample, attempt switch to ref
    qtbot.mouseClick(
        window.tabs.tabBar(),
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
        window.tabs.tabBar().tabRect(window.tabs.indexOf(window.sample)).center(),
    )
    assert window.tabs.currentIndex() == window.tabs.indexOf(window.sample)

    qtbot.mouseClick(
        window.tabs.tabBar(),
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
        window.tabs.tabBar().tabRect(window.tabs.indexOf(window.reference)).center(),
    )
    assert window.tabs.currentIndex() == window.tabs.indexOf(window.sample)

    click_though_input(qtbot, window.sample)

    # Enable ref and switch
    window.options.efficiency_method.setCurrentIndex(1)  # Reference Particle
    assert window.tabs.isTabEnabled(window.tabs.indexOf(window.reference))

    qtbot.mouseClick(
        window.tabs.tabBar(),
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
        window.tabs.tabBar().tabRect(window.tabs.indexOf(window.reference)).center(),
    )
    assert window.tabs.currentIndex() == window.tabs.indexOf(window.reference)

    click_though_input(qtbot, window.reference)


def test_spcal_single_quad_data(qtbot: QtBot):
    window = SPCalWindow()
    qtbot.add_widget(window)
    with qtbot.wait_exposed(window):
        window.show()

    npz = np.load(Path(__file__).parent.parent.joinpath("data/agilent_au_data.npz"))
    data = np.array(npz["au50nm"], dtype=[("Au", float)])

    window.sample.loadData(data, {"path": "test/data.csv", "dwelltime": 1e-4})

    # Default enabled
    assert window.tabs.isTabEnabled(window.tabs.indexOf(window.options))
    assert window.tabs.isTabEnabled(window.tabs.indexOf(window.sample))
    assert not window.tabs.isTabEnabled(window.tabs.indexOf(window.reference))
    assert window.tabs.isTabEnabled(window.tabs.indexOf(window.results))

    click_though_options(qtbot, window.options)

    # Switch to sample, attempt switch to ref
    qtbot.mouseClick(
        window.tabs.tabBar(),
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
        window.tabs.tabBar().tabRect(window.tabs.indexOf(window.sample)).center(),
    )
    assert window.tabs.currentIndex() == window.tabs.indexOf(window.sample)

    qtbot.mouseClick(
        window.tabs.tabBar(),
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
        window.tabs.tabBar().tabRect(window.tabs.indexOf(window.reference)).center(),
    )
    assert window.tabs.currentIndex() == window.tabs.indexOf(window.sample)

    click_though_input(qtbot, window.sample)

    qtbot.mouseClick(
        window.tabs.tabBar(),
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
        window.tabs.tabBar().tabRect(window.tabs.indexOf(window.results)).center(),
    )
    assert window.tabs.currentIndex() == window.tabs.indexOf(window.results)

    click_though_results(qtbot, window.results)

    # Enable ref and switch
    window.options.efficiency_method.setCurrentIndex(1)  # Reference Particle
    assert window.tabs.isTabEnabled(window.tabs.indexOf(window.reference))
    assert not window.tabs.isTabEnabled(window.tabs.indexOf(window.results))

    qtbot.mouseClick(
        window.tabs.tabBar(),
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
        window.tabs.tabBar().tabRect(window.tabs.indexOf(window.reference)).center(),
    )
    assert window.tabs.currentIndex() == window.tabs.indexOf(window.reference)

    click_though_input(qtbot, window.reference)
