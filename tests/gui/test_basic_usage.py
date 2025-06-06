from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtCore
from pytestqt.qtbot import QtBot

from spcal.gui.inputs import InputWidget, ReferenceWidget
from spcal.gui.main import SPCalWindow
from spcal.gui.options import OptionsWidget
from spcal.gui.results import ResultsWidget


def click_though_options(qtbot: QtBot, options: OptionsWidget):
    qtbot.keyClick(options.uptake.lineedit, QtCore.Qt.Key_1)
    assert options.uptake.value() == 1
    qtbot.keyClick(options.efficiency, QtCore.Qt.Key_1)
    assert options.efficiency.value() == 1.0

    assert options.dwelltime.lineedit.isReadOnly()

    assert not options.window_size.isEnabled()
    with qtbot.wait_signal(options.check_window.toggled, timeout=50):
        options.check_window.click()
    assert options.check_window.isChecked()
    assert options.window_size.isEnabled()
    assert options.window_size.text() == "1000"
    qtbot.keyClick(options.window_size, QtCore.Qt.Key_Backspace)
    assert options.window_size.value() == 100

    qtbot.keyClick(options.poisson.alpha, QtCore.Qt.Key_1)
    assert options.poisson.alpha.value() == 0.0011

    qtbot.keyClick(options.gaussian.alpha, QtCore.Qt.Key_Backspace)
    qtbot.keyClick(options.gaussian.alpha, QtCore.Qt.Key_5)
    qtbot.keyClick(options.gaussian.alpha, QtCore.Qt.Key_Enter)
    assert options.gaussian.alpha.value() == 2.867e-5
    assert options.gaussian.sigma.value() == 4.0235

    options.gaussian.sigma.setValue(4.2649)
    assert options.gaussian.alpha.value() == 1e-5

    with qtbot.wait_signal(options.check_iterative.toggled, timeout=50):
        options.check_iterative.click()
    assert options.check_iterative.isChecked()

    options.limit_method.setCurrentText("Manual Input")
    assert not options.compound_poisson.isEnabled()
    assert not options.gaussian.isEnabled()
    assert not options.poisson.isEnabled()
    assert not options.check_iterative.isEnabled()
    assert not options.check_window.isEnabled()
    options.limit_method.setCurrentText("Compound Poisson")
    assert options.check_iterative.isEnabled()
    assert options.check_window.isEnabled()
    options.compound_poisson.method.setCurrentIndex(0)
    assert not options.check_window.isEnabled()

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
        assert io.massfraction.value() is None
        qtbot.keyClick(io.massfraction, QtCore.Qt.Key_1)
        assert io.massfraction.value() == 1.0

        assert io.lod_count.isReadOnly()

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
        input.graph_toolbar.actionGeometry(input.action_graph_single).center(),
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
            results.action_graph_histogram_single
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
    results.scatter_fit_degree.setValue(2)


@pytest.mark.parametrize(
    "test_locales",
    [
        QtCore.QLocale.Language.English,
        QtCore.QLocale.Language.Spanish,
        QtCore.QLocale.Language.German,
    ],
    indirect=True,
)
def test_spcal_no_data(qtbot: QtBot, test_locales):
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
    assert not window.options.isComplete()

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

    window.options.limit_method.setCurrentText("Manual Input")
    for i, io in enumerate(window.sample.io):
        assert not io.lod_count.isReadOnly()
        with qtbot.wait_signal(window.sample.io.limitsChanged, timeout=100):
            qtbot.keyClick(io.lod_count, QtCore.Qt.Key_1)
            qtbot.keyClick(io.lod_count, QtCore.Qt.Key_Enter)
    for i, io in enumerate(window.reference.io):
        assert not io.lod_count.isReadOnly()
        with qtbot.wait_signal(window.reference.io.limitsChanged, timeout=100):
            qtbot.keyClick(io.lod_count, QtCore.Qt.Key_1)
            qtbot.keyClick(io.lod_count, QtCore.Qt.Key_Enter)

    window.resetInputs()


def test_spcal_single_quad_data(qtbot: QtBot, test_locales):
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
    assert window.options.isComplete()

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
    assert window.options.isComplete()
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

    window.options.efficiency_method.setCurrentIndex(2)  # Mass response
    assert window.options.isComplete()
    assert window.tabs.isTabEnabled(window.tabs.indexOf(window.reference))
    assert not window.tabs.isTabEnabled(window.tabs.indexOf(window.results))

    # Limits editable accross reload
    window.options.limit_method.setCurrentText("Manual Input")
    for i, io in enumerate(window.sample.io):
        assert not io.lod_count.isReadOnly()
        io.lod_count.setValue(None)

    window.sample.loadData(data, {"path": "test/data.csv", "dwelltime": 1e-4})

    # limits default to best
    for i, io in enumerate(window.sample.io):
        assert not io.lod_count.isReadOnly()
        assert io.lod_count.value() is not None

    window.resetInputs()
