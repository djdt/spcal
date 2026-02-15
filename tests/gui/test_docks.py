from PySide6 import QtGui
from pytestqt.qtbot import QtBot

from spcal.gui.docks.datafile import SPCalDataFilesDock
from spcal.gui.docks.central import SPCalCentralWidget
from spcal.gui.docks.instrumentoptions import (
    SPCalInstrumentOptionsDock,
    SPCalInstrumentOptionsWidget,
)
from spcal.gui.docks.isotopeoptions import SPCalIsotopeOptionsDock, IsotopeOptionTable
from spcal.gui.docks.limitoptions import SPCalLimitOptionsDock, SPCalLimitOptionsWidget
from spcal.gui.docks.outputs import SPCalOutputsDock, ResultOutputView
from spcal.gui.docks.toolbar import SPCalOptionsToolBar, SPCalViewToolBar
from spcal.isotope import ISOTOPE_TABLE
from spcal.processing.method import SPCalProcessingMethod
from spcal.processing.options import SPCalInstrumentOptions


def test_spcal_datfiles_dock(qtbot: QtBot, random_datafile_gen):
    dock = SPCalDataFilesDock()
    qtbot.addWidget(dock)

    with qtbot.waitExposed(dock):
        dock.show()

    assert dock.currentDataFile() is None
    assert len(dock.selectedDataFiles()) == 0

    dfs = [random_datafile_gen() for _ in range(5)]

    for df in dfs:
        with qtbot.waitSignal(dock.dataFileAdded):
            dock.addDataFile(df)

    assert len(dock.dataFiles()) == 5
    assert dock.currentDataFile() == dfs[-1]
    assert len(dock.selectedDataFiles()) == 1

    with qtbot.waitSignal(dock.dataFilesChanged):
        dock.list.selectAll()

    assert dock.currentDataFile() == dfs[-1]
    assert len(dock.selectedDataFiles()) == 5

    with qtbot.waitSignal(dock.dataFileRemoved):
        dock.model.removeRow(4)

    assert dock.currentDataFile() == dfs[-2]
    assert len(dock.dataFiles()) == 4

    dock.reset()

    assert len(dock.dataFiles()) == 0
    assert dock.currentDataFile() is None
    assert len(dock.selectedDataFiles()) == 0


def test_spcal_central_widget(
    qtbot: QtBot, default_method: SPCalProcessingMethod, random_datafile_gen
):
    widget = SPCalCentralWidget()
    qtbot.addWidget(widget)

    with qtbot.waitExposed(widget):
        widget.show()

    df = random_datafile_gen(
        size=1000,
        lam=10.0,
        isotopes=[
            ISOTOPE_TABLE[("Ag", 107)],
            ISOTOPE_TABLE[("Ag", 109)],
            ISOTOPE_TABLE[("Au", 197)],
        ],
        seed=131290,
    )

    default_method.limit_options.limit_method = "poisson"
    default_method.limit_options.poisson_kws["alpha"] = 1e-3

    results = default_method.processDataFile(df)
    default_method.filterResults(results)
    clusters = default_method.processClusters(results)

    widget.drawResultsParticle(
        list(results.values()),
        [QtGui.QColor(255, 0, 0), QtGui.QColor(0, 255, 0), QtGui.QColor(0, 0, 255)],
        ["a", "b", "c"],
        "signal",
    )
    widget.drawResultsHistogram(
        list(results.values()),
        [QtGui.QColor(255, 0, 0), QtGui.QColor(0, 255, 0), QtGui.QColor(0, 0, 255)],
        ["a", "b", "c"],
        "signal",
    )
    widget.drawResultsSpectra(df, next(iter(results.values())))
    widget.drawResultsComposition(
        list(results.values()),
        [QtGui.QColor(255, 0, 0), QtGui.QColor(0, 255, 0), QtGui.QColor(0, 0, 255)],
        "signal",
        clusters,
    )
    widget.drawResultsScatterExpr(
        results, "107Ag + 109Ag", "197Au / 2.0", "signal", "signal"
    )

    views = ["particle", "histogram", "composition", "spectra", "scatter"]
    for view in views:
        with qtbot.waitSignal(widget.requestRedraw):
            widget.setView(view)
        assert widget.currentView() == view

    widget.clear()

    for view in views:
        with qtbot.waitSignal(widget.requestRedraw):
            widget.setView(view)
        assert widget.currentView() == view


def test_spcal_instrument_options_dock_and_widget(qtbot: QtBot):
    dock = SPCalInstrumentOptionsDock(
        SPCalInstrumentOptions(None, 0.1), "efficiency", 0.04
    )
    qtbot.addWidget(dock)
    with qtbot.waitExposed(dock):
        dock.show()

    assert dock.options_widget.uptake.baseValue() is None
    assert dock.options_widget.efficiency.value() == 0.1
    assert dock.options_widget.calibration_mode.currentText() == "Efficiency"
    assert dock.options_widget.cluster_distance == 0.04
    assert not dock.options_widget.action_efficiency.isEnabled()

    assert dock.calibrationMode() == "efficiency"
    assert dock.clusterDistance() == 0.04

    with qtbot.waitSignal(dock.optionsChanged, timeout=100):
        dock.setCalibrationMode("mass response")
    assert dock.options_widget.calibration_mode.currentText() == "Mass Response"

    with qtbot.waitSignal(dock.optionsChanged):
        dock.options_widget.uptake.setBaseValue(0.3)

    assert dock.options_widget.action_efficiency.isEnabled()
    with qtbot.waitSignal(dock.efficiencyDialogRequested):
        dock.options_widget.button_efficiency.click()

    assert dock.instrumentOptions().uptake == 0.3
