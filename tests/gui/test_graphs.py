from pathlib import Path
from PySide6 import QtGui
import numpy as np

# from PySide6 import QtWidgets, QtGui
from pytestqt.qtbot import QtBot


from spcal.datafile import SPCalNuDataFile
from spcal.gui.graphs.base import SinglePlotGraphicsView
from spcal.gui.graphs.items import BarChart, HoverableChartItem, PieChart
from spcal.gui.graphs.particle import ExclusionRegion, ParticleView
from spcal.gui.graphs.histogram import HistogramView
from spcal.gui.graphs.composition import CompositionView
from spcal.gui.graphs.scatter import ScatterView
from spcal.gui.graphs.spectra import SpectraView
from spcal.gui.graphs.legends import ParticleItemSample, HistogramItemSample
from spcal.isotope import ISOTOPE_TABLE
from spcal.processing.method import SPCalProcessingMethod
from spcal.processing.options import SPCalIsotopeOptions


def test_graph_base(qtbot: QtBot):
    view = SinglePlotGraphicsView("test")
    qtbot.addWidget(view)

    with qtbot.waitExposed(view):
        view.show()


def test_graph_particle(qtbot: QtBot, default_method, random_result_generator):
    view = ParticleView()
    qtbot.addWidget(view)

    with qtbot.waitExposed(view):
        view.show()

    result = random_result_generator(default_method, size=10000, number=100)
    view.drawResult(result)

    assert view.plot.legend is not None
    assert len(view.plot.legend.items) == 1
    assert len(view.data_for_export) == 4

    # Test overlapping legend label
    item, label = view.plot.legend.items[0]
    assert isinstance(item, ParticleItemSample)
    assert not item.sceneBoundingRect().intersects(label.sceneBoundingRect())

    result = random_result_generator(default_method, size=1000, number=10)
    result.signals[:50] = np.nan

    # Test overlapping
    view.drawResult(result)
    assert len(view.plot.legend.items) == 2
    item, _ = view.plot.legend.items[0]
    item2, _ = view.plot.legend.items[1]
    assert not item.sceneBoundingRect().intersects(item2.sceneBoundingRect())

    view.addExclusionRegion(0.1, 0.3)
    assert view.exclusionRegions() == [(0.1, 0.3)]

    for item in view.plot.items:
        if isinstance(item, ExclusionRegion):
            item.requestRemoval.emit()

    assert view.exclusionRegions() == []

    view.clear()


def test_graph_histogram(qtbot: QtBot, default_method, random_result_generator):
    view = HistogramView()
    qtbot.addWidget(view)

    with qtbot.waitExposed(view):
        view.show()

    # single
    results = [random_result_generator(default_method, size=1000, number=100)]
    # results[0].filter_indicies = np.arange(50)
    view.drawResults(results)
    view.repaint()

    assert view.plot.legend is not None
    assert len(view.plot.legend.items) == 1
    assert len(view.data_for_export) == 4
    item, label = view.plot.legend.items[0]
    assert isinstance(item, HistogramItemSample)
    assert not item.sceneBoundingRect().intersects(label.sceneBoundingRect())

    view.clear()

    # multi
    results.append(random_result_generator(default_method, size=1000, number=80))
    view.drawResults(results, labels=["test1", "test2"])
    view.repaint()
    view.clear()

    # filtered
    view.draw_filtered = False
    view.drawResults(results, labels=["test1", "test2"])
    view.repaint()

    assert len(view.data_for_export) == 4

    view.clear()

    assert len(view.data_for_export) == 0


def test_graph_composition(qtbot: QtBot, default_method, random_result_generator):
    view = CompositionView()
    qtbot.addWidget(view)

    with qtbot.waitExposed(view):
        view.show()

    results = [
        random_result_generator(default_method, size=1000, number=100),
        random_result_generator(default_method, size=1000, number=100),
        random_result_generator(default_method, size=1000, number=100),
    ]
    results[0].peak_indicies = np.arange(100)
    results[1].peak_indicies = np.repeat(np.arange(50), 2)
    results[2].peak_indicies = np.repeat(np.arange(25), 4)
    for result in results:
        result.number_peak_indicies = 100

    view.drawResults(
        results,
        np.repeat(np.arange(10), 10) + 1,
        brushes=[
            QtGui.QBrush(QtGui.QColor(255, 0, 0)),
            QtGui.QBrush(QtGui.QColor(0, 255, 0)),
            QtGui.QBrush(QtGui.QColor(0, 0, 255)),
        ],
    )
    assert len(view.data_for_export) == 7
    for item in view.plot.items:
        if isinstance(item, HoverableChartItem):
            assert isinstance(item, PieChart)
            assert item.radius == 100.0  # 10^2

    view.clear()
    view.mode = "bar"

    view.drawResults(
        results,
        np.repeat(np.arange(10), 10) + 1,
        brushes=[
            QtGui.QBrush(QtGui.QColor(255, 0, 0)),
            QtGui.QBrush(QtGui.QColor(0, 255, 0)),
            QtGui.QBrush(QtGui.QColor(0, 0, 255)),
        ],
    )
    view.repaint()
    assert len(view.data_for_export) == 7
    for item in view.plot.items:
        if isinstance(item, HoverableChartItem):
            assert isinstance(item, BarChart)
            assert item.height == 100.0  # 10^2

    # Hover tested externally


def test_graph_scatter(
    qtbot: QtBot, default_method: SPCalProcessingMethod, random_result_generator
):
    view = ScatterView()
    qtbot.addWidget(view)

    default_method.instrument_options.uptake = 1.0
    default_method.instrument_options.efficiency = 0.1
    default_method.isotope_options[ISOTOPE_TABLE[("Au", 197)]] = SPCalIsotopeOptions(
        1.0, 1.0, 1.0
    )

    with qtbot.waitExposed(view):
        view.show()

    results = [
        random_result_generator(
            default_method, size=1000, number=100, isotope=ISOTOPE_TABLE[("Au", 197)]
        ),
        random_result_generator(
            default_method, size=1000, number=100, isotope=ISOTOPE_TABLE[("Ag", 107)]
        ),
        random_result_generator(
            default_method, size=1000, number=100, isotope=ISOTOPE_TABLE[("Ag", 109)]
        ),
    ]
    results[0].peak_indicies = np.arange(100)
    results[1].peak_indicies = np.repeat(np.arange(50), 2)
    results[2].peak_indicies = np.repeat(np.arange(25), 4)
    for result in results:
        result.number_peak_indicies = 100

    view.drawResultsExpr(results, "107Ag + 109Ag", "197Au", "signal", "signal")
    assert len(view.data_for_export) == 2
    view.clear()

    view.drawResultsExpr(results, "107Ag + 109Ag", "197Au", "mass", "signal")
    assert len(view.data_for_export) == 0
    view.clear()

    view.drawResultsExpr(results, "107Ag + 109Ag", "197Au", "signal", "mass")
    assert len(view.data_for_export) == 2
    view.clear()


def test_graph_spectra(qtbot: QtBot):
    view = SpectraView()
    qtbot.addWidget(view)

    with qtbot.waitExposed(view):
        view.show()

    signals = np.random.poisson(lam=100, size=(20, 100))

    df = SPCalNuDataFile(
        Path(),
        signals,
        np.linspace(0, 1, 100),
        np.arange(20, 120, 1.0),
        {},
        None,
        None,
        (0, None),
    )
    regions = np.array([[0, 5], [9, 10], [27, 29], [35, 37], [50, 70]])
    view.drawDataFile(df, regions)

    assert len(view.data_for_export) == 2

    view.clear()
    view.subtract_background = False
    view.drawDataFile(df, regions)

    assert len(view.data_for_export) == 2
    view.clear()

    assert len(view.data_for_export) == 0
