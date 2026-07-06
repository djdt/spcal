from spcal.gui.graphs.viewbox import ViewBoxForceScaleAtZero
from typing import Callable
from pathlib import Path
from PySide6 import QtGui, QtWidgets
import numpy as np

from PySide6 import QtCore
from pytestqt.qtbot import QtBot


from spcal.datafile import SPCalNuDataFile, SPCalDataFile
from spcal.gui.graphs.base import SinglePlotGraphicsView, AxisRangeDialog
from spcal.gui.graphs.calibration import CalibrationView
from spcal.gui.graphs.items import BarChart, HoverableChartItem, PieChart
from spcal.gui.graphs.particle import ExclusionRegion, ParticleView
from spcal.gui.graphs.histogram import HistogramView
from spcal.gui.graphs.composition import CompositionView
from spcal.gui.graphs.scatter import ScatterView
from spcal.gui.graphs.spectra import SpectraView
from spcal.gui.graphs.legends import ParticleItemSample, HistogramItemSample
from spcal.gui.graphs.util import text_for_mz

from spcal.isotope import ISOTOPE_TABLE
from spcal.processing.method import SPCalProcessingMethod
from spcal.processing.options import SPCalIsotopeOptions


def png_size(path: Path) -> tuple[int, int]:
    """Returns the width and height of a PNG file"""
    with path.open("rb") as fp:
        header = fp.read(8)
        assert header == b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a"
        fp.read(8)
        w = fp.read(4)
        h = fp.read(4)
    return int.from_bytes(w), int.from_bytes(h)


def test_graph_viewbox():
    vb = ViewBoxForceScaleAtZero()

    vb.setRange(xRange=(10.0, 30.0), yRange=(0.0, 20.0))
    vb.scaleBy([1.0, 2.0], center=QtCore.QPointF(5.0, 10.0))

    assert vb.viewRange()[1] == [-0.8, 40.8]

    vb.translateBy(x=0, y=10)
    assert vb.viewRange()[1] == [-0.8, 40.8]
    vb.translateBy(QtCore.QPointF(0.0, 10.0))
    assert vb.viewRange()[1] == [-0.8, 40.8]


def test_graph_base(qtbot: QtBot):
    view = SinglePlotGraphicsView("test", xlabel="xxx", ylabel="yyy", xunits="s")
    qtbot.addWidget(view)

    with qtbot.waitExposed(view):
        view.show()

    assert "xxx (s)" in view.plot.xaxis.labelString()
    assert "yyy" in view.plot.yaxis.labelString()

    bounds = view.dataBounds()
    assert np.allclose(bounds, (0, 1, 0, 1))

    # test curve diff reduction                    |--------|--|--|--|--|-----|
    item = view.plot.drawCurve(np.arange(10), np.array([0, 1, 2, 3, 2, 3, 6, 0, 0, 0]))
    assert item.xData is not None
    assert item.xData.size == 7

    # test hist drawing
    item = view.plot.drawHistogram(
        np.array([5, 4, 2, 3, 1]), np.array([0.1, 0.3, 0.4, 0.6, 0.8, 1.0]), width=0.8
    )
    assert item.xData is not None and item.yData is not None
    assert item.xData.size == 12
    assert item.xData[0] == 0.1
    assert item.xData[1] == 0.12
    assert np.all(item.yData[::2] == 0)

    # test line
    item = view.plot.drawLine(5.0, QtCore.Qt.Orientation.Horizontal)
    assert item.xData is not None and item.yData is not None
    assert np.all(item.yData == 5.0)
    assert item.xData[0] == 0.0
    assert item.xData[1] == 9.0

    # test line, vertical
    item = view.plot.drawLine(5.0, QtCore.Qt.Orientation.Vertical)
    assert item.xData is not None and item.yData is not None
    assert np.all(item.xData == 5.0)
    assert item.yData[0] == 0.0
    assert item.yData[1] == 6.0

    # test scatter
    item = view.plot.drawScatter(np.arange(5), np.random.random(5))
    assert item.getData()[0].size == 5

    bounds = view.dataBounds()
    assert np.allclose(bounds, (0, 9, 0, 6))

    rect = view.dataRect()
    assert rect.width() == 9
    assert rect.height() == 6

    view.clear()


def test_graph_base_ranges(qtbot: QtBot):
    view = SinglePlotGraphicsView("test", xlabel="xxx", ylabel="yyy", xunits="s")
    qtbot.addWidget(view)

    with qtbot.waitExposed(view):
        view.show()

    view.plot.drawCurve(np.arange(11), np.arange(11) * 2.0)
    assert view.dataBounds() == (0.0, 10.0, 0.0, 20.0)

    # test axis range
    assert view.plot.getViewBox().viewRange() == [[0.0, 1.0], [0.0, 1.0]]
    view.setAxisRange("x", 0.4, 0.5)
    view.setAxisRange("y", 0.3, 0.6)
    assert np.allclose(
        view.plot.getViewBox().viewRange(), [[0.4, 0.5], [0.3, 0.6]], atol=0.1
    )

    assert view.plot.getViewBox().mouseEnabled()[0]
    assert not view.plot.getViewBox().state["autoVisibleOnly"][0]  # type: ignore
    assert not view.plot.getViewBox().autoRangeEnabled()[0]

    view.setAxisAutoScale("x", True)

    assert not view.plot.getViewBox().mouseEnabled()[0]
    assert view.plot.getViewBox().state["autoVisibleOnly"][0]  # type: ignore
    assert view.plot.getViewBox().autoRangeEnabled()[0]

    view.setAxisAutoScale("y", True)


def test_graph_base_events(qtbot: QtBot):
    view = SinglePlotGraphicsView("test", xlabel="xxx", ylabel="yyy", xunits="s")
    qtbot.addWidget(view)

    with qtbot.waitExposed(view):
        view.show()

    view.plot.drawCurve(np.arange(11), np.arange(11) * 2.0, name="test")
    view.plot.getViewBox().enableAutoRange(x=True, y=False)

    assert view.dataBounds() == (0.0, 10.0, 0.0, 20.0)

    menu = view.axisMenu("x")
    assert menu.actions()[0].isChecked()
    menu.actions()[0].trigger()
    assert not view.plot.getViewBox().autoRangeEnabled()[0]

    menu = view.axisMenu("y")
    assert not menu.actions()[0].isChecked()

    menu = view.customContextMenu(QtCore.QPoint(view.mapToGlobal(view.rect().center())))
    assert (
        len(menu.actions()) == 6
    )  # copy image, export image, legend, zoom reset, 2 seps

    view.data_for_export = {"a": np.zeros(10)}
    view.context_menu_actions.append(QtGui.QAction("test"))

    menu = view.customContextMenu(QtCore.QPoint(view.mapToGlobal(view.rect().center())))
    assert len(menu.actions()) == 9  # + action, data export, legend and a sep

    # callbacks
    assert view.plot.legend is not None and view.plot.legend.isVisible()
    view.setLegendVisible(False)
    assert view.plot.legend is not None and not view.plot.legend.isVisible()

    view.copyToClipboard()
    assert QtWidgets.QApplication.clipboard().pixmap() is not None


def test_axis_range_dialog(qtbot: QtBot):
    dlg = AxisRangeDialog((20.0, 40.0), (0.0, 100.0))

    qtbot.addWidget(dlg)
    with qtbot.waitExposed(dlg):
        dlg.show()

    assert dlg.spinbox_lo.value() == 20.0
    assert dlg.spinbox_hi.value() == 40.0
    for spinbox in [dlg.spinbox_lo, dlg.spinbox_hi]:
        assert spinbox.minimum() == 0.0
        assert spinbox.maximum() == 100.0

    dlg.spinbox_lo.setValue(10.0)
    dlg.spinbox_hi.setValue(15.0)

    with qtbot.waitSignal(
        dlg.rangeSelected,
        check_params_cb=lambda min, max: min == 10.0 and max == 15.0,
        timeout=100,
    ):
        dlg.accept()


def test_graph_base_export(qtbot: QtBot, tmp_path: Path):
    view = SinglePlotGraphicsView("test", xlabel="x", ylabel="y", xunits="s")
    qtbot.addWidget(view)

    view.data_for_export["test1"] = np.arange(10)
    view.data_for_export["test2"] = np.ones(5)

    view.exportData(tmp_path.joinpath("plot_export.csv"))
    view.exportData(tmp_path.joinpath("plot_export.npz"))

    csv = np.genfromtxt(tmp_path.joinpath("plot_export.csv"), names=True)
    assert np.allclose(csv["test1"], np.arange(10))
    assert np.allclose(csv["test2"][:5], 1.0)  # padded with nans

    npz = np.load(tmp_path.joinpath("plot_export.npz"))
    assert np.allclose(npz["test1"], np.arange(10))
    assert np.allclose(npz["test2"], 1.0)


def test_graph_base_font_overlap(qtbot: QtBot):
    view = SinglePlotGraphicsView("test", xlabel="xxx", ylabel="yyy", xunits="s")
    qtbot.addWidget(view)

    with qtbot.waitExposed(view):
        view.show()

    view.plot.drawCurve(np.arange(10), np.random.random(10), name="curve")

    assert view.plot.legend is not None

    font = view.font()

    for i in range(8, 32, 4):
        font.setPointSize(i)
        view.setFont(font)
        for item, label in view.plot.legend.items:
            assert not item.sceneBoundingRect().intersects(label.sceneBoundingRect())


def test_graph_calibration(qtbot: QtBot):
    view = CalibrationView()
    qtbot.addWidget(view)

    with qtbot.waitExposed(view):
        view.show()

    xs = np.arange(10)
    ys = np.arange(10) + np.random.random(10)

    view.plot.drawScatter(xs, ys)
    view.drawTrendline(xs, ys, weighting="1/x")


def test_graph_particle(
    qtbot: QtBot,
    default_method: SPCalProcessingMethod,
    random_datafile_generator: Callable[..., SPCalDataFile],
):
    view = ParticleView()
    qtbot.addWidget(view)

    with qtbot.waitExposed(view):
        view.show()

    df = random_datafile_generator(
        size=10000,
        number=100,
        isotopes=[ISOTOPE_TABLE[("Ag", 107)], ISOTOPE_TABLE[("Ag", 109)]],
    )
    results = default_method.processDataFile(df)

    view.drawResult(results[df.selected_isotopes[0]])

    assert view.plot.legend is not None
    assert len(view.plot.legend.items) == 1
    assert len(view.data_for_export) == 2

    # Test overlapping legend label
    item, label = view.plot.legend.items[0]
    assert isinstance(item, ParticleItemSample)
    assert not item.sceneBoundingRect().intersects(label.sceneBoundingRect())

    # Test overlapping
    view.drawResult(results[df.selected_isotopes[1]])
    assert len(view.plot.legend.items) == 2
    item, _ = view.plot.legend.items[0]
    item2, _ = view.plot.legend.items[1]
    assert not item.sceneBoundingRect().intersects(item2.sceneBoundingRect())

    # test legend mouse clicks
    pos = view.mapFromScene(item.mapToScene(item.rect().center()))

    assert item.item.isVisible()
    assert item.detections.isVisible()
    assert item2.item.isVisible()
    assert item2.detections.isVisible()

    qtbot.mouseClick(
        view.viewport(),
        QtCore.Qt.MouseButton.LeftButton,
        pos=pos,
    )
    assert not item.item.isVisible()
    assert not item.detections.isVisible()

    qtbot.mouseClick(
        view.viewport(),
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.ShiftModifier,
        pos=pos,
    )
    assert item.item.isVisible()
    assert item.detections.isVisible()

    assert not item2.item.isVisible()
    assert not item2.detections.isVisible()

    view.update()
    QtWidgets.QApplication.processEvents()

    pos = view.mapFromScene(item.mapToScene(item.rect().topLeft()))

    for line in item.lines:
        assert line.isVisible()
    for line in item2.lines:
        assert line.isVisible()

    qtbot.mouseClick(
        view.viewport(),
        QtCore.Qt.MouseButton.LeftButton,
        pos=pos,
    )
    for line in item.lines:
        assert not line.isVisible()

    qtbot.mouseClick(
        view.viewport(),
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.ShiftModifier,
        pos=pos,
    )

    for line in item.lines:
        assert line.isVisible()
    for line in item2.lines:
        assert not line.isVisible()

    view.update()  # force draw
    QtWidgets.QApplication.processEvents()

    view.addExclusionRegion(0.1, 0.3)
    assert view.exclusionRegions() == [(0.1, 0.3)]

    for item in view.plot.items:
        if isinstance(item, ExclusionRegion):
            item.requestRemoval.emit()

    assert view.exclusionRegions() == []

    view.clear()


def test_graph_histogram(
    qtbot: QtBot,
    default_method: SPCalProcessingMethod,
    random_datafile_generator: Callable[..., SPCalDataFile],
):
    view = HistogramView()
    qtbot.addWidget(view)

    with qtbot.waitExposed(view):
        view.show()

    df = random_datafile_generator(
        size=10000,
        number=100,
        isotopes=[ISOTOPE_TABLE["Fe", 56], ISOTOPE_TABLE["Fe", 57]],
    )
    results = default_method.processDataFile(df)

    # single
    view.drawResults(list(results.values()))
    view.repaint()

    assert view.plot.legend is not None
    assert len(view.plot.legend.items) == 2
    assert len(view.data_for_export) == 8

    # test legend
    item, label = view.plot.legend.items[0]
    assert isinstance(item, HistogramItemSample)
    assert not item.sceneBoundingRect().intersects(label.sceneBoundingRect())
    item2, _ = view.plot.legend.items[1]
    assert isinstance(item2, HistogramItemSample)
    assert not item2.sceneBoundingRect().intersects(item.sceneBoundingRect())

    pos = view.mapFromScene(item.mapToScene(item.rect().center()))

    assert item.item.isVisible()
    assert item2.item.isVisible()

    qtbot.mouseClick(
        view.viewport(),
        QtCore.Qt.MouseButton.LeftButton,
        pos=pos,
    )
    assert not item.item.isVisible()

    qtbot.mouseClick(
        view.viewport(),
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.ShiftModifier,
        pos=pos,
    )
    assert item.item.isVisible()
    assert not item2.item.isVisible()

    view.update()  # force draw
    QtWidgets.QApplication.processEvents()

    view.clear()

    # multi
    view.drawResults(list(results.values()), labels=["test1", "test2"])
    view.repaint()
    view.clear()

    # filtered
    view.draw_filtered = False
    view.drawResults(list(results.values()), labels=["test1", "test2"])
    view.repaint()

    assert len(view.data_for_export) == 4

    view.clear()

    assert len(view.data_for_export) == 0


def test_graph_composition(
    qtbot: QtBot,
    default_method: SPCalProcessingMethod,
    random_datafile_generator: Callable[..., SPCalDataFile],
):
    view = CompositionView()
    qtbot.addWidget(view)

    with qtbot.waitExposed(view):
        view.show()

    positions = np.array([10, 50, 100, 200, 500, 550, 600, 670, 800, 900])
    df = random_datafile_generator(
        size=1000,
        number=[positions, positions, positions],
        isotopes=[
            ISOTOPE_TABLE["Fe", 54],
            ISOTOPE_TABLE["Fe", 56],
            ISOTOPE_TABLE["Fe", 57],
        ],
    )
    results = list(default_method.processDataFile(df).values())

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
            assert item.radius == 1.0  # 10^2

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
            assert item.height == 1.0  # 10^2

    # Hover tested externally


def test_graph_scatter(
    qtbot: QtBot,
    default_method: SPCalProcessingMethod,
    random_datafile_generator: Callable[..., SPCalDataFile],
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

    df = random_datafile_generator(
        size=1000,
        number=100,
        isotopes=[
            ISOTOPE_TABLE["Au", 197],
            ISOTOPE_TABLE["Ag", 107],
            ISOTOPE_TABLE["Ag", 109],
        ],
    )
    results = list(default_method.processDataFile(df).values())

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


def test_graph_image_export_particle(
    qtbot: QtBot,
    tmp_path: Path,
    default_method: SPCalProcessingMethod,
    random_datafile_generator: Callable[..., SPCalDataFile],
):
    view = ParticleView()
    output = tmp_path.joinpath("particle_image.png")
    df = random_datafile_generator()
    results = default_method.processDataFile(df)

    for result in results.values():
        view.drawResult(result, label=str(result.isotope))

    qtbot.addWidget(view)
    with qtbot.waitExposed(view):
        view.show()

    view.exportImageWithOptions(
        output,
        dpi=300,
        size=QtCore.QSize(1800, 1200),
        font=QtGui.QFont("serif"),
    )

    assert output.exists()
    assert png_size(output) == (1800, 1200)

    # test setting defaults
    settings = QtCore.QSettings()
    settings.clear()

    options = view.getDefaultImageExportOptions()
    assert options[0] == view.viewport().size()
    assert options[1] == 96
    assert options[2] == view.font()
    assert options[3] == QtGui.QColor(QtCore.Qt.GlobalColor.black)
    assert options[4] == QtGui.QColor(QtCore.Qt.GlobalColor.white)

    view.setDefaultImageExportOptions(
        QtCore.QSize(1800, 1200),
        300,
        QtGui.QFont("serif"),
        QtGui.QColor(255, 0, 0),
        QtGui.QColor(0, 0, 255),
    )
    options = view.getDefaultImageExportOptions()
    assert options[0] == QtCore.QSize(1800, 1200)
    assert options[1] == 300
    assert options[2] == QtGui.QFont("serif")
    assert options[3] == QtGui.QColor(255, 0, 0)
    assert options[4] == QtGui.QColor(0, 0, 255)


def test_graph_image_export_histogram(
    qtbot: QtBot,
    tmp_path: Path,
    default_method: SPCalProcessingMethod,
    random_datafile_generator: Callable[..., SPCalDataFile],
):
    view = HistogramView()
    output = tmp_path.joinpath("histogram_image.png")
    df = random_datafile_generator()
    results = default_method.processDataFile(df)

    for result in results.values():
        view.drawResult(result, label=str(result.isotope))

    qtbot.addWidget(view)
    with qtbot.waitExposed(view):
        view.show()

    view.exportImageWithOptions(
        output,
        dpi=96,
        size=QtCore.QSize(600, 400),
        font=QtGui.QFont("sans"),
    )

    assert output.exists()
    assert png_size(output) == (600, 400)


def test_graph_image_export_composition(
    qtbot: QtBot,
    tmp_path: Path,
    default_method: SPCalProcessingMethod,
    random_datafile_generator: Callable[..., SPCalDataFile],
):
    view = CompositionView()
    output = tmp_path.joinpath("histogram_image.png")
    df = random_datafile_generator()

    results = default_method.processDataFile(df)
    default_method.filterResults(results)
    clusters = default_method.processClusters(results)

    view.drawResults(list(results.values()), clusters)

    qtbot.addWidget(view)
    with qtbot.waitExposed(view):
        view.show()

    view.exportImageWithOptions(
        output,
        dpi=96,
        size=QtCore.QSize(600, 400),
        font=QtGui.QFont("sans"),
    )

    assert output.exists()
    assert png_size(output) == (600, 400)


def test_graph_text_for_mz():
    assert text_for_mz(56.0) == "56.00(Fe)"
    assert text_for_mz(190.0) == "190.00(Os)"
    assert text_for_mz(116.0) == "116.00(Cd,Sn)"
