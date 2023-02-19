from typing import Dict, List, Tuple

import numpy as np
import numpy.lib.recfunctions as rfn
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.fit import fit_lognormal, fit_normal, lognormal_pdf, normal_pdf
from spcal.gui.graphs.base import PlotCurveItemFix, SinglePlotGraphicsView
from spcal.gui.graphs.items import PieChart
from spcal.gui.graphs.legends import (
    HistogramItemSample,
    MultipleItemSampleProxy,
    StaticRectItemSample,
)

# from spcal.gui.graphs.plots import ParticlePlotItem
from spcal.gui.graphs.viewbox import ViewBoxForceScaleAtZero


class CompositionView(SinglePlotGraphicsView):
    def __init__(self, parent: QtWidgets.QWidget | None = None):

        super().__init__("Detection Compositions", parent=parent)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.setAspectLocked(1.0)
        self.plot.legend.setSampleType(StaticRectItemSample)
        self.xaxis.hide()
        self.yaxis.hide()

        self.pies: List[PieChart] = []

    def drawData(
        self,
        compositions: np.ndarray,
        counts: np.ndarray,
        pen: QtGui.QPen | None = None,
        brushes: List[QtGui.QBrush] | None = None,
    ) -> None:
        self.pies.clear()

        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 1.0)
            pen.setCosmetic(True)
        if brushes is None:
            brushes = [QtGui.QBrush() for _ in range(len(compositions.dtype.names))]
        assert len(brushes) >= len(compositions.dtype.names)

        size = 100.0
        radii = np.sqrt(counts * np.pi)
        radii = radii / np.amax(radii) * size
        spacing = size * 2.0

        for i, (count, radius, comp) in enumerate(zip(counts, radii, compositions)):
            pie = PieChart(radius, rfn.structured_to_unstructured(comp), brushes)
            pie.setPos(i * spacing, 0)
            label = pyqtgraph.TextItem(f"{count}", color="black", anchor=(0.5, 0.0))
            label.setPos(i * spacing, -size)
            self.plot.addItem(pie)
            self.plot.addItem(label)
            self.pies.append(pie)

        # link all pie hovers
        for i in range(len(self.pies)):
            for j in range(len(self.pies)):
                if i == j:
                    continue
                self.pies[i].hovered.connect(self.pies[j].setHoveredIdx)

        for name, brush in zip(compositions.dtype.names, brushes):
            if np.sum(compositions[name]) > 0.0:
                self.plot.legend.addItem(StaticRectItemSample(brush), name)

    def zoomReset(self) -> None:  # No plotdata
        self.plot.autoRange()


class ResponseView(SinglePlotGraphicsView):
    def __init__(
        self,
        downsample: int = 64,
        parent: pyqtgraph.GraphicsWidget | None = None,
    ):
        super().__init__("Response TIC", "Time", "Intensity", parent=parent)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.enableAutoRange(x=True, y=True)
        self.plot.setDownsampling(ds=downsample, mode="subsample", auto=True)

        self.signal: pyqtgraph.PlotCurveItem | None = None
        self.signal_mean: pyqtgraph.PlotCurveItem | None = None

        region_pen = QtGui.QPen(QtCore.Qt.red, 1.0)
        region_pen.setCosmetic(True)

        self.region = pyqtgraph.LinearRegionItem(
            pen="grey",
            hoverPen="red",
            brush=QtGui.QBrush(QtCore.Qt.NoBrush),
            hoverBrush=QtGui.QBrush(QtCore.Qt.NoBrush),
            swapMode="block",
        )
        # self.region.movable = False  # prevent moving of region, but not lines
        self.region.lines[0].addMarker("|>", 0.9)
        self.region.lines[1].addMarker("<|", 0.9)
        self.region.sigRegionChangeFinished.connect(self.updateMean)

    @property
    def region_start(self) -> int:
        return int(self.region.lines[0].value())  # type: ignore

    @property
    def region_end(self) -> int:
        return int(self.region.lines[1].value())  # type: ignore

    def drawData(
        self,
        x: np.ndarray,
        y: np.ndarray,
        pen: QtGui.QPen | None = None,
    ) -> None:
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 1.0)
            pen.setCosmetic(True)

        # optimise by removing points with 0 change in gradient
        diffs = np.diff(y, n=2, append=0, prepend=0) != 0
        self.signal = pyqtgraph.PlotCurveItem(
            x=x[diffs], y=y[diffs], pen=pen, connect="all", skipFiniteCheck=True
        )
        self.plot.addItem(self.signal)

        self.region.blockSignals(True)
        self.region.setRegion((x[0], x[-1]))
        self.region.setBounds((x[0], x[-1]))
        self.region.blockSignals(False)
        self.plot.addItem(self.region)

    def drawMean(self, mean: float, pen: QtGui.QPen | None = None) -> None:
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.red, 2.0, QtCore.Qt.PenStyle.DashLine)
            pen.setCosmetic(True)

        if self.signal_mean is None:
            self.signal_mean = pyqtgraph.PlotCurveItem(
                pen=pen,
                connect="all",
                skipFiniteCheck=True,
            )
            self.plot.addItem(self.signal_mean)

        self.signal_mean.updateData(
            x=[self.region_start, self.region_end], y=[mean, mean]
        )

    def updateMean(self) -> None:
        if self.signal is None or self.signal_mean is None:
            return
        mean = np.mean(self.signal.yData[self.region_start : self.region_end])
        self.drawMean(mean)


class ScatterView(SinglePlotGraphicsView):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__("Scatter", parent=parent)

    def drawData(
        self,
        x: np.ndarray,
        y: np.ndarray,
        logx: bool = False,
        logy: bool = False,
        pen: QtGui.QPen | None = None,
        brush: QtGui.QBrush | None = None,
    ) -> None:
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 1.0)
            pen.setCosmetic(True)
        if brush is None:
            brush = QtGui.QBrush(QtCore.Qt.black)

        if logx:
            x = np.log10(x)
        if logy:
            y = np.log10(y)

        self.plot.setLogMode(logx, logy)

        curve = pyqtgraph.ScatterPlotItem(x=x, y=y, pen=pen, brush=brush)
        self.plot.addItem(curve)

        xmin, xmax = np.amin(x), np.amax(x)
        ymin, ymax = np.amin(y), np.amax(y)

        self.plot.setLimits(
            xMin=xmin - (xmax - xmin) * 0.05,
            xMax=xmax + (xmax - xmin) * 0.05,
            yMin=ymin - (ymax - ymin) * 0.05,
            yMax=ymax + (ymax - ymin) * 0.05,
        )
        self.plot.enableAutoRange(x=True, y=True)  # rescale to max bounds

    def drawFit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        degree: int = 1,
        logx: bool = False,
        logy: bool = False,
        pen: QtGui.QPen | None = None,
    ) -> None:
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.red, 1.0)
            pen.setCosmetic(True)
        poly = np.polynomial.Polynomial.fit(x, y, degree)

        xmin, xmax = np.amin(x), np.amax(x)
        sx = np.linspace(xmin, xmax, 1000)

        sy = poly(sx)

        if logx:
            sx = np.log10(sx)
        if logy:
            sy = np.log10(sy)

        curve = pyqtgraph.PlotCurveItem(
            x=sx, y=sy, pen=pen, connect="all", skipFiniteCheck=True
        )
        self.plot.addItem(curve)


class ParticleView(SinglePlotGraphicsView):
    regionChanged = QtCore.Signal()

    def __init__(
        self,
        xscale: float = 1.0,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(
            "Signal",
            xlabel="Time",
            xunits="s",
            ylabel="Intensity (counts)",
            parent=parent,
        )
        self.xaxis.setScale(xscale)

        self.plot.setMouseEnabled(y=False)
        self.plot.setAutoVisible(y=True)
        self.plot.enableAutoRange(y=True)
        self.plot.setLimits(yMin=0.0)

        self.legend_items: Dict[str, MultipleItemSampleProxy] = {}
        self.limit_items: List[pyqtgraph.PlotCurveItem] = []

        region_pen = QtGui.QPen(QtCore.Qt.red, 1.0)
        region_pen.setCosmetic(True)

        self.region = pyqtgraph.LinearRegionItem(
            pen="grey",
            hoverPen="red",
            brush=QtGui.QBrush(QtCore.Qt.NoBrush),
            hoverBrush=QtGui.QBrush(QtCore.Qt.NoBrush),
            swapMode="block",
        )
        self.region.sigRegionChangeFinished.connect(self.regionChanged)
        self.region.movable = False  # prevent moving of region, but not lines
        self.region.lines[0].addMarker("|>", 0.9)
        self.region.lines[1].addMarker("<|", 0.9)
        self.plot.addItem(self.region)

    @property
    def region_start(self) -> int:
        return int(self.region.lines[0].value())  # type: ignore

    @property
    def region_end(self) -> int:
        return int(self.region.lines[1].value())  # type: ignore

    def clear(self) -> None:
        self.legend_items.clear()
        super().clear()

    def clearScatters(self) -> None:
        for item in self.plot.listDataItems():
            if isinstance(item, pyqtgraph.ScatterPlotItem):
                self.plot.removeItem(item)

    def clearLimits(self) -> None:
        for limit in self.limit_items:
            self.plot.removeItem(limit)
        self.limit_items.clear()

    def drawSignal(
        self,
        name: str,
        x: np.ndarray,
        y: np.ndarray,
        pen: QtGui.QPen | None = None,
    ) -> None:
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 1.0)
            pen.setCosmetic(True)

        # optimise by removing points with 0 change in gradient
        diffs = np.diff(y, n=2, append=0, prepend=0) != 0
        curve = PlotCurveItemFix(
            x=x[diffs], y=y[diffs], pen=pen, connect="all", skipFiniteCheck=True
        )

        self.legend_items[name] = MultipleItemSampleProxy(pen.color(), items=[curve])

        self.plot.addItem(curve)
        self.plot.legend.addItem(self.legend_items[name], name)

        self.plot.addItem(self.region)

    def drawMaxima(
        self,
        name: str,
        x: np.ndarray,
        y: np.ndarray,
        brush: QtGui.QBrush | None = None,
        symbol: str = "t",
    ) -> None:
        if brush is None:
            brush = QtGui.QBrush(QtCore.Qt.red)

        scatter = pyqtgraph.ScatterPlotItem(
            x=x, y=y, size=6, symbol=symbol, pen=None, brush=brush
        )
        self.plot.addItem(scatter)

        self.legend_items[name].addItem(scatter)

    def drawLimits(
        self,
        x: np.ndarray,
        # mean: float | np.ndarray,
        limit: float | np.ndarray,
        pen: QtGui.QPen | None = None,
    ) -> None:
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 1.0, QtCore.Qt.DashLine)
            pen.setCosmetic(True)

        if isinstance(limit, float) or limit.size == 1:
            nx, y = [x[0], x[-1]], [limit, limit]
        else:
            diffs = np.diff(limit, n=2, append=0, prepend=0) != 0
            nx, y = x[diffs], limit[diffs]

        curve = pyqtgraph.PlotCurveItem(
            x=nx,
            y=y,
            name="Detection Threshold",
            pen=pen,
            connect="all",
            skipFiniteCheck=True,
        )
        self.limit_items.append(curve)
        self.plot.addItem(curve)


class PCAView(SinglePlotGraphicsView):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(xlabel="PC 1", ylabel="PC 2", parent=parent)

    def drawPCA(self, x: np.ndarray, y: np.ndarray) -> None:
        self.plot.clear()

        scatter = pyqtgraph.ScatterPlotItem(
            x=x,
            y=y,
        )
