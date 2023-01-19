from typing import List

import numpy as np
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.graphs.base import MultiPlotGraphicsView, SinglePlotGraphicsView
from spcal.gui.graphs.legends import MultipleItemSampleProxy
from spcal.gui.graphs.plots import HistogramPlotItem, ParticlePlotItem


class CompositionView(SinglePlotGraphicsView):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(
            "Detection Compositions",
            xlabel="Compositions",
            ylabel="No. Events",
            parent=parent,
        )

    def drawData(
        self,
        compositions: np.ndarray,
        counts: np.ndarray,
        pen: QtGui.QPen | None = None,
        brushes: List[QtGui.QBrush] | None = None,
    ) -> None:

        x = np.arange(counts.size)
        y0 = np.zeros(counts.size)

        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 1.0)
            pen.setCosmetic(True)
        if brushes is None:
            brushes = [QtGui.QBrush() for _ in range(len(compositions.dtype.names))]
        assert len(brushes) >= len(compositions.dtype.names)

        y0 = counts.astype(float)
        for name, brush in zip(compositions.dtype.names, brushes):
            height = compositions[name] * counts
            y0 -= height
            bars = pyqtgraph.BarGraphItem(
                x=x, height=height, width=0.75, y0=y0, pen=pen, brush=brush
            )
            self.plot.addItem(bars)
            self.plot.legend.addItem(MultipleItemSampleProxy(brush, items=[bars]), name)

        y_range = np.amax(counts)
        self.xaxis.setTicks([[(t, str(t + 1)) for t in x]])
        self.plot.setLimits(
            yMin=-y_range * 0.05, yMax=y_range * 1.1, xMin=-1, xMax=compositions.size
        )


class ScatterView(SinglePlotGraphicsView):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__("Scatter", xlabel="", ylabel="", parent=parent)

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


class HistogramView(MultiPlotGraphicsView):
    def getHistogramPlot(self, name: str, **add_kws) -> HistogramPlotItem:
        if name not in self.plots:
            return self.addHistogramPlot(name, **add_kws)
        return self.plots[name]

    def addHistogramPlot(
        self, name: str, xlabel: str = "", xunit: str = ""
    ) -> HistogramPlotItem:
        self.plots[name] = HistogramPlotItem(name=name, xlabel=xlabel, xunit=xunit)
        self.plots[name].setXLink(self.layout.getItem(0, 0))
        self.plots[name].setYLink(self.layout.getItem(0, 0))

        self.layout.addItem(self.plots[name])
        self.layout.nextRow()

        self.resizeEvent(QtGui.QResizeEvent(QtCore.QSize(0, 0), QtCore.QSize(0, 0)))

        return self.plots[name]


class ParticleView(MultiPlotGraphicsView):
    regionChanged = QtCore.Signal()

    def addParticlePlot(
        self, name: str, xscale: float = 1.0, downsample: int = 1
    ) -> ParticlePlotItem:
        self.plots[name] = ParticlePlotItem(name=name, xscale=xscale)
        self.plots[name].setDownsampling(ds=downsample, mode="peak", auto=True)
        self.plots[name].setXLink(self.layout.getItem(0, 0))

        self.plots[name].region.sigRegionChanged.connect(self.plotRegionChanged)
        self.plots[name].region.sigRegionChangeFinished.connect(self.regionChanged)

        self.layout.addItem(self.plots[name])

        self.layout.nextRow()
        self.resizeEvent(QtGui.QResizeEvent(QtCore.QSize(0, 0), QtCore.QSize(0, 0)))

        return self.plots[name]

    def plotRegionChanged(self, region: pyqtgraph.LinearRegionItem) -> None:
        self.blockSignals(True)
        for plot in self.plots.values():
            plot.region.blockSignals(True)
            plot.region.setRegion(region.getRegion())
            plot.region.blockSignals(False)
        self.blockSignals(False)
