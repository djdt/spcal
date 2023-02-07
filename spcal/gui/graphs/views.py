from typing import List

import numpy as np
import numpy.lib.recfunctions as rfn
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.graphs.base import (
    MultiPlotGraphicsView,
    SinglePlotGraphicsView,
)
from spcal.gui.graphs.items import PieChart
from spcal.gui.graphs.legends import MultipleItemSampleProxy
from spcal.gui.graphs.plots import ParticlePlotItem


class CompositionView(QtWidgets.QGraphicsView):
    def __init__(self, parent: QtWidgets.QWidget | None = None):

        self._scene = QtWidgets.QGraphicsScene(0, 0, 1, 1)
        super().__init__(
            self._scene,
            #         "Detection Compositions",
            #         xlabel="Compositions",
            #         ylabel="No. Events",
            parent=parent,
        )

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.fitInView(self.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def clear(self) -> None:
        if self.scene() is not None:
            self.scene().clear()

    def drawData(
        self,
        compositions: np.ndarray,
        counts: np.ndarray,
        pen: QtGui.QPen | None = None,
        brushes: List[QtGui.QBrush] | None = None,
    ) -> None:

        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 1.0)
            pen.setCosmetic(True)
        if brushes is None:
            brushes = [QtGui.QBrush() for _ in range(len(compositions.dtype.names))]
        assert len(brushes) >= len(compositions.dtype.names)

        size = 100.0
        max_r = np.amax(counts)
        spacing = size * 3.0

        self.setSceneRect(
            -spacing / 2.0, -spacing / 2.0, counts.size * spacing, spacing
        )
        self.fitInView(self.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

        for i, (count, comp) in enumerate(zip(counts, compositions)):
            pie = PieChart(
                count / max_r * size, rfn.structured_to_unstructured(comp), brushes
            )
            pie.setPos(i * spacing, 0)
            label = QtWidgets.QGraphicsSimpleTextItem(str(count))
            label.setPos(i * spacing, spacing / 3.0)
            label.moveBy(
                -label.boundingRect().width() / 2.0, label.boundingRect().height()
            )
            self.scene().addItem(pie)
            self.scene().addItem(label)


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

        region_pen.setStyle(QtCore.Qt.PenStyle.DashLine)

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


# class HistogramView(MultiPlotGraphicsView):
#     #     def getHistogramPlot(self, name: str, **add_kws) -> HistogramPlotItem:
#     #         if name not in self.plots:
#     #             return self.addHistogramPlot(name, **add_kws)
#     #         return self.plots[name]

#     def addPlot(
#         self,
#         name: str,
#         xlabel: str = "",
#         xunit: str = "",
#         **kwargs,
#     ) -> HistogramPlotItem:
#         plot = HistogramPlotItem(name=name, xlabel=xlabel, xunit=xunit)
#         super().addPlot(name, plot, xlink=True, **kwargs)
#         return plot

#     self.plots[name] = HistogramPlotItem(name=name, xlabel=xlabel, xunit=xunit)
#     self.plots[name].setXLink(self.layout.getItem(0, 0))
#     # self.plots[name].setYLink(self.layout.getItem(0, 0))

#     self.layout.addItem(self.plots[name])
#     self.layout.nextRow()
#     self.resizeEvent(QtGui.QResizeEvent(QtCore.QSize(0, 0), QtCore.QSize(0, 0)))

#     return self.plots[name]


class ParticleView(MultiPlotGraphicsView):
    regionChanged = QtCore.Signal()

    def addParticlePlot(
        self, name: str, xscale: float = 1.0, downsample: int = 1
    ) -> ParticlePlotItem:
        plot = ParticlePlotItem(name=name, xscale=xscale)
        plot.setDownsampling(ds=downsample, mode="peak", auto=True)

        plot.region.sigRegionChanged.connect(self.plotRegionChanged)
        plot.region.sigRegionChangeFinished.connect(self.regionChanged)

        super().addPlot(name, plot, xlink=True)
        return plot

    def plotRegionChanged(self, region: pyqtgraph.LinearRegionItem) -> None:
        self.blockSignals(True)
        for plot in self.plots.values():
            plot.region.blockSignals(True)
            plot.region.setRegion(region.getRegion())
            plot.region.blockSignals(False)
        self.blockSignals(False)
