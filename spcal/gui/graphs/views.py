from typing import Dict, List, Tuple

import numpy as np
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.graphs.legends import MultipleItemSampleProxy
from spcal.gui.graphs.plots import HistogramPlotItem, ParticlePlotItem
from spcal.gui.util import create_action


class CompositionView(pyqtgraph.GraphicsView):
    def __init__(
        self,
        pen: QtGui.QPen | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent=parent, background="white")
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 1.0)
            pen.setCosmetic(True)

        self.xaxis = pyqtgraph.AxisItem("bottom", pen=pen, textPen=pen, tick_pen=pen)
        self.xaxis.setLabel("Composition")

        self.yaxis = pyqtgraph.AxisItem("left", pen=pen, textPen=pen, tick_pen=pen)
        self.yaxis.setLabel("No. Events")

        self.plot = pyqtgraph.PlotItem(
            title="Detection Compositions",
            name="hist",
            axisItems={"bottom": self.xaxis, "left": self.yaxis},
            enableMenu=False,
            parent=parent,
        )
        self.plot.hideButtons()
        self.plot.setMouseEnabled(x=False)
        self.plot.addLegend(
            offset=(-5, 5), verSpacing=-5, colCount=1, labelTextColor="black"
        )
        self.setCentralWidget(self.plot)

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

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        action_copy_image = create_action(
            "insert-image",
            "Copy Image to Clipboard",
            "Copy an image of the plot to the clipboard.",
            self.copyToClipboard,
        )
        menu = QtWidgets.QMenu(self)
        menu.addAction(action_copy_image)
        menu.exec(event.globalPos())

    def copyToClipboard(self) -> None:
        """Copy current view to system clipboard."""
        pixmap = QtGui.QPixmap(self.viewport().size())
        painter = QtGui.QPainter(pixmap)
        self.render(painter)
        painter.end()
        QtWidgets.QApplication.clipboard().setPixmap(pixmap)  # type: ignore

    def clear(self) -> None:
        self.plot.clear()
        self.plot.legend.clear()

    def zoomReset(self) -> None:
        self.plot.setRange(
            xRange=self.plot.vb.state["limits"]["xLimits"],
            yRange=self.plot.vb.state["limits"]["yLimits"],
            disableAutoRange=False,
        )


class ParticleView(pyqtgraph.GraphicsView):
    regionChanged = QtCore.Signal()

    def __init__(
        self,
        downsample: int = 1,
        minimum_plot_height: int = 150,
        parent: QtWidgets.QWidget | None = None,
    ):
        self.downsample = downsample
        self.minimum_plot_height = minimum_plot_height
        self.layout = pyqtgraph.GraphicsLayout()

        super().__init__(parent=parent, background="white")
        self.setCentralWidget(self.layout)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.plots: Dict[str, ParticlePlotItem] = {}

    # Taken from pyqtgraph.widgets.MultiPlotWidget
    def setRange(self, *args, **kwds):
        pyqtgraph.GraphicsView.setRange(self, *args, **kwds)
        if self.centralWidget is not None:
            r = self.range
            minHeight = self.layout.currentRow * self.minimum_plot_height
            if r.height() < minHeight:
                r.setHeight(minHeight)
                r.setWidth(r.width() - self.verticalScrollBar().width())
            self.centralWidget.setGeometry(r)

    # Taken from pyqtgraph.widgets.MultiPlotWidget
    def resizeEvent(self, event: QtGui.QResizeEvent):
        if self.closed:
            return
        if self.autoPixelRange:
            self.range = QtCore.QRectF(0, 0, self.size().width(), self.size().height())
        ParticleView.setRange(
            self, self.range, padding=0, disableAutoPixel=False
        )  ## we do this because some subclasses like to redefine setRange in an incompatible way.
        self.updateMatrix()

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        action_copy_image = create_action(
            "insert-image",
            "Copy Image to Clipboard",
            "Copy an image of the plot to the clipboard.",
            self.copyToClipboard,
        )
        menu = QtWidgets.QMenu(self)
        menu.addAction(action_copy_image)
        menu.exec(event.globalPos())

    def copyToClipboard(self) -> None:
        """Copy current view to system clipboard."""
        pixmap = QtGui.QPixmap(self.viewport().size())
        painter = QtGui.QPainter(pixmap)
        self.render(painter)
        painter.end()
        QtWidgets.QApplication.clipboard().setPixmap(pixmap)  # type: ignore

    def addParticlePlot(self, name: str, xscale: float = 1.0) -> ParticlePlotItem:
        self.plots[name] = ParticlePlotItem(name=name, xscale=xscale)
        self.plots[name].setDownsampling(ds=self.downsample, mode="peak", auto=True)
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

    # def setLinkedYAxis(self, linked: bool = True) -> None:
    #     plots = list(self.plots.values())
    #     ymax = np.argmax([plot.vb.state["limits"]["yLimits"][1] for plot in plots])

    #     for plot in self.plots.values():
    #         if linked:
    #             plot.setYLink(plots[ymax])
    #             # plot.setLimits(yMin=0, yMax=ymax)
    #         else:
    #             plot.setYLink(None)
    #     for plot in self.plots.values():
    #         plot.vb.updateViewRange()  # rescale to max bounds

    def clear(self) -> None:
        self.layout.clear()
        self.plots = {}

    def zoomReset(self) -> None:
        for plot in self.plots.values():
            plot.autoRange()


class HistogramView(pyqtgraph.GraphicsView):
    def __init__(
        self,
        minimum_plot_height: int = 200,
        parent: QtWidgets.QWidget | None = None,
    ):
        self.minimum_plot_height = minimum_plot_height
        self.layout = pyqtgraph.GraphicsLayout()

        super().__init__(parent=parent, background="white")
        self.setCentralWidget(self.layout)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.plots: Dict[str, HistogramPlotItem] = {}

    # Taken from pyqtgraph.widgets.MultiPlotWidget
    def setRange(self, *args, **kwds):
        pyqtgraph.GraphicsView.setRange(self, *args, **kwds)
        if self.centralWidget is not None:
            r = self.range
            minHeight = self.layout.currentRow * self.minimum_plot_height
            if r.height() < minHeight:
                r.setHeight(minHeight)
                r.setWidth(r.width() - self.verticalScrollBar().width())
            self.centralWidget.setGeometry(r)

    # Taken from pyqtgraph.widgets.MultiPlotWidget
    def resizeEvent(self, event: QtGui.QResizeEvent):
        if self.closed:
            return
        if self.autoPixelRange:
            self.range = QtCore.QRectF(0, 0, self.size().width(), self.size().height())
        ParticleView.setRange(
            self, self.range, padding=0, disableAutoPixel=False
        )  ## we do this because some subclasses like to redefine setRange in an incompatible way.
        self.updateMatrix()

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        action_copy_image = create_action(
            "insert-image",
            "Copy Image to Clipboard",
            "Copy an image of the plot to the clipboard.",
            self.copyToClipboard,
        )
        menu = QtWidgets.QMenu(self)
        menu.addAction(action_copy_image)
        menu.exec(event.globalPos())

    def copyToClipboard(self) -> None:
        """Copy current view to system clipboard."""
        pixmap = QtGui.QPixmap(self.viewport().size())
        painter = QtGui.QPainter(pixmap)
        self.render(painter)
        painter.end()
        QtWidgets.QApplication.clipboard().setPixmap(pixmap)  # type: ignore

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

    def clear(self) -> None:
        self.layout.clear()
        self.plots = {}

    def bounds(self) -> Tuple[float, float, float, float]:
        bounds = np.array(
            [plot.vb.childrenBounds() for plot in self.plots.values()], dtype=float
        )
        if np.all(np.isnan(bounds)):
            return 0.0, 1.0, 0.0, 1.0
        return (
            np.nanmin(bounds[:, 0, 0]),
            np.nanmax(bounds[:, 0, 1]),
            np.nanmin(bounds[:, 1, 0]),
            np.nanmax(bounds[:, 1, 1]),
        )

    def zoomReset(self) -> None:
        if self.layout.getItem(0, 0) is None:
            return
        xmin, xmax, ymin, ymax = self.bounds()

        for plot in self.plots.values():
            plot.setLimits(xMin=xmin, xMax=xmax, yMin=ymin, yMax=ymax)

        self.layout.getItem(0, 0).setRange(
            xRange=(xmin, xmax),
            yRange=(ymin, ymax),
            disableAutoRange=True,
        )


class ScatterView(pyqtgraph.GraphicsView):
    def __init__(
        self,
        pen: QtGui.QPen | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent=parent, background="white")
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 1.0)
            pen.setCosmetic(True)

        self.xaxis = pyqtgraph.AxisItem("bottom", pen=pen, textPen=pen, tick_pen=pen)
        self.xaxis.setLabel("")

        self.yaxis = pyqtgraph.AxisItem("left", pen=pen, textPen=pen, tick_pen=pen)
        self.yaxis.setLabel("")

        self.plot = pyqtgraph.PlotItem(
            title="Scatter",
            name="scatter",
            axisItems={"bottom": self.xaxis, "left": self.yaxis},
            enableMenu=False,
            parent=parent,
        )
        self.plot.hideButtons()
        # self.plot.setMouseEnabled(x=False)
        self.plot.addLegend(
            offset=(-5, 5), verSpacing=-5, colCount=1, labelTextColor="black"
        )
        self.setCentralWidget(self.plot)

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

    # def drawLimits(self)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        action_copy_image = create_action(
            "insert-image",
            "Copy Image to Clipboard",
            "Copy an image of the plot to the clipboard.",
            self.copyToClipboard,
        )
        menu = QtWidgets.QMenu(self)
        menu.addAction(action_copy_image)
        menu.exec(event.globalPos())

    def copyToClipboard(self) -> None:
        """Copy current view to system clipboard."""
        pixmap = QtGui.QPixmap(self.viewport().size())
        painter = QtGui.QPainter(pixmap)
        self.render(painter)
        painter.end()
        QtWidgets.QApplication.clipboard().setPixmap(pixmap)  # type: ignore

    def clear(self) -> None:
        self.plot.clear()
        self.plot.legend.clear()

    def zoomReset(self) -> None:
        self.plot.setRange(
            xRange=self.plot.vb.state["limits"]["xLimits"],
            yRange=self.plot.vb.state["limits"]["yLimits"],
            disableAutoRange=False,
        )
