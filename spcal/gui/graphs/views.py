from typing import Dict, List, Tuple

import numpy as np
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.graphs.legends import MultipleItemSampleProxy
from spcal.gui.graphs.plots import HistogramPlotItem, ParticlePlotItem
from spcal.gui.util import create_action


class _SPCalGraphicsView(pyqtgraph.GraphicsView):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent=parent, background="white")

        self.action_copy_image = create_action(
            "insert-image",
            "Copy Image to Clipboard",
            "Copy an image of the plot to the clipboard.",
            self.copyToClipboard,
        )

        self.action_show_legend = create_action(
            "view-hidden",
            "Hide Legend",
            "Toggle visibility of the legend.",
            self.toggleLegendVisible,
        )

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        menu = QtWidgets.QMenu(self)
        menu.addAction(self.action_copy_image)
        if all(legend is not None for legend in self.legends()):
            if any(legend.isVisible() for legend in self.legends()):
                self.action_show_legend.setIcon(QtGui.QIcon.fromTheme("view-hidden"))
                self.action_show_legend.setText("Hide Legend")
            else:
                self.action_show_legend.setIcon(QtGui.QIcon.fromTheme("view-visible"))
                self.action_show_legend.setText("Show Legend")

            menu.addAction(self.action_show_legend)
        menu.exec(event.globalPos())

    def copyToClipboard(self) -> None:
        """Copy current view to system clipboard."""
        pixmap = QtGui.QPixmap(self.viewport().size())
        painter = QtGui.QPainter(pixmap)
        self.render(painter)
        painter.end()
        QtWidgets.QApplication.clipboard().setPixmap(pixmap)  # type: ignore

    def legends(self) -> List[pyqtgraph.LegendItem]:
        return []

    def toggleLegendVisible(self) -> None:
        for legend in self.legends():
            legend.setVisible(not legend.isVisible())


class _SinglePlotGraphicsView(_SPCalGraphicsView):
    def __init__(
        self,
        title: str,
        xlabel: str,
        ylabel: str,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent=parent)

        pen = QtGui.QPen(QtCore.Qt.black, 1.0)
        pen.setCosmetic(True)

        self.xaxis = pyqtgraph.AxisItem("bottom", pen=pen, textPen=pen, tick_pen=pen)
        self.xaxis.setLabel(xlabel)

        self.yaxis = pyqtgraph.AxisItem("left", pen=pen, textPen=pen, tick_pen=pen)
        self.yaxis.setLabel(ylabel)

        self.plot = pyqtgraph.PlotItem(
            title=title,
            name="central_plot",
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

    def clear(self) -> None:
        self.plot.clear()
        self.plot.legend.clear()

    def legends(self) -> List[pyqtgraph.LegendItem]:
        return [self.plot.legend]

    def zoomReset(self) -> None:
        self.plot.setRange(
            xRange=self.plot.vb.state["limits"]["xLimits"],
            yRange=self.plot.vb.state["limits"]["yLimits"],
            disableAutoRange=False,
        )


class _ScrollableGraphicsView(_SPCalGraphicsView):
    def __init__(
        self, minimum_plot_height: int = 200, parent: QtWidgets.QWidget | None = None
    ):
        self.minimum_plot_height = minimum_plot_height
        self.layout = pyqtgraph.GraphicsLayout()
        self.plots: Dict[str, pyqtgraph.PlotItem] = {}

        super().__init__(parent=parent)

        self.setCentralWidget(self.layout)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)

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
        )  # we do this because some subclasses like to redefine
        # setRange in an incompatible way.
        self.updateMatrix()

    def clear(self) -> None:
        self.layout.clear()
        self.plots = {}

    def legends(self) -> List[pyqtgraph.LegendItem]:
        return [plot.legend for plot in self.plots.values()]

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


class CompositionView(_SinglePlotGraphicsView):
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


class ScatterView(_SinglePlotGraphicsView):
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


class HistogramView(_ScrollableGraphicsView):
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


class ParticleView(_ScrollableGraphicsView):
    regionChanged = QtCore.Signal()

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
    ):
        self.downsample = 1
        super().__init__(minimum_plot_height=150, parent=parent)

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
