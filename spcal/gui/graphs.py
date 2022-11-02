from PySide6 import QtCore, QtGui, QtWidgets
import numpy as np
import pyqtgraph

from typing import Dict, List, Tuple

from spcal.gui.util import create_action

color_schemes = {
    "IBM Carbon": [
        QtGui.QColor("#6929c4"),  # Purple 70
        QtGui.QColor("#1192e8"),  # Cyan 50
        QtGui.QColor("#005d5d"),  # Teal 70
        QtGui.QColor("#9f1853"),  # Magenta 70
        QtGui.QColor("#fa4d56"),  # Red 50
        QtGui.QColor("#570408"),  # Red 90
        QtGui.QColor("#198038"),  # Green 60
        QtGui.QColor("#002d9c"),  # Blue 80
        QtGui.QColor("#ee538b"),  # Magenta 50
        QtGui.QColor("#b26800"),  # Yellow 50
        QtGui.QColor("#009d9a"),  # Teal 50
        QtGui.QColor("#012749"),  # Cyan 90
        QtGui.QColor("#8a3800"),  # Orange 70
        QtGui.QColor("#a56eff"),  # Purple 50
    ],
    "Base16": [  # ordered for clarity
        QtGui.QColor("#ab4642"),  # 08 red
        QtGui.QColor("#7cafc2"),  # 0d blue
        QtGui.QColor("#dc9656"),  # 09 orange
        QtGui.QColor("#a1b56c"),  # 0b green
        QtGui.QColor("#f7ca88"),  # 0a yellow
        QtGui.QColor("#ba8baf"),  # 0e magenta
        QtGui.QColor("#86c1b9"),  # 0c teal
        QtGui.QColor("#a16946"),  # 0f brown
        QtGui.QColor("#b8b8b8"),  # 03 grey
    ],
    "ColorBrewer Set1": [
        QtGui.QColor("#e41a1c"),
        QtGui.QColor("#377eb8"),
        QtGui.QColor("#4daf4a"),
        QtGui.QColor("#984ea3"),
        QtGui.QColor("#ff7f00"),
        QtGui.QColor("#ffff33"),
        QtGui.QColor("#a65628"),
        QtGui.QColor("#f781bf"),
    ],
    "Tableau 10": [
        QtGui.QColor("#1f77b4"),
        QtGui.QColor("#ff7f0e"),
        QtGui.QColor("#2ca02c"),
        QtGui.QColor("#d62728"),
        QtGui.QColor("#9467bd"),
        QtGui.QColor("#8c564b"),
        QtGui.QColor("#e377c2"),
        QtGui.QColor("#7f7f7f"),
        QtGui.QColor("#bcbd22"),
        QtGui.QColor("#17becf"),
    ],
    "Okabe Ito": [  # https://jfly.uni-koeln.de/color/
        QtGui.QColor(0, 0, 0),
        QtGui.QColor(230, 159, 0),
        QtGui.QColor(86, 180, 233),
        QtGui.QColor(0, 158, 115),
        QtGui.QColor(240, 228, 66),
        QtGui.QColor(0, 114, 178),
        QtGui.QColor(213, 94, 0),
        QtGui.QColor(204, 121, 167),
    ],
}

symbols = ["t", "o", "s", "d", "+", "star", "t1", "x"]


class ViewBoxForceScaleAtZero(pyqtgraph.ViewBox):
    def scaleBy(
        self,
        s: List[float] | None = None,
        center: QtCore.QPointF | None = None,
        x: float | None = None,
        y: float | None = None,
    ) -> None:
        if center is not None:
            center.setY(0.0)
        super().scaleBy(s, center, x, y)

    def translateBy(
        self,
        t: QtCore.QPointF | None = None,
        x: float | None = None,
        y: float | None = None,
    ) -> None:
        if t is not None:
            t.setY(0.0)
        if y is not None:
            y = 0.0
        super().translateBy(t, x, y)


class ResultsFractionView(pyqtgraph.GraphicsView):
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
            self.plot.legend.addItem(pyqtgraph.BarGraphItem(brush=brush), name)

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


class HistogramPlotItem(pyqtgraph.PlotItem):
    def __init__(
        self,
        name: str,
        xlabel: str,
        xunit: str,
        pen: QtGui.QPen | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        self.name = name
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 1.0)
            pen.setCosmetic(True)

        self.xaxis = pyqtgraph.AxisItem("bottom", pen=pen, textPen=pen, tick_pen=pen)
        self.xaxis.setLabel(label=xlabel, units=xunit)
        self.yaxis = pyqtgraph.AxisItem("left", pen=pen, textPen=pen, tick_pen=pen)
        self.yaxis.setLabel("No. Events")
        self.yaxis.enableAutoSIPrefix(False)

        super().__init__(
            title="Results Histogram",
            name="hist",
            axisItems={"bottom": self.xaxis, "left": self.yaxis},
            viewBox=ViewBoxForceScaleAtZero(enableMenu=False),
            parent=parent,
        )
        self.hideButtons()
        self.addLegend(
            offset=(-5, 5), verSpacing=-5, colCount=1, labelTextColor="black"
        )

    def drawData(
        self,
        name: str,
        x: np.ndarray,
        bins: str | np.ndarray = "auto",
        pen: QtGui.QPen | None = None,
        brush: QtGui.QBrush | None = None,
    ) -> None:
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 1.0)
            pen.setCosmetic(True)
        if brush is None:
            brush = QtGui.QBrush(QtCore.Qt.black)

        hist, edges = np.histogram(x, bins)
        curve = pyqtgraph.PlotCurveItem(
            x=np.concatenate([edges, [edges[0]]]),  # draw bottom
            y=np.concatenate([hist, [0.0]]),
            stepMode="center",
            fillLevel=0,
            fillOutline=True,
            pen=pen,
            brush=brush,
            skipFiniteCheck=True,
        )
        self.legend.addItem(pyqtgraph.BarGraphItem(brush=brush), name)
        self.addItem(curve)


class ResultsHistogramView(pyqtgraph.GraphicsView):
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

    def addHistogramPlot(self, name: str, xlabel: str, xunit: str) -> HistogramPlotItem:
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


class ParticlePlotItem(pyqtgraph.PlotItem):
    limit_colors = {"mean": QtCore.Qt.red, "lc": QtCore.Qt.green, "ld": QtCore.Qt.blue}

    def __init__(
        self,
        name: str,
        xscale: float = 1.0,
        pen: QtGui.QPen | None = None,
        parent: pyqtgraph.GraphicsWidget | None = None,
    ):
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 1.0)
            pen.setCosmetic(True)

        self.xaxis = pyqtgraph.AxisItem("bottom", pen=pen, textPen=pen, tick_pen=pen)
        self.xaxis.setLabel(text="Time", units="s")
        self.xaxis.setScale(xscale)
        self.xaxis.enableAutoSIPrefix(False)

        self.yaxis = pyqtgraph.AxisItem("left", pen=pen, textPen=pen, tick_pen=pen)
        self.yaxis.setLabel(text="Intensity", units="counts")

        super().__init__(
            title=name,
            name=name,
            axisItems={"bottom": self.xaxis, "left": self.yaxis},
            enableMenu=False,
            parent=parent,
        )
        self.setMouseEnabled(y=False)
        self.setAutoVisible(y=True)
        self.enableAutoRange(y=True)
        self.hideButtons()
        self.addLegend(
            offset=(-5, 5), verSpacing=-5, colCount=3, labelTextColor="black"
        )

        self.signals: List[pyqtgraph.PlotCurveItem] = []
        self.scatters: List[pyqtgraph.ScatterPlotItem] = []
        self.limits: List[pyqtgraph.PlotCurveItem] = []

        region_pen = QtGui.QPen(QtCore.Qt.red, 1.0)
        region_pen.setCosmetic(True)

        self.region = pyqtgraph.LinearRegionItem(
            pen="grey",
            hoverPen="red",
            brush=QtGui.QBrush(QtCore.Qt.NoBrush),
            hoverBrush=QtGui.QBrush(QtCore.Qt.NoBrush),
            swapMode="block",
        )
        self.region.movable = False  # prevent moving of region, but not lines
        self.region.lines[0].addMarker("|>", 0.9)
        self.region.lines[1].addMarker("<|", 0.9)
        self.addItem(self.region)

    @property
    def region_start(self) -> int:
        return int(self.region.lines[0].value())  # type: ignore

    @property
    def region_end(self) -> int:
        return int(self.region.lines[1].value())  # type: ignore

    def clearSignal(self) -> None:
        for signal in self.signals:
            self.removeItem(signal)
        self.signals.clear()

    def clearScatters(self) -> None:
        for scatter in self.scatters:
            self.removeItem(scatter)
        self.scatters.clear()

    def clearLimits(self) -> None:
        for limit in self.limits:
            self.removeItem(limit)
        self.limits.clear()

    def drawSignal(
        self,
        x: np.ndarray,
        y: np.ndarray,
        pen: QtGui.QPen | None = None,
        label: str | None = None,
    ) -> None:
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 1.0)
            pen.setCosmetic(True)

        # optimise by removing points with 0 change in gradient
        diffs = np.diff(y, n=2, append=0, prepend=0) != 0
        curve = pyqtgraph.PlotCurveItem(
            x=x[diffs], y=y[diffs], pen=pen, connect="all", skipFiniteCheck=True
        )
        if label is not None:
            curve.opts["name"] = label

        self.signals.append(curve)
        self.addItem(curve)

        self.setLimits(
            xMin=x[0],
            xMax=x[-1],
            yMin=0,
            minXRange=(x[-1] - x[0]) * 1e-5,
            minYRange=np.ptp(y[diffs]) * 0.01,
        )
        self.enableAutoRange(y=True)  # rescale to max bounds

        self.region.setRegion((x[0], x[-1]))

    def drawMaxima(
        self,
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
        self.scatters.append(scatter)
        self.addItem(scatter)

    def drawLimits(self, x: np.ndarray, limits: np.ndarray) -> None:
        skip_lc = np.all(limits["lc"] == limits["ld"])
        for name in ["mean", "lc", "ld"]:
            if name == "lc" and skip_lc:
                continue

            pen = QtGui.QPen(self.limit_colors[name], 1.0, QtCore.Qt.DashLine)
            pen.setCosmetic(True)

            if limits[name].size == 1:
                x, y = [x[0], x[-1]], [limits[name][0], limits[name][0]]
            else:
                diffs = np.diff(limits[name], n=2, append=0, prepend=0) != 0
                x, y = x[diffs], limits[name][diffs]

            curve = pyqtgraph.PlotCurveItem(
                x=x, y=y, name=name, pen=pen, connect="all", skipFiniteCheck=True
            )
            self.limits.append(curve)
            self.addItem(curve)


class ParticleView(pyqtgraph.GraphicsView):
    regionChanged = QtCore.Signal(str)

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

        self.plots[name].region.sigRegionChangeFinished.connect(
            lambda: self.regionChanged.emit(name)
        )

        self.layout.addItem(self.plots[name])

        self.layout.nextRow()
        self.resizeEvent(QtGui.QResizeEvent(QtCore.QSize(0, 0), QtCore.QSize(0, 0)))

        return self.plots[name]

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


if __name__ == "__main__":  # test colors
    app = QtWidgets.QApplication()
    scene = QtWidgets.QGraphicsScene(
        -50,
        -50,
        200 + 100 * max(len(v) for v in color_schemes.values()),
        100 + 100 * len(color_schemes),
    )
    view = QtWidgets.QGraphicsView(scene)

    y = 0
    for name, colors in color_schemes.items():
        label = QtWidgets.QGraphicsTextItem(name)
        label.setPos(0, y)
        view.scene().addItem(label)
        x = 0
        for color in colors:
            x += 100
            rect = QtWidgets.QGraphicsRectItem(x, y, 50, 50)
            rect.setBrush(QtGui.QBrush(color))
            view.scene().addItem(rect)
        y += 100

    view.show()
    app.exec()
