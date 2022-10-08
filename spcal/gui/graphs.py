from PySide6 import QtCore, QtGui, QtWidgets
import numpy as np
import pyqtgraph

from typing import Optional, Tuple


# class ResultsView(pyqtgraph.GraphicsView):
#     def __init__(
#         self, minimum_plot_height: int = 150, parent: Optional[QtWidgets.QWidget] = None
#     ):
#         self.minimum_plot_height = minimum_plot_height
#         self.layout = pyqtgraph.GraphicsLayout()
#         super().__init__(parent=parent, background=QtCore.Qt.white)
#         self.setCentralWidget(self.layout)
#         self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)

#         self.black_pen = QtGui.QPen(QtCore.Qt.black, 1.0)
#         self.black_pen.setCosmetic(True)

#     def createHistogramAxis(self, orientation: str) -> pyqtgraph.AxisItem:
#         axis = pyqtgraph.AxisItem(
#             orientation,
#             pen=self.black_pen,
#             textPen=self.black_pen,
#             tick_pen=self.black_pen,
#         )
#         if orientation == "bottom":
#             axis.setLabel("Time", units="s")
#         elif orientation == "left":
#             axis.setLabel("Intensity", units="Count")
#         else:
#             raise ValueError("createParticleAxis: use 'bottom' or 'left'")

#         axis.enableAutoSIPrefix(True)
#         return axis

# a, b = np.histogram(x, bins="auto")
# p = pyqtgraph.plot(b, a, stepMode="center", fillLevel=0, fillOutline=True, brush=(0, 0, 255, 128))
# p.setBackground("white")
# a, b = np.histogram(y, bins="auto")
# p.plot(b, a, stepMode="center", fillLevel=0, fillOutline=True, brush=(255, 0, 0, 128))
# pyqtgraph.exec()


class ParticleView(pyqtgraph.GraphicsView):
    analyticalLimitsChanged = QtCore.Signal(str)

    def __init__(
        self,
        downsample: int = 1,
        minimum_plot_height: int = 150,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        self.downsample = downsample
        self.minimum_plot_height = minimum_plot_height
        self.layout = pyqtgraph.GraphicsLayout()

        super().__init__(parent=parent, background=QtCore.Qt.white)
        self.setCentralWidget(self.layout)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.black_pen = QtGui.QPen(QtCore.Qt.black, 1.0)
        self.black_pen.setCosmetic(True)

        self.plots = {}
        self.plot_colors = [
            QtGui.QColor(105, 41, 196),
            QtGui.QColor(17, 146, 232),
            QtGui.QColor(0, 93, 93),
            QtGui.QColor(159, 24, 83),
            QtGui.QColor(250, 77, 86),
            QtGui.QColor(87, 4, 8),
            QtGui.QColor(25, 128, 56),
            QtGui.QColor(0, 45, 156),
            QtGui.QColor(238, 83, 139),
            QtGui.QColor(178, 134, 0),
            QtGui.QColor(0, 157, 154),
            QtGui.QColor(1, 39, 73),
            QtGui.QColor(138, 56, 0),
            QtGui.QColor(165, 110, 255),
        ]
        self.limit_colors = {
            "mean": QtCore.Qt.red,
            "lc": QtCore.Qt.green,
            "ld": QtCore.Qt.blue,
        }

    # Taken from pyqtgraph.widgets.MultiPlotWidget
    def setRange(self, *args, **kwds):
        pyqtgraph.GraphicsView.setRange(self, *args, **kwds)
        if self.centralWidget is not None:
            r = self.range
            minHeight = len(self.layout.rows) * self.minimum_plot_height
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

    def createParticleAxis(self, orientation: str, scale: float = 1.0) -> pyqtgraph.AxisItem:
        axis = pyqtgraph.AxisItem(
            orientation,
            pen=self.black_pen,
            textPen=self.black_pen,
            tick_pen=self.black_pen,
        )
        if orientation == "bottom":
            axis.setLabel("Time", units="s")
        elif orientation == "left":
            axis.setLabel("Intensity", units="Count")
        else:
            raise ValueError("createParticleAxis: use 'bottom' or 'left'")

        axis.enableAutoSIPrefix(False)
        axis.setScale(scale)
        return axis

    def addParticlePlot(self, name: str, xscale: float = 1.0) -> None:
        axis_pen = QtGui.QPen(QtCore.Qt.black, 1.0)
        axis_pen.setCosmetic(True)
        self.plots[name] = self.layout.addPlot(
            title=name,
            axisItems={
                "bottom": self.createParticleAxis("bottom", scale=xscale),
                "left": self.createParticleAxis("left"),
            },
            enableMenu=False,
            name=name,
        )
        self.plots[name].setMouseEnabled(y=False)
        self.plots[name].setAutoVisible(y=True)
        self.plots[name].enableAutoRange(y=True)
        self.plots[name].hideButtons()
        self.plots[name].addLegend(
            offset=(-5, 5), verSpacing=-5, colCount=3, labelTextColor="black"
        )
        self.plots[name].setDownsampling(ds=self.downsample, mode="peak", auto=True)
        self.plots[name].setXLink(self.layout.getItem(0, 0))
        self.layout.nextRow()
        self.resizeEvent(QtGui.QResizeEvent(QtCore.QSize(0, 0), QtCore.QSize(0, 0)))

    # def setLinkedYAxis(self, linked: bool = True) -> None:
    #     plots = list(self.plots.values())
    #     ymax = np.argmax([plot.vb.state["limits"]["yLimits"][1] for plot in plots])
    #     print(ymax)

    #     for plot in self.plots.values():
    #         if linked:
    #             plot.setYLink(plots[ymax])
    #             # plot.setLimits(yMin=0, yMax=ymax)
    #         else:
    #             plot.setYLink(None)
    #     for plot in self.plots.values():
    #         plot.vb.updateViewRange()  # rescale to max bounds

    def drawParticleSignal(
        self,
        name: str,
        x: np.ndarray,
        y: np.ndarray,
        pen: Optional[QtGui.QPen] = None,
        label: Optional[str] = None,
    ) -> None:
        plot = self.plots[name]

        if pen is None:
            pen = self.black_pen

        # optimise by removing points with 0 change in gradient
        diffs = np.diff(y, n=2, append=0, prepend=0) != 0
        item = pyqtgraph.PlotCurveItem(
            x=x[diffs],
            y=y[diffs],
            pen=pen,
            connect="all",
            skipFiniteCheck=True,
        )
        if label is not None:
            item.opts["name"] = label
        plot.addItem(item)
        plot.setLimits(xMin=x[0], xMax=x[-1])
        plot.enableAutoRange()  # rescale to max bounds

    def drawParticleMaxima(
        self,
        name: str,
        x: np.ndarray,
        y: np.ndarray,
        brush: Optional[QtGui.QBrush] = None,
    ) -> None:
        plot = self.plots[name]

        if brush is None:
            brush = QtGui.QBrush(QtCore.Qt.red)

        scatter = pyqtgraph.ScatterPlotItem(
            parent=plot,
            x=x,
            y=y,
            size=6,
            symbol="t",
            pen=None,
            brush=brush,
        )

        plot.addItem(scatter)

    def drawParticleLimits(self, name: str, x: np.ndarray, limits: np.ndarray) -> None:
        plot = self.plots[name]
        skip_lc = np.all(limits["lc"] == limits["ld"])
        for name in limits.dtype.names:
            if name == "lc" and skip_lc:
                continue

            pen = QtGui.QPen(self.limit_colors[name], 1.0, QtCore.Qt.DashLine)
            pen.setCosmetic(True)
            if limits[name].size == 1:
                x, y = [x[0], x[-1]], [limits[name][0], limits[name][0]]
            else:
                diffs = np.diff(limits[name], n=2, append=0, prepend=0) != 0
                x, y = x[diffs], y[diffs]
            item = pyqtgraph.PlotCurveItem(
                x=x,
                y=y,
                name=name,
                pen=pen,
                connect="all",
                skipFiniteCheck=True,
            )
            plot.addItem(item)

    def drawAnalyticalLimits(self, name: str, x0: float, x1: float) -> None:
        plot = self.plots[name]
        for x in [x0, x1]:
            line = pyqtgraph.InfiniteLine(
                x, angle=90, movable=False, label="limit", name="limit"
            )
            # line.sigPositionChangeFinished.connect(
            #     lambda: self.analyticalLimitsChanged.emit(name)
            # )
            plot.addItem(line)

    def analyticalLimits(self, name: str) -> Tuple[float, float]:
        limits = [
            item
            for item in self.plots[name].items
            if isinstance(item, pyqtgraph.InfiniteLine) and item._name == "limit"
        ]
        limits = sorted(item.value() for item in limits)
        return (limits[0], limits[1])

    def clear(self) -> None:
        self.layout.clear()
        self.plots = {}
