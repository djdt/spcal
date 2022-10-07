from PySide6 import QtCore, QtGui, QtWidgets
import numpy as np
import pyqtgraph

from typing import Optional


class ParticleView(pyqtgraph.GraphicsView):
    def __init__(
        self, minimum_plot_height: int = 150, parent: Optional[QtWidgets.QWidget] = None
    ):
        self.minimum_plot_height = minimum_plot_height
        self.layout = pyqtgraph.GraphicsLayout()

        super().__init__(parent=parent, background=QtGui.QColor(255, 255, 255))
        self.setCentralWidget(self.layout)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.plots = {}
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

    def createParticleAxis(self, orientation: str):
        axis_pen = QtGui.QPen(QtCore.Qt.black, 1.0)
        axis_pen.setCosmetic(True)
        axis = pyqtgraph.AxisItem(
            orientation,
            pen=axis_pen,
            textPen=axis_pen,
            tick_pen=axis_pen,
        )
        if orientation == "bottom":
            axis.setLabel("Time", units="s")
        elif orientation == "left":
            axis.setLabel("Intensity", units="Count")
        else:
            raise ValueError("createParticleAxis: use 'bottom' or 'left'")

        axis.enableAutoSIPrefix(False)
        return axis

    def addParticlePlot(self, name: str) -> None:
        axis_pen = QtGui.QPen(QtCore.Qt.black, 1.0)
        axis_pen.setCosmetic(True)
        self.plots[name] = self.layout.addPlot(
            title=name,
            axisItems={
                "bottom": self.createParticleAxis("bottom"),
                "left": self.createParticleAxis("left"),
            },
            enableMenu=False,
        )
        self.plots[name].setMouseEnabled(y=False)
        self.plots[name].setAutoVisible(y=True)
        self.plots[name].enableAutoRange(y=True)
        self.plots[name].hideButtons()
        self.plots[name].addLegend(offset=(5, 0))
        try:  # link view to the first plot
            self.plots[name].setXLink(next(iter(self.plots.values())))
        except StopIteration:
            pass
        self.layout.nextRow()
        self.resizeEvent(QtGui.QResizeEvent(QtCore.QSize(0, 0), QtCore.QSize(0, 0)))

    def drawParticleSignal(self, name: str, x: np.ndarray, y: np.ndarray) -> None:
        plot = self.plots[name]

        # optimise by removing points with 0 change in gradient
        diffs = np.diff(y, n=2, append=0, prepend=0) != 0

        pen = QtGui.QPen(QtCore.Qt.black, 1.0)
        pen.setCosmetic(True)

        item = pyqtgraph.PlotDataItem(x=x[diffs], y=y[diffs], pen=pen, connect="all")
        item.setDownsampling(auto=True)

        plot.addItem(item)
        plot.setLimits(xMin=x[0], xMax=x[-1], yMin=0, yMax=np.amax(y))

    def drawParticleMaxima(self, name: str, x: np.ndarray, y: np.ndarray) -> None:
        plot = self.plots[name]

        scatter = pyqtgraph.ScatterPlotItem(
            parent=plot,
            x=x,
            y=y,
            size=5,
            symbol="t1",
            pen=None,
            brush=QtGui.QBrush(QtCore.Qt.red),
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
                plot.addLine(y=limits[name][0], label=name, name=name, pen=pen)
            else:
                diffs = np.diff(limits[name], n=2, append=0, prepend=0) != 0
                item = pyqtgraph.PlotDataItem(
                    x=x[diffs], y=limits[name][diffs], name=name, pen=pen, connect="all"
                )
                item.setDownsampling(auto=True)
                plot.addItem(item)

    def clear(self) -> None:
        self.layout.clear()
        self.plots = {}
