from PySide6 import QtCore, QtGui, QtWidgets
import numpy as np
import pyqtgraph

from typing import Optional


class ParticleView(pyqtgraph.GraphicsView):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent=parent, background=QtGui.QColor(255, 255, 255))

        self.layout = pyqtgraph.GraphicsLayout()
        self.setCentralWidget(self.layout)
        self.plots = {}

        self.setSizeAdjustPolicy(QtWidgets.QScrollArea.AdjustIgnored)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn)

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

    def addParticlePlot(self, name: str, x: np.ndarray, y: np.ndarray) -> None:
        plot = self.layout.addPlot(
            title=name,
            axisItems={
                "bottom": self.createParticleAxis("bottom"),
                "left": self.createParticleAxis("left"),
            },
            enableMenu=False,
        )
        pen = QtGui.QPen(QtCore.Qt.black, 1.0)
        pen.setCosmetic(True)

        plot.plot(
            x=x,
            y=y,
            pen=pen,
            connect="all",
        )
        plot.setDownsampling(auto=True)
        plot.setLimits(xMin=x[0], xMax=x[-1], yMin=0, yMax=np.amax(y))
        plot.setMouseEnabled(y=False)
        plot.setAutoVisible(y=True)
        plot.enableAutoRange(y=True)
        plot.hideButtons()

        try:  # link view to the first plot
            plot.setXLink(next(iter(self.plots.values())))
        except StopIteration:
            pass

        plot.setMinimumHeight(200)

        self.plots[name] = plot
        self.layout.nextRow()

    def addParticleMaxima(self, name: str, x: np.ndarray, y: np.ndarray) -> None:
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

    def addParticleLimits(self, name: str, limits: np.ndarray) -> None:
        colors = {"mean": QtCore.Qt.red, "lc": QtCore.Qt.green, "ld": QtCore.Qt.blue}
        plot = self.plots[name]
        for name in limits.dtype.names:
            pen = QtGui.QPen(colors[name], 1.0, QtCore.Qt.DashLine)
            pen.setCosmetic(True)
            if limits[name].size == 1:
                plot.addLine(y=limits[name][0], name=name, pen=pen)
            else:
                plot.addPlot()

    def clear(self) -> None:
        self.layout.clear()
        self.plots = {}
