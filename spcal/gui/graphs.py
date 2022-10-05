from PySide2 import QtCore, QtGui, QtWidgets
# @Todo move to PySide6 as drawing is 20-50% faster
import numpy as np
import pyqtgraph

from typing import Optional


class ParticleView(pyqtgraph.GraphicsView):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent=parent, background=QtGui.QColor(255, 255, 255))

        self.layout = pyqtgraph.GraphicsLayout()
        self.setCentralWidget(self.layout)
        self.plots = {}

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

        plot.plot(y=y, x=np.arange(y.size) * 1e-3, pen=pen, skipFiniteCheck=True)
        plot.setLimits(xMin=x[0], xMax=x[-1], yMin=0, yMax=np.amax(y))
        plot.setMouseEnabled(y=False)
        plot.setAutoVisible(y=True)
        plot.enableAutoRange(y=True)
        plot.hideButtons()

        try:  # link view to the first plot
            plot.setXLink(next(iter(self.plots.values())))
        except StopIteration:
            pass

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
