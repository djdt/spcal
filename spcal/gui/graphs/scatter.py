import numpy as np
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.graphs.base import SinglePlotGraphicsView


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
        weighting: str = "none",
        set_title: bool = True,
        pen: QtGui.QPen | None = None,
    ) -> None:
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.red, 1.0)
            pen.setCosmetic(True)

        if weighting == "none":
            w = np.ones_like(x)
        elif weighting == "1/x":
            w = 1.0 / x
        elif weighting == "1/x²":
            w = 1.0 / (x**2)
        elif weighting == "1/y":
            w = 1.0 / y
        elif weighting == "1/y²":
            w = 1.0 / (y**2)
        else:
            raise ValueError(f"drawFit: unknown weighting '{weighting}'")

        poly = np.polynomial.Polynomial.fit(x, y, degree, w=w / w.sum())
        poly = poly.convert()

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

        if set_title:
            self.plot.setTitle(f"{x.size} points; y = {poly}")
