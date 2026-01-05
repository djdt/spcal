import numpy as np
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.cluster import prepare_results_for_clustering
from spcal.gui.graphs.base import SinglePlotGraphicsView

from spcal.processing.result import SPCalProcessingResult


class ScatterView(SinglePlotGraphicsView):
    def __init__(
        self, font: QtGui.QFont | None = None, parent: QtWidgets.QWidget | None = None
    ):
        super().__init__("Scatter", font=font, parent=parent)

    def drawResults(
        self,
        result_x: SPCalProcessingResult,
        result_y: SPCalProcessingResult,
        key: str,
        logx: bool = False,
        logy: bool = False,
        pen: QtGui.QPen | None = None,
        brush: QtGui.QBrush | None = None,
    ):
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 1.0)
            pen.setCosmetic(True)
        if brush is None:
            brush = QtGui.QBrush(QtCore.Qt.GlobalColor.black)

        if result_x.peak_indicies is None or result_y.peak_indicies is None:
            raise ValueError("peak_indicies have not been generated")

        npeaks = result_x.number_peak_indicies
        x, y = np.zeros(npeaks, dtype=np.float32), np.zeros(npeaks, dtype=np.float32)
        np.add.at(
            x,
            result_x.peak_indicies[result_x.filter_indicies],
            result_x.calibrated(key),
        )
        np.add.at(
            y,
            result_y.peak_indicies[result_y.filter_indicies],
            result_y.calibrated(key),
        )
        valid = np.zeros(npeaks, dtype=bool)
        valid[result_x.peak_indicies[result_x.filter_indicies]] = True
        valid[result_y.peak_indicies[result_y.filter_indicies]] = True

        self.drawScatter(x[valid], y[valid], pen=pen, brush=brush)

        self.plot.xaxis.setLabel(str(result_x.isotope))
        self.plot.yaxis.setLabel(str(result_y.isotope))

        self.setDataLimits(-0.05, 1.05, -0.05, 1.05)

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
    ):
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.red, 1.0)
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
