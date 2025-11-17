import numpy as np
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.graphs.base import SinglePlotGraphicsView
from spcal.gui.graphs.viewbox import ViewBoxForceScaleAtZero


def zero_truncated_compound_poisson_lognormal(
    size: int, lam: float, mu: float, sigma: float
) -> np.ndarray:
    """Draw values from a zero-truncated compound-Poisson-lognormal.

    This is the same as ``compound_poisson_lognormal`` except all Poisson values
    must be 1 or greater.

    Args:
        size: number of values
        lam: the expected value of the Poisson
        mu: the location of the lognormal
        sigma: the shape of the lognormal

    Returns:
        values of size ``size``
    """
    x = np.zeros(size, dtype=np.float32)
    u = np.random.uniform(np.exp(-lam), 1.0, size=size)
    zp = 1 + np.random.poisson(lam - (-np.log(u)))

    for i in range(1, zp.max()):
        x[zp >= i] += np.random.lognormal(mu, sigma, size=np.count_nonzero(zp >= i))
    return x


class SingleIonScatterView(SinglePlotGraphicsView):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(
            "Extracted Parameters",
            xlabel="m/z",
            ylabel="Shape (σ)",
            parent=parent,
        )

        self.points: pyqtgraph.ScatterPlotItem | None = None
        self.lines: dict[str, pyqtgraph.PlotCurveItem] = {}

        self.plot.setLimits(xMin=0.0, yMin=0.0)

    def clear(self):
        super().clear()
        self.points = None
        self.lines.clear()

    def drawData(
        self,
        x: np.ndarray,
        y: np.ndarray,
        pen: QtGui.QPen | None = None,
        brush: QtGui.QBrush | None = None,
    ):
        if self.points is not None:
            self.plot.removeItem(self.points)

        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 1.0)
            pen.setCosmetic(True)
        if brush is None:
            brush = QtGui.QBrush(QtCore.Qt.GlobalColor.black)

        self.points = pyqtgraph.ScatterPlotItem(x=x, y=y, pen=pen, brush=brush)
        self.plot.addItem(self.points)

        self.setDataLimits(-0.05, 1.05, -0.05, 1.05)

    def setValid(self, valid: np.ndarray):
        if self.points is None:
            return
        brush_valid = QtGui.QBrush(QtCore.Qt.GlobalColor.black)
        brush_invalid = QtGui.QBrush(QtCore.Qt.GlobalColor.red)
        brushes = [brush_valid if x else brush_invalid for x in valid]
        self.points.setBrush(brushes)

    def drawMaxDifference(
        self,
        poly: np.polynomial.Polynomial,
        max_difference: float,
        pen: QtGui.QPen | None = None,
    ):
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.red, 1.0)
            pen.setCosmetic(True)

        if "max_diff" not in self.lines:
            max_diff = pyqtgraph.PlotCurveItem(pen=pen, connect="pairs")
            self.plot.addItem(max_diff)
            self.lines["max_diff"] = max_diff

        xs = [poly.domain[0], poly.domain[-1], poly.domain[0], poly.domain[-1]]
        ys = poly(xs)
        ys += [
            max_difference,
            max_difference,
            -max_difference,
            -max_difference,
        ]
        self.lines["max_diff"].setData(x=xs, y=ys)
        self.lines["max_diff"].setPen(pen)

    def drawInterpolationLine(
        self, xs: np.ndarray, ys: np.ndarray, pen: QtGui.QPen | None = None
    ):
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.blue, 1.0)
            pen.setCosmetic(True)

        if "interp" not in self.lines:
            interp = pyqtgraph.PlotCurveItem(x=xs, y=ys, pen=pen, skipFiniteCheck=True)
            self.plot.addItem(interp)
            self.lines["interp"] = interp
        else:
            self.lines["interp"].setData(x=xs, y=ys)


class SingleIonHistogramView(SinglePlotGraphicsView):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(
            "Signal Distribution",
            xlabel="Raw Signal",
            ylabel="No. Events",
            viewbox=ViewBoxForceScaleAtZero(),
            parent=parent,
        )
        self.plot.setLimits(xMin=0.0, yMin=0.0)

        self._hist, self._edges = None, None

        self.hist_curve: pyqtgraph.PlotCurveItem | None = None
        self.fit_curve: pyqtgraph.PlotCurveItem | None = None
        self.fit_curves = []

    def clear(self):
        super().clear()
        self.hist_curve = None
        self.fit_curve = None
        self.fit_curves.clear()

    def drawHist(
        self,
        hist: np.ndarray,
        edges: np.ndarray,
        bar_width: float = 1.0,
        bar_offset: float = 0.0,
        pen: QtGui.QPen | None = None,
        brush: QtGui.QBrush | None = None,
    ):
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 1.0)
            pen.setCosmetic(True)

        if brush is None:
            brush = QtGui.QBrush(QtCore.Qt.GlobalColor.gray)

        assert bar_width > 0.0 and bar_width <= 1.0
        assert bar_offset >= 0.0 and bar_offset < 1.0

        widths = np.diff(edges)

        x = np.repeat(edges, 2)

        # Calculate bar start and end points for width / offset
        x[1:-1:2] += widths * ((1.0 - bar_width) / 2.0 + bar_offset)
        x[2::2] -= widths * ((1.0 - bar_width) / 2.0 - bar_offset)

        y = np.zeros(hist.size * 2 + 1, dtype=hist.dtype)
        y[1:-1:2] = hist

        if self.hist_curve is None:
            self.hist_curve = pyqtgraph.PlotCurveItem(
                x=x,
                y=y,
                stepMode="center",
                fillLevel=0,
                fillOutline=True,
                pen=pen,
                brush=brush,
                skipFiniteCheck=True,
            )
            self.plot.addItem(self.hist_curve)
        else:
            self.hist_curve.setData(x=x, y=y, pen=pen, brush=brush)
        self._hist, self._edges = hist, edges
