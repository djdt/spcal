import numpy as np
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.dists import lognormal
from spcal.gui.graphs.base import SinglePlotGraphicsView
from spcal.gui.graphs.viewbox import ViewBoxForceScaleAtZero


class SingleIonScatterView(SinglePlotGraphicsView):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(
            "Extracted Parameters",
            xlabel="m/z",
            ylabel="Shape (σ)",
            viewbox=ViewBoxForceScaleAtZero(),
            parent=parent,
        )

        self.points: pyqtgraph.ScatterPlotItem | None = None
        self.max_diff: pyqtgraph.PlotCurveItem | None = None
        self.plot.setLimits(xMin=0.0, yMin=0.0)

    def clear(self) -> None:
        super().clear()
        self.points = None
        self.max_diff = None

    def drawData(
        self,
        x: np.ndarray,
        y: np.ndarray,
        pen: QtGui.QPen | None = None,
        brush: QtGui.QBrush | None = None,
    ) -> None:
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

    def setValid(self, valid: np.ndarray) -> None:
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
    ) -> None:
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.red, 1.0)
            pen.setCosmetic(True)

        if self.max_diff is None:
            self.max_diff = pyqtgraph.PlotCurveItem(pen=pen, connect="pairs")
            self.plot.addItem(self.max_diff)

        xs = [poly.domain[0], poly.domain[-1], poly.domain[0], poly.domain[-1]]
        ys = poly(xs)
        ys += [
            max_difference,
            max_difference,
            -max_difference,
            -max_difference,
        ]
        self.max_diff.setData(x=xs, y=ys)
        self.max_diff.setPen(pen)


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

    def clear(self) -> None:
        super().clear()
        self.hist_curve = None
        self.fit_curve = None

    def drawHist(
        self,
        hist: np.ndarray,
        edges: np.ndarray,
        bar_width: float = 1.0,
        bar_offset: float = 0.0,
        pen: QtGui.QPen | None = None,
        brush: QtGui.QBrush | None = None,
    ) -> None:
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

    def drawLognormalFit(
        self,
        mu: float,
        sigma: float,
        normalise: bool = True,
        pen: QtGui.QPen | None = None,
    ):
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.red, 2.0)
            pen.setCosmetic(True)

        if self._hist is None or self._edges is None:
            return None

        xs = np.linspace(self._edges[0], self._edges[-1], 1000)
        ys = lognormal.pdf(xs, mu, sigma)

        density_factor = 1.0 / (np.sum(self._hist) * (self._edges[1] - self._edges[0]))
        ys /= density_factor

        if normalise:
            xmax = xs[np.argmax(ys)]
            ys = ys / ys.max() * self._hist[np.searchsorted(self._edges, xmax)]

        if self.fit_curve is None:
            self.fit_curve = pyqtgraph.PlotCurveItem(
                pen=pen, connect="all", skipFiniteCheck=True
            )
            self.plot.addItem(self.fit_curve)

        self.fit_curve.setData(x=xs, y=ys)
