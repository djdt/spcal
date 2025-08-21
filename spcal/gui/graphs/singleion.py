import numpy as np
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.dists import lognormal
from spcal.gui.graphs.base import SinglePlotGraphicsView
from spcal.gui.graphs.viewbox import ViewBoxForceScaleAtZero


class SingleIonHistogramView(SinglePlotGraphicsView):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(
            "Single Ion Distribution",
            xlabel="Single Ion Signal",
            ylabel="No. Events",
            viewbox=ViewBoxForceScaleAtZero(),
            parent=parent,
        )
        self.plot.setLimits(xMin=0.0, yMin=0.0)

        self.hist, self.edges = None, None

    def draw(
        self,
        data: np.ndarray,
        pen: QtGui.QPen | None = None,
        brush: QtGui.QBrush | None = None,
    ) -> pyqtgraph.PlotCurveItem:
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 1.0)
            pen.setCosmetic(True)
        if brush is None:
            brush = QtGui.QBrush(QtCore.Qt.GlobalColor.gray)

        if data.ndim == 1:
            hist, edges = np.histogram(data, bins=100)
            curve = self.drawHist(hist, edges, pen=pen, brush=brush)
        else:
            hist, edges = data[:-1, 1], data[:, 0]
            curve = self.drawHist(hist, edges, pen=pen, brush=brush)

        self.hist, self.edges = hist, edges

        return curve

    def drawHist(
        self,
        hist: np.ndarray,
        edges: np.ndarray,
        bar_width: float = 1.0,
        bar_offset: float = 0.0,
        pen: QtGui.QPen | None = None,
        brush: QtGui.QBrush | None = None,
    ) -> pyqtgraph.PlotCurveItem:
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

        curve = pyqtgraph.PlotCurveItem(
            x=x,
            y=y,
            stepMode="center",
            fillLevel=0,
            fillOutline=True,
            pen=pen,
            brush=brush,
            skipFiniteCheck=True,
        )
        self.plot.addItem(curve)
        return curve

    def drawLognormalFit(
        self,
        mu: float,
        sigma: float,
        normalise: bool = True,
        pen: QtGui.QPen | None = None,
    ) -> pyqtgraph.PlotCurveItem | None:
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.red, 2.0)
            pen.setCosmetic(True)

        if self.hist is None or self.edges is None:
            return None

        xs = np.linspace(self.edges[0], self.edges[-1], 1000)
        ys = lognormal.pdf(xs, mu, sigma)

        density_factor = 1.0 / (np.sum(self.hist) * (self.edges[1] - self.edges[0]))
        ys /= density_factor

        if normalise:
            xmax = xs[np.argmax(ys)]
            ys = ys / ys.max() * self.hist[np.searchsorted(self.edges, xmax)]

        curve = pyqtgraph.PlotCurveItem(
            x=xs, y=ys, pen=pen, connect="all", skipFiniteCheck=True
        )
        self.plot.addItem(curve)
        return curve
