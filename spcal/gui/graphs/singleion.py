import numpy as np
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.calc import erfinv
from spcal.fit import lognormal_pdf
from spcal.gui.graphs.base import SinglePlotGraphicsView
from spcal.gui.graphs.viewbox import ViewBoxForceScaleAtZero


class SingleIonView(SinglePlotGraphicsView):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(
            "Single Ion Distribution",
            xlabel="Single Ion Signal",
            ylabel="No. Events",
            viewbox=ViewBoxForceScaleAtZero(),
            parent=parent,
        )
        self.plot.setLimits(xMin=0.0, yMin=0.0)

    def draw(
        self,
        data: np.ndarray | float,
        sigma: float = 0.50,
        pen: QtGui.QPen | None = None,
        brush: QtGui.QBrush | None = None,
        draw_fit: str | None = None,
    ) -> None:
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 1.0)
            pen.setCosmetic(True)
        if brush is None:
            brush = QtGui.QBrush(QtCore.Qt.black)

        if isinstance(data, float):
            mu = np.log(data) + sigma**2
            p99 = np.exp(mu + np.sqrt(2.0 * sigma**2) * erfinv(2.0 * 0.99 - 1.0))

            xs = np.linspace(0.0, p99, 1000)

            ys = lognormal_pdf(xs, mu, sigma)
            curve = pyqtgraph.PlotCurveItem(
                x=xs,
                y=np.nan_to_num(ys),
                pen=pen,
                connect="all",
                skipFiniteCheck=True,
                antialias=True,
            )
        elif data.ndim == 1:
            hist, edges = np.histogram(data)
            self.drawHist(hist, edges, pen=pen, brush=brush)
        else:
            hist, edges = data[:-1, 1], data[:, 0]
            curve = self.drawHist(hist, edges, pen=pen, brush=brush)

        self.plot.addItem(curve)

    def drawHist(
        self,
        hist: np.ndarray,
        edges: np.ndarray,
        bar_width: float = 0.5,
        bar_offset: float = 0.0,
        pen: QtGui.QPen | None = None,
        brush: QtGui.QBrush | None = None,
    ) -> pyqtgraph.PlotCurveItem:
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 1.0)
            pen.setCosmetic(True)

        if brush is None:
            brush = QtGui.QBrush(QtCore.Qt.black)

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
        return curve
