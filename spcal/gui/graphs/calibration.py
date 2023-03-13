import numpy as np
import pyqtgraph
from PySide6 import QtCore, QtGui

from spcal.gui.graphs.base import SinglePlotGraphicsView


class CalibrationView(SinglePlotGraphicsView):
    def __init__(
        self,
        parent: pyqtgraph.GraphicsWidget | None = None,
    ):
        super().__init__("Calibration", "Concentration", "Response", parent=parent)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.enableAutoRange(x=True, y=True)

    def drawPoints(
        self,
        x: np.ndarray,
        y: np.ndarray,
        name: str | None = None,
        draw_trendline: bool = False,
        pen: QtGui.QPen | None = None,
        brush: QtGui.QPen | None = None,
    ) -> None:
        if brush is None:
            brush = QtGui.QBrush(QtCore.Qt.red)
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 1.0)
            pen.setCosmetic(True)

        scatter = pyqtgraph.ScatterPlotItem(
            x, y, symbol="o", size=10, pen=pen, brush=brush
        )
        self.plot.addItem(scatter)

        if name is not None:
            self.plot.legend.addItem(scatter, name)

        if draw_trendline:
            tpen = QtGui.QPen(brush.color(), 1.0)
            tpen.setCosmetic(True)
            self.drawTrendline(x, y, pen=tpen)

    def drawTrendline(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weighting: str = "none",
        pen: QtGui.QPen | None = None,
    ) -> None:
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.red, 1.0)
            pen.setCosmetic(True)

        if weighting != "none":
            raise NotImplementedError("Weighting not yet implemented.")
        if x.size < 2 or np.all(x == x[0]):
            return

        b, m = np.polynomial.polynomial.polyfit(x, y, 1, w=None)

        x0, x1 = x.min(), x.max()

        line = pyqtgraph.PlotCurveItem([x0, x1], [m * x0 + b, m * x1 + b], pen=pen)

        self.plot.addItem(line)
