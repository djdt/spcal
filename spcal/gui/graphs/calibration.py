import numpy as np
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.calc import weighted_linreg, weights_from_weighting
from spcal.gui.graphs.base import SinglePlotGraphicsView
from spcal.gui.graphs.particle import ExclusionRegion
from spcal.gui.util import create_action


class CalibrationView(SinglePlotGraphicsView):
    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__("Calibration", "Concentration", "Response", parent=parent)
        assert self.plot.vb is not None
        self.plot.vb.setMouseEnabled(x=False, y=False)
        self.plot.vb.enableAutoRange(x=True, y=True)

    def drawTrendline(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weighting: str = "equal",
        pen: QtGui.QPen | None = None,
    ):
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.red, 1.0)
            pen.setCosmetic(True)

        if x.size < 2 or np.all(x == x[0]):
            return

        weights = weights_from_weighting(x, weighting)
        m, b, r2, err = weighted_linreg(x, y, w=weights)
        x0, x1 = x.min(), x.max()

        line = pyqtgraph.PlotCurveItem([x0, x1], [m * x0 + b, m * x1 + b], pen=pen)
        text = pyqtgraph.TextItem(f"r² = {r2:.4f}", anchor=(1, 1))
        text.setPos(x1, m * x1 + b)

        self.plot.addItem(line)
        self.plot.addItem(text)
