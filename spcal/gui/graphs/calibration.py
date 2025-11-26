import numpy as np
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.calc import weighted_linreg
from spcal.gui.graphs.base import SinglePlotGraphicsView
from spcal.gui.graphs.particle import ExclusionRegion


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
        weighting: str = "none",
        pen: QtGui.QPen | None = None,
    ):
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.red, 1.0)
            pen.setCosmetic(True)

        if weighting != "none":
            raise NotImplementedError("Weighting not yet implemented.")
        if x.size < 2 or np.all(x == x[0]):
            return

        m, b, r2, err = weighted_linreg(x, y, w=None)
        x0, x1 = x.min(), x.max()

        line = pyqtgraph.PlotCurveItem([x0, x1], [m * x0 + b, m * x1 + b], pen=pen)
        text = pyqtgraph.TextItem(f"r² = {r2:.4f}", anchor=(1, 1))
        text.setPos(x1, m * x1 + b)

        self.plot.addItem(line)
        self.plot.addItem(text)


class IntensityView(SinglePlotGraphicsView):
    def __init__(
        self,
        downsample: int = 64,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__("Response TIC", "Time", "Intensity", parent=parent)
        assert self.plot.vb is not None
        self.plot.vb.setMouseEnabled(x=False, y=False)
        self.plot.vb.enableAutoRange(x=True, y=True)
        self.plot.setDownsampling(ds=downsample, mode="subsample", auto=True)

    def addExclusionRegion(self, start: float | None = None, end: float | None = None):
        if self.plot.vb is None:
            return
        x0, x1 = self.plot.vb.state["limits"]["xLimits"]
        if start is None or end is None:
            pos = self.plot.vb.mapSceneToView(self.mapFromGlobal(self.cursor().pos()))
            start = pos.x() - (x1 - x0) * 0.05
            end = pos.x() + (x1 - x0) * 0.05
        region = ExclusionRegion(start, end)  # type: ignore not None, see above
        region.sigRegionChangeFinished.connect(self.exclusionRegionChanged)
        region.requestRemoval.connect(self.removeExclusionRegion)
        region.setBounds((x0, x1))
        self.plot.addItem(region)
