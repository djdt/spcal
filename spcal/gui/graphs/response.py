import numpy as np
import pyqtgraph
from PySide6 import QtCore, QtGui

from spcal.gui.graphs.base import SinglePlotGraphicsView


class ResponseView(SinglePlotGraphicsView):
    def __init__(
        self,
        downsample: int = 64,
        parent: pyqtgraph.GraphicsWidget | None = None,
    ):
        super().__init__("Response TIC", "Time", "Intensity", parent=parent)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.enableAutoRange(x=True, y=True)
        self.plot.setDownsampling(ds=downsample, mode="subsample", auto=True)

        self.signal: pyqtgraph.PlotCurveItem | None = None
        self.signal_mean: pyqtgraph.PlotCurveItem | None = None

        region_pen = QtGui.QPen(QtCore.Qt.red, 1.0)
        region_pen.setCosmetic(True)

        self.region = pyqtgraph.LinearRegionItem(
            pen="grey",
            hoverPen="red",
            brush=QtGui.QBrush(QtCore.Qt.NoBrush),
            hoverBrush=QtGui.QBrush(QtCore.Qt.NoBrush),
            swapMode="block",
        )
        # self.region.movable = False  # prevent moving of region, but not lines
        self.region.lines[0].addMarker("|>", 0.9)
        self.region.lines[1].addMarker("<|", 0.9)
        self.region.sigRegionChangeFinished.connect(self.updateMean)

    @property
    def region_start(self) -> int:
        return int(self.region.lines[0].value())  # type: ignore

    @property
    def region_end(self) -> int:
        return int(self.region.lines[1].value())  # type: ignore

    def clear(self):
        super().clear()
        self.signal_mean = None

    def drawData(
        self,
        x: np.ndarray,
        y: np.ndarray,
        pen: QtGui.QPen | None = None,
    ):
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 1.0)
            pen.setCosmetic(True)

        # optimise by removing points with 0 change in gradient
        diffs = np.diff(y, n=2, append=0, prepend=0) != 0
        self.signal = pyqtgraph.PlotCurveItem(
            x=x[diffs], y=y[diffs], pen=pen, connect="all", skipFiniteCheck=True
        )
        self.plot.addItem(self.signal)

        self.region.setBounds((x[0], x[-1]))
        self.plot.addItem(self.region)

    def drawMean(self, mean: float, pen: QtGui.QPen | None = None):
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.red, 2.0, QtCore.Qt.PenStyle.DashLine)
            pen.setCosmetic(True)

        if self.signal_mean is None:
            self.signal_mean = pyqtgraph.PlotCurveItem(
                pen=pen,
                connect="all",
                skipFiniteCheck=True,
            )
            self.plot.addItem(self.signal_mean)

        self.signal_mean.updateData(
            x=[self.region_start, self.region_end], y=[mean, mean]
        )

    def updateMean(self):
        if self.signal is None:
            return
        ds, _, _ = self.plot.downsampleMode()
        mean = np.mean(
            self.signal.yData[self.region_start // ds : self.region_end // ds]
        )
        self.drawMean(mean)
