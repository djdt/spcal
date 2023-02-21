from typing import Dict, List

import numpy as np
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.graphs.base import PlotCurveItemFix, SinglePlotGraphicsView
from spcal.gui.graphs.legends import MultipleItemSampleProxy


class ParticleView(SinglePlotGraphicsView):
    regionChanged = QtCore.Signal()

    def __init__(
        self,
        xscale: float = 1.0,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(
            "Signal",
            xlabel="Time",
            xunits="s",
            ylabel="Intensity (counts)",
            parent=parent,
        )
        self.xaxis.setScale(xscale)

        self.plot.setMouseEnabled(y=False)
        self.plot.setAutoVisible(y=True)
        self.plot.enableAutoRange(y=True)
        self.plot.setLimits(yMin=0.0)

        self.legend_items: Dict[str, MultipleItemSampleProxy] = {}
        self.limit_items: List[pyqtgraph.PlotCurveItem] = []

        region_pen = QtGui.QPen(QtCore.Qt.red, 1.0)
        region_pen.setCosmetic(True)

        self.region = pyqtgraph.LinearRegionItem(
            pen="grey",
            hoverPen="red",
            brush=QtGui.QBrush(QtCore.Qt.NoBrush),
            hoverBrush=QtGui.QBrush(QtCore.Qt.NoBrush),
            swapMode="block",
        )
        self.region.sigRegionChangeFinished.connect(self.regionChanged)
        self.region.movable = False  # prevent moving of region, but not lines
        self.region.lines[0].addMarker("|>", 0.9)
        self.region.lines[1].addMarker("<|", 0.9)
        self.plot.addItem(self.region)

    @property
    def region_start(self) -> int:
        return int(self.region.lines[0].value())  # type: ignore

    @property
    def region_end(self) -> int:
        return int(self.region.lines[1].value())  # type: ignore

    def clear(self) -> None:
        self.legend_items.clear()
        super().clear()

    def clearScatters(self) -> None:
        for item in self.plot.listDataItems():
            if isinstance(item, pyqtgraph.ScatterPlotItem):
                self.plot.removeItem(item)

    def clearLimits(self) -> None:
        for limit in self.limit_items:
            self.plot.removeItem(limit)
        self.limit_items.clear()

    def drawSignal(
        self,
        name: str,
        x: np.ndarray,
        y: np.ndarray,
        pen: QtGui.QPen | None = None,
    ) -> None:
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 1.0)
            pen.setCosmetic(True)

        # optimise by removing points with 0 change in gradient
        diffs = np.diff(y, n=2, append=0, prepend=0) != 0
        curve = PlotCurveItemFix(
            x=x[diffs], y=y[diffs], pen=pen, connect="all", skipFiniteCheck=True
        )

        self.legend_items[name] = MultipleItemSampleProxy(pen.color(), items=[curve])

        self.plot.addItem(curve)
        self.plot.legend.addItem(self.legend_items[name], name)

    def drawMaxima(
        self,
        name: str,
        x: np.ndarray,
        y: np.ndarray,
        brush: QtGui.QBrush | None = None,
        symbol: str = "t",
    ) -> None:
        if brush is None:
            brush = QtGui.QBrush(QtCore.Qt.red)

        scatter = pyqtgraph.ScatterPlotItem(
            x=x, y=y, size=6, symbol=symbol, pen=None, brush=brush
        )
        self.plot.addItem(scatter)

        self.legend_items[name].addItem(scatter)

    def drawLimits(
        self,
        x: np.ndarray,
        # mean: float | np.ndarray,
        limit: float | np.ndarray,
        pen: QtGui.QPen | None = None,
    ) -> None:
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 1.0, QtCore.Qt.DashLine)
            pen.setCosmetic(True)

        if isinstance(limit, float) or limit.size == 1:
            nx, y = [x[0], x[-1]], [limit, limit]
        else:
            diffs = np.diff(limit, n=2, append=0, prepend=0) != 0
            nx, y = x[diffs], limit[diffs]

        curve = pyqtgraph.PlotCurveItem(
            x=nx,
            y=y,
            name="Detection Threshold",
            pen=pen,
            connect="all",
            skipFiniteCheck=True,
        )
        self.limit_items.append(curve)
        self.plot.addItem(curve)
