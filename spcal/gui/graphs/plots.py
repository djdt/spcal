from typing import Dict, List, Tuple

import numpy as np
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.graphs.legends import HistogramItemSample, MultipleItemSampleProxy
from spcal.gui.graphs.viewbox import ViewBoxForceScaleAtZero


class HistogramPlotItem(pyqtgraph.PlotItem):
    def __init__(
        self,
        name: str,
        xlabel: str,
        xunit: str,
        pen: QtGui.QPen | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        self.name = name
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 1.0)
            pen.setCosmetic(True)

        self.xaxis = pyqtgraph.AxisItem("bottom", pen=pen, textPen=pen, tick_pen=pen)
        self.yaxis = pyqtgraph.AxisItem("left", pen=pen, textPen=pen, tick_pen=pen)
        self.yaxis.enableAutoSIPrefix(False)

        super().__init__(
            title="Results Histogram",
            name="hist",
            axisItems={"bottom": self.xaxis, "left": self.yaxis},
            viewBox=ViewBoxForceScaleAtZero(enableMenu=False),
            parent=parent,
        )
        # Todo: not working
        self.xaxis.setLabel(text=xlabel, units=xunit)
        self.yaxis.setLabel("No. Events")

        self.hideButtons()
        self.addLegend(
            offset=(-5, 5),
            verSpacing=-5,
            colCount=1,
            labelTextColor="black",
            sampleType=HistogramItemSample,
        )

        self.legends: Dict[str, HistogramItemSample] = {}

    def drawData(
        self,
        name: str,
        x: np.ndarray,
        bins: str | np.ndarray = "auto",
        bar_width: float = 0.5,
        bar_offset: float = 0.0,
        pen: QtGui.QPen | None = None,
        brush: QtGui.QBrush | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 1.0)
            pen.setCosmetic(True)
        if brush is None:
            brush = QtGui.QBrush(QtCore.Qt.black)

        assert bar_width > 0.0 and bar_width <= 1.0
        assert bar_offset >= 0.0 and bar_offset < 1.0

        hist, edges = np.histogram(x, bins)
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

        self.legends[name] = HistogramItemSample(curve)
        self.addItem(curve)
        self.legend.addItem(self.legends[name], name)

        return hist, (x[1:-1:2] + x[2:-1:2]) / 2.0

    def drawFit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        pen: QtGui.QPen | None = None,
        name: str | None = None,
    ) -> None:

        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 1.0)
            pen.setCosmetic(True)

        curve = pyqtgraph.PlotCurveItem(
            x=x, y=y, pen=pen, connect="all", skipFiniteCheck=True
        )
        self.addItem(curve)
        if name is not None:
            self.legends[name].setFit(curve)
            self.legend.updateSize()


class ParticlePlotItem(pyqtgraph.PlotItem):
    def __init__(
        self,
        name: str,
        xscale: float = 1.0,
        pen: QtGui.QPen | None = None,
        parent: pyqtgraph.GraphicsWidget | None = None,
    ):
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.black, 1.0)
            pen.setCosmetic(True)

        self.xaxis = pyqtgraph.AxisItem("bottom", pen=pen, textPen=pen, tick_pen=pen)
        self.xaxis.setLabel(text="Time", units="s")
        self.xaxis.setScale(xscale)
        self.xaxis.enableAutoSIPrefix(False)

        self.yaxis = pyqtgraph.AxisItem("left", pen=pen, textPen=pen, tick_pen=pen)
        self.yaxis.setLabel(text="Intensity (counts)", units="")

        super().__init__(
            title=name,
            name=name,
            axisItems={"bottom": self.xaxis, "left": self.yaxis},
            enableMenu=False,
            parent=parent,
        )
        self.setMouseEnabled(y=False)
        self.setAutoVisible(y=True)
        self.enableAutoRange(y=True)
        self.hideButtons()
        self.addLegend(
            offset=(-5, 5), verSpacing=-5, colCount=1, labelTextColor="black"
        )

        self.signals: Dict[str, pyqtgraph.PlotCurveItem] = {}
        self.scatters: Dict[str, pyqtgraph.ScatterPlotItem] = {}
        self.legends: Dict[str, MultipleItemSampleProxy] = {}
        self.limits: List[pyqtgraph.PlotCurveItem] = []

        region_pen = QtGui.QPen(QtCore.Qt.red, 1.0)
        region_pen.setCosmetic(True)

        self.region = pyqtgraph.LinearRegionItem(
            pen="grey",
            hoverPen="red",
            brush=QtGui.QBrush(QtCore.Qt.NoBrush),
            hoverBrush=QtGui.QBrush(QtCore.Qt.NoBrush),
            swapMode="block",
        )
        self.region.movable = False  # prevent moving of region, but not lines
        self.region.lines[0].addMarker("|>", 0.9)
        self.region.lines[1].addMarker("<|", 0.9)
        self.addItem(self.region)

    @property
    def region_start(self) -> int:
        return int(self.region.lines[0].value())  # type: ignore

    @property
    def region_end(self) -> int:
        return int(self.region.lines[1].value())  # type: ignore

    def clearSignal(self) -> None:
        for signal in self.signals.values():
            self.removeItem(signal)
        self.signals.clear()

    def clearScatters(self) -> None:
        for scatter in self.scatters.values():
            self.removeItem(scatter)
        self.scatters.clear()

    def clearLimits(self) -> None:
        for limit in self.limits:
            self.removeItem(limit)
        self.limits.clear()

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
        curve = pyqtgraph.PlotCurveItem(
            x=x[diffs], y=y[diffs], pen=pen, connect="all", skipFiniteCheck=True
        )

        legend_item = MultipleItemSampleProxy(pen.color(), items=[curve])
        self.signals[name] = curve
        self.legends[name] = legend_item

        self.addItem(curve)
        self.legend.addItem(legend_item, name)

        self.setLimits(
            xMin=x[0],
            xMax=x[-1],
            yMin=0,
            minXRange=(x[-1] - x[0]) * 1e-5,
            minYRange=np.ptp(y[diffs]) * 0.01,
        )
        self.enableAutoRange(y=True)  # rescale to max bounds

        self.region.blockSignals(True)
        self.region.setBounds((x[0], x[-1]))
        self.region.blockSignals(False)

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
        self.scatters[name] = scatter
        self.addItem(scatter)

        self.legends[name].addItem(scatter)

    def drawLimits(
        self,
        x: np.ndarray,
        mean: float | np.ndarray,
        threshold: float | np.ndarray,
    ) -> None:
        # skip_lc = np.all(limits["lc"] == limits["ld"])
        pen = QtGui.QPen(QtCore.Qt.red, 1.0, QtCore.Qt.DashLine)
        pen.setCosmetic(True)

        for limit, label, color in zip(
            [mean, threshold],
            ["Mean", "Detection Threshold"],
            [QtCore.Qt.red, QtCore.Qt.blue],
        ):
            pen.setColor(color)
            if isinstance(limit, float):
                nx, y = [x[0], x[-1]], [limit, limit]
            else:
                diffs = np.diff(limit, n=2, append=0, prepend=0) != 0
                nx, y = x[diffs], limit[diffs]

            curve = pyqtgraph.PlotCurveItem(
                x=nx, y=y, name=label, pen=pen, connect="all", skipFiniteCheck=True
            )
            self.limits.append(curve)
            self.addItem(curve)
