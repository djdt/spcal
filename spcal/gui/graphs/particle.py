import numpy as np
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.graphs.base import SinglePlotGraphicsView
from spcal.gui.graphs.legends import MultipleItemSampleProxy
from spcal.gui.util import create_action
from spcal.processing import SPCalProcessingResult


class ExclusionRegion(pyqtgraph.LinearRegionItem):
    requestRemoval = QtCore.Signal()

    def __init__(self, start: float, end: float):
        super().__init__(
            values=(start, end),
            pen="grey",
            hoverPen="red",
            brush=QtGui.QBrush(QtCore.Qt.BrushStyle.BDiagPattern),
            hoverBrush=QtGui.QBrush(QtCore.Qt.BrushStyle.BDiagPattern),
            swapMode="block",
        )
        self.lines[0].addMarker("|>", 0.9)
        self.lines[1].addMarker("<|", 0.9)

    @property
    def start(self) -> float:
        return float(self.lines[0].value())

    @property
    def end(self) -> float:
        return float(self.lines[1].value())

    def contextMenuEvent(self, event: QtWidgets.QGraphicsSceneContextMenuEvent):
        close_action = create_action(
            "window-close",
            "Remove Exclusion",
            "Stop blocking processin in this area.",
            self.requestRemoval,
        )
        menu = QtWidgets.QMenu()
        menu.addAction(close_action)
        menu.exec(event.screenPos())


class ParticleView(SinglePlotGraphicsView):
    exclusionRegionChanged = QtCore.Signal()
    requestPeakProperties = QtCore.Signal()

    def __init__(
        self,
        xscale: float = 1.0,
        font: QtGui.QFont | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(
            "Signal",
            xlabel="Time",
            xunits="s",
            ylabel="Intensity (counts)",
            font=font,
            parent=parent,
        )
        self.plot.xaxis.setScale(xscale)
        self.plot.xaxis.enableAutoSIPrefix(False)

        assert self.plot.vb is not None
        self.plot.vb.setLimits(xMin=0.0, xMax=1.0, yMin=0.0)
        self.setAutoScaleY(True)

        if self.plot.legend is not None:
            self.plot.legend.setColumnCount(3)

        self.action_peak_properties = create_action(
            "office-chart-area-focus-peak-node",
            "Peak Properties",
            "Show peak widths, heights and other properties.",
            self.requestPeakProperties,
        )
        self.context_menu_actions.append(self.action_peak_properties)

        self.action_exclusion_region = create_action(
            "removecell",
            "Add Exclusion Region",
            "Prevent analysis in a region of the data.",
            self.addExclusionRegion,
        )
        self.action_exclusion_region.triggered.connect(self.exclusionRegionChanged)
        self.context_menu_actions.append(self.action_exclusion_region)

    def exclusionRegions(self) -> list[tuple[float, float]]:
        regions = []
        for item in self.plot.items:
            if isinstance(item, ExclusionRegion):
                regions.append((item.start, item.end))
        return regions

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

    def removeExclusionRegion(self):
        region = self.sender()
        if not isinstance(region, ExclusionRegion):
            return
        self.plot.removeItem(region)
        self.exclusionRegionChanged.emit()

    def drawResult(
        self,
        result: SPCalProcessingResult,
        pen: QtGui.QPen | None = None,
        brush: QtGui.QBrush | None = None,
        scatter_size: float = 6.0,
        scatter_symbol: str = "t",
    ):
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 1.0)
            pen.setCosmetic(True)
        if brush is None:
            brush = QtGui.QBrush(QtCore.Qt.GlobalColor.red)

        curve = self.drawCurve(result.times, result.signals, pen)

        scatter = self.drawScatter(
            result.times[result.maxima[result.filter_indicies]],
            result.signals[result.maxima[result.filter_indicies]],
            pen=None,
            brush=brush,
            size=scatter_size,
            symbol=scatter_symbol,
        )

        for val, name, style in zip(
            [result.limit.detection_threshold, result.limit.mean_signal],
            ["Threshold", "Mean"],
            [QtCore.Qt.PenStyle.DashLine, QtCore.Qt.PenStyle.DotLine],
        ):
            pen.setStyle(style)
            if isinstance(val, np.ndarray):
                self.drawCurve(result.times, val, pen=pen, name=name)
            else:
                self.drawLine(
                    float(val), QtCore.Qt.Orientation.Horizontal, pen=pen, name=name
                )

        self.data_for_export[str(result.isotope) + "_x"] = curve.xData  # type: ignore , just set
        self.data_for_export[str(result.isotope) + "_y"] = curve.yData  # type: ignore , just set
        self.data_for_export[str(result.isotope) + "_particle_x"] = scatter.getData()[0]
        self.data_for_export[str(result.isotope) + "_particle_y"] = scatter.getData()[1]

        if self.plot.legend is not None:
            legend = MultipleItemSampleProxy(
                pen.color(),
                items=[curve, scatter],  # type: ignore , works
            )
            self.plot.legend.addItem(legend, str(result.isotope))

        self.setDataLimits(xMin=0.0, xMax=1.0)
