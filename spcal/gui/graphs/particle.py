import numpy as np
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.detection import detection_maxima
from spcal.gui.graphs.base import SinglePlotGraphicsView
from spcal.gui.graphs.legends import MultipleItemSampleProxy
from spcal.gui.util import create_action
from spcal.processing import SPCalProcessingResult


class ParticleView(SinglePlotGraphicsView):
    regionChanged = QtCore.Signal()
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
        self.has_image_export = True
        self.xaxis.setScale(xscale)
        self.xaxis.enableAutoSIPrefix(False)

        # self.raw_signals: dict[str, np.ndarray] = {}  # for export
        self.result_items: dict[
            str, tuple[pyqtgraph.PlotCurveItem, pyqtgraph.ScatterPlotItem]
        ] = {}

        assert self.plot.vb is not None
        self.plot.vb.setLimits(xMin=0.0, xMax=1.0, yMin=0.0)
        self.setAutoScaleY(True)

        self.plot.legend.setColumnCount(3)
        # self.legend_items: dict[str, MultipleItemSampleProxy] = {}
        # self.limit_items: list[pyqtgraph.PlotCurveItem] = []

        region_pen = QtGui.QPen(QtCore.Qt.GlobalColor.red, 1.0)
        region_pen.setCosmetic(True)

        self.region = pyqtgraph.LinearRegionItem(
            pen="grey",
            hoverPen="red",
            brush=QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush),
            hoverBrush=QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush),
            swapMode="block",
        )
        self.region.sigRegionChangeFinished.connect(self.regionChanged)
        self.region.movable = False  # prevent moving of region, but not lines
        self.region.lines[0].addMarker("|>", 0.9)
        self.region.lines[1].addMarker("<|", 0.9)
        self.plot.addItem(self.region)

        self.action_peak_properties = create_action(
            "office-chart-area-focus-peak-node",
            "Peak Properties",
            "Show peak widths, heights and other properties.",
            self.requestPeakProperties,
        )
        self.context_menu_actions.append(self.action_peak_properties)

    @property
    def region_start(self) -> int:
        return int(self.region.lines[0].value())  # type: ignore

    @property
    def region_end(self) -> int:
        return int(self.region.lines[1].value())  # type: ignore

    def dataForExport(self) -> dict[str, np.ndarray]:
        start, end = self.region_start, self.region_end
        return {k: v[start:end] for k, v in self.export_data.items()}

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

        maxima = detection_maxima(
            result.signals, result.regions[result.filter_indicies]
        )

        scatter = self.drawScatter(
            result.times[maxima],
            result.signals[maxima],
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

        if self.plot.legend is not None:
            legend = MultipleItemSampleProxy(
                pen.color(),
                items=[curve, scatter],  # type: ignore , works
            )
            self.plot.legend.addItem(legend, str(result.isotope))

        self.setDataLimits(xMin=0.0, xMax=1.0)
