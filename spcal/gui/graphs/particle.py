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

        self.legend_items: dict[str, MultipleItemSampleProxy] = {}
        self.limit_items: list[pyqtgraph.PlotCurveItem] = []

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

    def clear(self) -> None:
        self.legend_items.clear()
        super().clear()

    def drawResult(
        self,
        result: SPCalProcessingResult,
        pen: QtGui.QPen | None = None,
        brush: QtGui.QBrush | None = None,
        scatter_symbol: str = "t",
        scatter_size: float = 6.0,
    ):
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 1.0)
            pen.setCosmetic(True)
        if brush is None:
            brush = QtGui.QBrush(QtCore.Qt.GlobalColor.red)

        diffs = np.diff(result.signals, n=2, append=0, prepend=0) != 0
        curve = pyqtgraph.PlotCurveItem(
            x=result.times[diffs],
            y=result.signals[diffs],
            pen=pen,
            connect="all",
            skipFiniteCheck=True,
        )
        self.plot.addItem(curve)

        maxima = detection_maxima(result.signals, result.regions)
        scatter = pyqtgraph.ScatterPlotItem(
            x=result.times[maxima],
            y=result.signals[maxima],
            size=scatter_size,
            symbol=scatter_symbol,
            pen=None,
            brush=brush,
        )
        self.plot.addItem(scatter)

        legend = MultipleItemSampleProxy(
            pen.color(),
            items=[curve, scatter],  # type: ignore , works
        )

        self.plot.legend.addItem(legend, result.isotope)
        # self.result_items[result.isotope] = (curve, scatter)

    def drawLimits(
        self,
        x: np.ndarray,
        mean: float | np.ndarray,
        limit: float | np.ndarray,
        pen: QtGui.QPen | None = None,
    ) -> None:
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 1.0)
            pen.setCosmetic(True)

        for val, name, style in zip(
            [limit, mean],
            ["Detection Threshold", "Mean"],
            [QtCore.Qt.PenStyle.DashLine, QtCore.Qt.PenStyle.SolidLine],
        ):
            if isinstance(val, float) or val.size == 1:
                nx, y = [x[0], x[-1]], [val, val]
            else:
                diffs = np.diff(val, n=2, append=0, prepend=0) != 0
                nx, y = x[diffs], val[diffs]

            pen.setStyle(style)

            curve = pyqtgraph.PlotCurveItem(
                x=nx,
                y=y,
                name=name,
                pen=pen,
                connect="all",
                skipFiniteCheck=True,
            )
            self.limit_items.append(curve)
            self.plot.addItem(curve)
