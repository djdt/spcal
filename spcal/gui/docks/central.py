import logging

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.datafile import SPCalDataFile
from spcal.gui.dialogs.graphoptions import (
    CompositionsOptionsDialog,
    HistogramOptionsDialog,
    SpectraOptionsDialog,
)
from spcal.gui.graphs import symbols
from spcal.gui.graphs.base import SinglePlotGraphicsView
from spcal.gui.graphs.composition import CompositionView
from spcal.gui.graphs.histogram import HistogramView
from spcal.gui.graphs.particle import ParticleView
from spcal.gui.graphs.scatter import ScatterView
from spcal.gui.graphs.spectra import SpectraView
from spcal.gui.util import create_action
from spcal.isotope import SPCalIsotopeBase
from spcal.processing.result import SPCalProcessingResult

logger = logging.getLogger(__name__)


class SPCalCentralWidget(QtWidgets.QStackedWidget):
    requestRedraw = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Signal Graph")

        settings = QtCore.QSettings()
        font = QtGui.QFont(
            str(settings.value("GraphFont/Family", "SansSerif")),
            pointSize=int(settings.value("GraphFont/PointSize", 10)),  # type: ignore
        )

        self.particle = ParticleView(font=font)
        self.histogram = HistogramView(font=font)
        self.composition = CompositionView(font=font)  # type: ignore
        self.spectra = SpectraView(font=font)
        self.scatter = ScatterView(font=font)

        self.addWidget(self.particle)  # type: ignore , works
        self.addWidget(self.histogram)  # type: ignore , works
        self.addWidget(self.composition)  # type: ignore , works
        self.addWidget(self.spectra)  # type: ignore , works
        self.addWidget(self.scatter)  # type: ignore , works

        self.action_view_options = create_action(
            "configure",
            "Graph Options",
            "Set options specific to the current graph.",
            self.dialogGraphOptions,
        )
        self.action_view_options.setEnabled(False)

        self.action_zoom_reset = create_action(
            "zoom-original",
            "Reset Zoom",
            "Reset the zoom to the full graph extent.",
            self.zoomReset,
        )

    def clear(self):
        for i in range(self.count()):
            widget = self.widget(i)
            if isinstance(widget, SinglePlotGraphicsView):
                widget.clear()

    @QtCore.Slot()
    def setView(self, view: str):
        if view == "composition":
            self.setCurrentWidget(self.composition)  # type: ignore
        elif view == "histogram":
            self.setCurrentWidget(self.histogram)  # type: ignore
        elif view == "particle":
            self.setCurrentWidget(self.particle)  # type: ignore
        elif view == "spectra":
            self.setCurrentWidget(self.spectra)  # type: ignore
        elif view == "scatter":
            self.setCurrentWidget(self.scatter)  # type: ignore
        else:
            raise ValueError(f"unknown view {view}")

        self.action_view_options.setEnabled(view != "particle")
        self.requestRedraw.emit()

    def currentView(self) -> str:
        view = self.currentWidget()
        if view == self.particle:
            return "particle"
        elif view == self.histogram:
            return "histogram"
        elif view == self.composition:
            return "composition"
        elif view == self.spectra:
            return "spectra"
        elif view == self.scatter:
            return "scatter"
        else:
            raise ValueError("current view is invalid")

    def setGraphFont(self, font: QtGui.QFont):
        for i in range(self.count()):
            widget = self.widget(i)
            if isinstance(widget, SinglePlotGraphicsView):
                widget.setFont(font)

    def setCompositionOptions(self, min_size: str | float, mode: str):
        self.composition.min_size = min_size
        self.composition.mode = mode
        self.requestRedraw.emit()

    def setHistogramOptions(
        self, widths: dict[str, float | None], percentile: float, draw_filtered: bool
    ):
        self.histogram.bin_widths = widths
        self.histogram.max_percentile = percentile
        self.histogram.draw_filtered = draw_filtered
        self.requestRedraw.emit()

    def setSpectraOptions(self, subtract_background: bool):
        self.spectra.subtract_background = subtract_background
        self.requestRedraw.emit()

    def dialogGraphOptions(self):
        view = self.currentView()
        if view == "histogram":
            dlg = HistogramOptionsDialog(
                bin_widths=self.histogram.bin_widths,
                percentile=self.histogram.max_percentile,
                draw_filtered=self.histogram.draw_filtered,
                parent=self,
            )
            dlg.optionsChanged.connect(self.setHistogramOptions)
        elif view == "composition":
            dlg = CompositionsOptionsDialog(
                minimum_size=self.composition.min_size,
                mode=self.composition.mode,
                parent=self,
            )
            dlg.optionsChanged.connect(self.setCompositionOptions)
        elif view == "spectra":
            dlg = SpectraOptionsDialog(
                subtract_background=self.spectra.subtract_background, parent=self
            )
            dlg.optionsChanged.connect(self.setSpectraOptions)
        else:
            return

        dlg.open()
        return dlg

    def drawResultsParticle(
        self,
        results: list[SPCalProcessingResult],
        colors: list[QtGui.QColor],
        names: list[str],
        key: str,
    ):
        for i, (result, color, name) in enumerate(zip(results, colors, names)):
            symbol = symbols[i % len(symbols)]
            pen = QtGui.QPen(color, 1.0)  # cant change to other than 1, slow
            pen.setCosmetic(True)
            brush = QtGui.QBrush(color)
            self.particle.drawResult(
                result,
                pen=pen,
                brush=brush,
                scatter_size=5.0 * np.sqrt(self.devicePixelRatio()),
                scatter_symbol=symbol,
                label=name,
            )

    def drawResultsComposition(
        self,
        results: list[SPCalProcessingResult],
        colors: list[QtGui.QColor],
        key: str,
        clusters: np.ndarray,
    ):
        pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 1.0 * self.devicePixelRatio())
        pen.setCosmetic(True)

        brushes = []
        for result, color in zip(results, colors):
            color.setAlphaF(0.66)
            brushes.append(QtGui.QBrush(color))
        self.composition.drawResults(results, clusters, key, pen, brushes)

    def drawResultsHistogram(
        self,
        results: list[SPCalProcessingResult],
        colors: list[QtGui.QColor],
        names: list[str],
        key: str,
    ):
        pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 1.0 * self.devicePixelRatio())
        pen.setCosmetic(True)

        drawable = []
        brushes = []
        for result, color in zip(results, colors):
            if result.canCalibrate(key):
                drawable.append(result)
                color.setAlphaF(0.66)
                brushes.append(QtGui.QBrush(color))

        if len(drawable) > 0:
            self.histogram.drawResults(
                drawable, key, pen=pen, brushes=brushes, labels=names
            )

    def drawResultsScatterExpr(
        self,
        results: dict[SPCalIsotopeBase, SPCalProcessingResult],
        text_x: str,
        text_y: str,
        key_x: str,
        key_y: str,
    ):
        pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 1.0)
        pen.setCosmetic(True)

        self.scatter.drawResultsExpr(results, text_x, text_y, key_x, key_y)

    # def drawResultsScatter(
    #     self, result_x: SPCalProcessingResult, result_y: SPCalProcessingResult, key: str
    # ):
    #     pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 1.0)
    #     pen.setCosmetic(True)
    #
    #     self.scatter.drawResults(result_x, result_y, key, pen=pen)

    def drawResultsSpectra(
        self,
        data_file: SPCalDataFile,
        result: SPCalProcessingResult,
        reverse_result: SPCalProcessingResult | None = None,
    ):
        pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 2.0 * self.devicePixelRatio())
        pen.setCosmetic(True)

        regions = result.regions[result.filter_indicies]

        try:
            self.spectra.drawDataFile(data_file, regions, pen=pen)
        except NotImplementedError:
            logger.warning(f"spectra not implemented for {type(data_file)}")
            return
        if reverse_result is not None:
            reverse_regions = reverse_result.regions[reverse_result.filter_indicies]
            self.spectra.drawDataFile(
                data_file, reverse_regions, negative=True, pen=pen
            )

        self.spectra.setDataLimits(yMin=-0.05, yMax=1.05)
        self.spectra.zoomReset()

    def zoomReset(self):
        view = self.currentWidget()
        assert isinstance(view, SinglePlotGraphicsView)
        view.zoomReset()
