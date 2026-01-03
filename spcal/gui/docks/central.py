import logging

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.datafile import SPCalDataFile
from spcal.gui.dialogs.graphoptions import (
    CompositionsOptionsDialog,
    HistogramOptionsDialog,
)
from spcal.gui.graphs import symbols
from spcal.gui.graphs.base import SinglePlotGraphicsView
from spcal.gui.graphs.composition import CompositionView
from spcal.gui.graphs.histogram import HistogramView
from spcal.gui.graphs.particle import ParticleView
from spcal.gui.graphs.spectra import SpectraView
from spcal.gui.util import create_action
from spcal.processing import SPCalProcessingResult

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
        self.composition = CompositionView(font=font)
        self.spectra = SpectraView(font=font)

        self.addWidget(self.particle)  # type: ignore , works
        self.addWidget(self.histogram)  # type: ignore , works
        self.addWidget(self.composition)  # type: ignore , works
        self.addWidget(self.spectra)  # type: ignore , works

        self.action_view_composition = create_action(
            "office-chart-pie",
            "Composition View",
            "Cluster and view results as pie or bar charts.",
            lambda: self.setView("composition"),
            checkable=True,
        )
        self.action_view_histogram = create_action(
            "view-object-histogram-linear",
            "Results View",
            "View signal and calibrated results as histograms.",
            lambda: self.setView("histogram"),
            checkable=True,
        )
        self.action_view_particle = create_action(
            "office-chart-line",
            "Particle View",
            "View raw signal and detected particle peaks.",
            lambda: self.setView("particle"),
            checkable=True,
        )
        self.action_view_spectra = create_action(
            "none",
            "Spectra View",
            "View the mass spectra of selected peaks.",
            lambda: self.setView("spectra"),
            checkable=True,
        )
        self.action_view_particle.setChecked(True)

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

    def setView(self, view: str):
        if view == "composition":
            self.setCurrentWidget(self.composition)
        elif view == "histogram":
            self.setCurrentWidget(self.histogram)
        elif view == "particle":
            self.setCurrentWidget(self.particle)
        elif view == "spectra":
            self.setCurrentWidget(self.spectra)

        self.action_view_options.setEnabled(view != "particle")

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

    def drawResultsSpectra(
        self, data_file: SPCalDataFile, result: SPCalProcessingResult
    ):
        pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 2.0 * self.devicePixelRatio())
        pen.setCosmetic(True)

        bg_regions = np.reshape(result.regions.ravel()[1:-1], (-1, 2))

        self.spectra.drawDataFile(data_file, result.regions, pen=pen)
        self.spectra.drawDataFile(data_file, bg_regions, negative=True, pen=pen)
        self.spectra.setDataLimits(yMin=-0.05, yMax=1.05)
        self.spectra.zoomReset()

    def zoomReset(self):
        view = self.currentWidget()
        assert isinstance(view, SinglePlotGraphicsView)
        view.zoomReset()
