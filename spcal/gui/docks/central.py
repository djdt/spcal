import logging

from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.dialogs.graphoptions import (
    CompositionsOptionsDialog,
    HistogramOptionsDialog,
    SpectraOptionsDialog,
)
from spcal.gui.graphs.base import SinglePlotGraphicsView
from spcal.gui.graphs.composition import CompositionView
from spcal.gui.graphs.histogram import HistogramView
from spcal.gui.graphs.particle import ParticleView
from spcal.gui.graphs.scatter import ScatterView
from spcal.gui.graphs.spectra import SpectraView
from spcal.gui.util import create_action

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
        self.scatter = ScatterView(font=font)

        self.addWidget(self.particle)
        self.addWidget(self.histogram)
        self.addWidget(self.composition)
        self.addWidget(self.spectra)
        self.addWidget(self.scatter)

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
            self.setCurrentWidget(self.composition)
        elif view == "histogram":
            self.setCurrentWidget(self.histogram)
        elif view == "particle":
            self.setCurrentWidget(self.particle)
        elif view == "spectra":
            self.setCurrentWidget(self.spectra)
        elif view == "scatter":
            self.setCurrentWidget(self.scatter)
        else:  # pragma: no cover, error
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
        else:  # pragma: no cover, error
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

    def dialogGraphOptions(self) -> QtWidgets.QDialog | None:
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

    def zoomReset(self):
        view = self.currentWidget()
        assert isinstance(view, SinglePlotGraphicsView)
        view.zoomReset()
