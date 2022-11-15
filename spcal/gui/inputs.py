import logging
from typing import Dict, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

import spcal
from spcal.calc import calculate_limits
from spcal.detection import combine_detections, detection_maxima
from spcal.gui.dialogs import ImportDialog
from spcal.gui.graphs import ParticleView, color_schemes, symbols
from spcal.gui.iowidgets import IOStack, ReferenceIOStack, SampleIOStack
from spcal.gui.options import OptionsWidget
from spcal.gui.util import create_action
from spcal.gui.widgets import ElidedLabel

logger = logging.getLogger(__name__)


class InputWidget(QtWidgets.QWidget):
    optionsChanged = QtCore.Signal()
    detectionsChanged = QtCore.Signal()
    limitsChanged = QtCore.Signal()

    dataImported = QtCore.Signal()

    def __init__(
        self,
        io_stack: IOStack,
        options: OptionsWidget,
        color_scheme: str = "IBM Carbon",
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setAcceptDrops(True)

        self.color_scheme = color_scheme

        self.import_options = {}

        self.responses = np.array([])
        self.events = np.array([])
        self.detections = np.array([])
        self.labels = np.array([])
        self.regions = np.array([])
        self.limits: Dict[str, Tuple[str, Dict[str, float], np.ndarray]] = {}

        self.redraw_graph_requested = False
        self.draw_mode = "Overlay"

        self.graph_toolbar = QtWidgets.QToolBar()
        self.graph_toolbar.setOrientation(QtCore.Qt.Vertical)
        self.graph_toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)

        self.graph = ParticleView()
        self.graph.regionChanged.connect(self.saveTrimRegion)
        self.graph.regionChanged.connect(self.updateLimits)
        self.last_region: Tuple[int, int] | None = None

        self.io = io_stack

        self.limitsChanged.connect(self.updateDetections)
        self.limitsChanged.connect(self.drawLimits)

        self.detectionsChanged.connect(self.updateOutputs)
        self.detectionsChanged.connect(self.drawDetections)

        self.options = options
        self.options.dwelltime.valueChanged.connect(self.updateLimits)
        self.options.method.currentTextChanged.connect(self.updateLimits)
        self.options.window_size.editingFinished.connect(self.updateLimits)
        self.options.check_use_window.toggled.connect(self.updateLimits)
        self.options.sigma.editingFinished.connect(self.updateLimits)
        self.options.manual.editingFinished.connect(self.updateLimits)
        self.options.error_rate_alpha.editingFinished.connect(self.updateLimits)
        self.options.error_rate_beta.editingFinished.connect(self.updateLimits)
        self.options.efficiency_method.currentTextChanged.connect(
            self.onEfficiencyMethodChanged
        )

        self.button_file = QtWidgets.QPushButton("Open File")
        self.button_file.pressed.connect(self.dialogLoadFile)

        self.label_file = ElidedLabel()

        # Actions

        self.action_graph_overlay = create_action(
            "office-chart-line",
            "Overlay Signals",
            "Overlay all element signals in a single plot.",
            lambda: self.setDrawMode("Overlay"),
            checkable=True,
        )
        self.action_graph_overlay.setChecked(True)
        self.action_graph_stacked = create_action(
            "object-rows",
            "Stacked Signals",
            "Draw element signals in separate plots.",
            lambda: self.setDrawMode("Stacked"),
            checkable=True,
        )
        self.action_graph_zoomout = create_action(
            "zoom-original",
            "Zoom Out",
            "Reset the plot view.",
            self.graph.zoomReset,
        )
        action_group_graph_view = QtGui.QActionGroup(self)
        action_group_graph_view.addAction(self.action_graph_overlay)
        action_group_graph_view.addAction(self.action_graph_stacked)
        self.graph_toolbar.addActions(action_group_graph_view.actions())
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding
        )
        self.graph_toolbar.addWidget(spacer)
        self.graph_toolbar.addAction(self.action_graph_zoomout)

        # Layouts

        self.io.layout_top.insertWidget(0, self.button_file, 0, QtCore.Qt.AlignLeft)
        self.io.layout_top.insertWidget(1, self.label_file, 1, QtCore.Qt.AlignLeft)

        layout_graph = QtWidgets.QHBoxLayout()
        layout_graph.addWidget(self.graph_toolbar, 0)
        layout_graph.addWidget(self.graph, 1)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.io, 0)
        layout.addLayout(layout_graph, 1)

        self.setLayout(layout)

    def setDrawMode(self, mode: str) -> None:
        self.draw_mode = mode
        self.drawGraph()
        self.drawDetections()
        self.drawLimits()

    def setColorScheme(self, scheme: str) -> None:
        self.color_scheme = scheme
        self.drawGraph()
        self.drawDetections()
        self.drawLimits()

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if (
            event.mimeData().hasHtml()
            or event.mimeData().hasText()
            or event.mimeData().hasUrls()
        ):
            event.acceptProposedAction()
        else:  # pragma: no cover
            super().dragEnterEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                self.dialogLoadFile(url.toLocalFile())
                break
            event.acceptProposedAction()
        elif event.mimeData().hasHtml():
            pass
        else:
            super().dropEvent(event)

    def dialogLoadFile(self, file: str | None = None) -> None:
        if file is None:
            file, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Open",
                "",
                "CSV Documents(*.csv *.txt *.text);;All files(*)",
            )
        if file == "" or file is None:
            return

        dlg = ImportDialog(file, self)
        dlg.dataImported.connect(self.loadData)
        dlg.open()

    def loadData(self, data: np.ndarray, options: dict) -> None:
        # Load any values that need to be set from the import dialog inputs
        self.import_options = options
        self.options.dwelltime.setBaseValue(options["dwelltime"])
        self.label_file.setText(str(options["path"]))

        self.responses = data
        self.events = np.arange(data.size)

        self.io.repopulate(data.dtype.names)
        # Update graph, limits and detections
        self.last_region = None
        self.drawGraph()
        self.updateLimits()

        self.dataImported.emit()

    def saveTrimRegion(self) -> None:
        plot = next(iter(self.graph.plots.values()))
        self.last_region = plot.region_start, plot.region_end

    def trimRegion(self, name: str) -> Tuple[int, int]:
        if self.draw_mode == "Overlay":
            plot = self.graph.plots["Overlay"]
        else:
            plot = self.graph.plots[name]
        return plot.region_start, plot.region_end

    def updateDetections(self) -> None:
        names = self.responses.dtype.names

        detections = {}
        labels = {}
        regions = {}
        for name in names:
            trim = self.trimRegion(name)
            responses = self.responses[name][trim[0] : trim[1]]
            if responses.size > 0 and name in self.limits:
                limits = self.limits[name][2]
                (
                    detections[name],
                    labels[name],
                    regions[name],
                ) = spcal.accumulate_detections(responses, limits["lc"], limits["ld"])

        self.detections, self.labels, self.regions = combine_detections(
            detections, labels, regions
        )

        self.detectionsChanged.emit()

    def updateLimits(self) -> None:
        if self.responses.size == 0:
            return

        method = self.options.method.currentText()
        sigma = (
            float(self.options.sigma.text())
            if self.options.sigma.hasAcceptableInput()
            else 3.0
        )
        alpha = (
            float(self.options.error_rate_alpha.text())
            if self.options.error_rate_alpha.hasAcceptableInput()
            else 0.05
        )
        beta = (
            float(self.options.error_rate_beta.text())
            if self.options.error_rate_beta.hasAcceptableInput()
            else 0.05
        )
        window_size = (
            int(self.options.window_size.text())
            if self.options.window_size.hasAcceptableInput()
            and self.options.window_size.isEnabled()
            else None
        )

        names = self.responses.dtype.names
        for name in names:
            trim = self.trimRegion(name)
            response = self.responses[name][trim[0] : trim[1]]
            if response.size == 0:
                self.limits.pop(name)
                continue

            if method == "Manual Input":
                limit = float(self.options.manual.text())
                self.limits[name] = (
                    method,
                    {},
                    np.array(
                        [(np.mean(response), limit, limit)],
                        dtype=calculate_limits.dtype,
                    ),
                )
            else:
                self.limits[name] = calculate_limits(
                    response, method, sigma, (alpha, beta), window=window_size
                )
        self.limitsChanged.emit()

    def updateOutputs(self) -> None:
        names = self.responses.dtype.names

        for name in names:
            io = self.io[name]
            if name not in self.detections.dtype.names:
                io.clearOutputs()
            else:
                trim = self.trimRegion(name)
                io.updateOutputs(
                    self.responses[name][trim[0] : trim[1]],
                    self.detections[name],
                    self.labels,
                    self.limits[name],
                )

    def drawGraph(self) -> None:
        self.graph.clear()
        if len(self.responses) == 0:
            return

        dwell = self.options.dwelltime.baseValue()
        if dwell is None:
            raise ValueError("dwell is None")

        if self.draw_mode == "Overlay":
            plot = self.graph.addParticlePlot("Overlay", xscale=dwell)

        scheme = color_schemes[self.color_scheme]
        for i, name in enumerate(self.responses.dtype.names):
            ys = self.responses[name]
            if self.draw_mode == "Stacked":
                plot = self.graph.addParticlePlot(name, xscale=dwell)
                self.graph.layout.nextRow()
            elif self.draw_mode == "Overlay":
                pass
            else:
                raise ValueError("drawGraph: draw_mode must be 'Stacked', 'Overlay'.")

            pen = QtGui.QPen(scheme[i % len(scheme)], 1.0)
            pen.setCosmetic(True)
            plot.drawSignal(self.events, ys, label=name, pen=pen)

        region = (
            (self.events[0], self.events[-1])
            if self.last_region is None
            else self.last_region
        )
        for plot in self.graph.plots.values():
            plot.region.blockSignals(True)
            plot.region.setRegion(region)
            plot.region.blockSignals(False)
        self.last_region = region

    def drawDetections(self) -> None:
        scheme = color_schemes[self.color_scheme]

        for plot in self.graph.plots.values():
            plot.clearScatters()

        if self.detections.dtype.names is None or self.detections.size == 0:
            return

        for i, name in enumerate(self.detections.dtype.names):
            color = scheme[i % len(scheme)]
            symbol = symbols[i % len(symbols)]

            if self.draw_mode == "Overlay":
                plot = self.graph.plots["Overlay"]
            else:
                plot = self.graph.plots[name]

            detected = np.flatnonzero(self.detections[name])

            if detected.size > 0:
                trim = self.trimRegion(name)
                maxima = (
                    detection_maxima(
                        self.responses[name][trim[0] : trim[1]],
                        self.regions[detected],
                    )
                    + trim[0]
                )
                plot.drawMaxima(
                    self.events[maxima],
                    self.responses[name][maxima],
                    brush=QtGui.QBrush(color),
                    symbol=symbol,
                )

    def drawLimits(self) -> None:
        if self.draw_mode == "Overlay":
            return
        for name in self.limits:
            trim = self.trimRegion(name)
            plot = self.graph.plots[name]
            plot.clearLimits()
            plot.drawLimits(self.events[trim[0] : trim[1]], self.limits[name][2])

    def resetInputs(self) -> None:
        self.blockSignals(True)
        for i in range(self.io.stack.count()):
            self.io.stack.widget(i).clearInputs()
        self.blockSignals(False)

    def onEfficiencyMethodChanged(self, method: str) -> None:
        for io in self.io.widgets():
            io.response.setEnabled(method != "Mass Response")

    def isComplete(self) -> bool:
        return (
            self.detections.size > 0
            and self.detections.dtype.names is not None
            and len(self.detections.dtype.names) > 0
        )


class SampleWidget(InputWidget):
    def __init__(self, options: OptionsWidget, parent: QtWidgets.QWidget | None = None):
        super().__init__(SampleIOStack(), options, parent=parent)
        assert isinstance(self.io, SampleIOStack)


class ReferenceWidget(InputWidget):
    def __init__(self, options: OptionsWidget, parent: QtWidgets.QWidget | None = None):
        super().__init__(ReferenceIOStack(), options, parent=parent)

        # dwelltime covered by detectionsChanged
        self.options.uptake.valueChanged.connect(lambda: self.updateEfficiency(None))
        self.io.optionsChanged.connect(self.updateEfficiency)
        self.detectionsChanged.connect(self.updateEfficiency)

    def updateEfficiency(self, _name: str | None = None) -> None:
        if self.responses.size == 0:
            return
        if _name is None or _name == "Overlay":
            names = self.responses.dtype.names
        else:
            names = [_name]

        dwell = self.options.dwelltime.baseValue()
        assert dwell is not None
        time = self.events.size * dwell
        uptake = self.options.uptake.baseValue()

        for name in names:
            self.io[name].updateEfficiency(self.detections[name], dwell, time, uptake)
