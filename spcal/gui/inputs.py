from PySide6 import QtCore, QtGui, QtWidgets

import numpy as np
import logging

import spcal

from spcal.calc import calculate_limits
from spcal.detection import detection_maxima

from spcal.gui.dialogs import ImportDialog
from spcal.gui.iowidgets import IOStack, SampleIOStack, ReferenceIOStack
from spcal.gui.graphs import ParticleView, graph_colors
from spcal.gui.options import OptionsWidget
from spcal.gui.util import create_action
from spcal.gui.widgets import ElidedLabel

from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class InputWidget(QtWidgets.QWidget):
    optionsChanged = QtCore.Signal()
    detectionsChanged = QtCore.Signal(str)
    limitsChanged = QtCore.Signal(str)

    def __init__(
        self,
        io_stack: IOStack,
        options: OptionsWidget,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.setAcceptDrops(True)

        self.graph_toolbar = QtWidgets.QToolBar()
        self.graph_toolbar.setOrientation(QtCore.Qt.Vertical)
        self.graph_toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)

        self.graph = ParticleView()
        self.graph.regionChanged.connect(self.updateLimits)

        self.io = io_stack

        self.redraw_graph_requested = False
        self.draw_mode = "Overlay"

        self.limitsChanged.connect(self.updateDetections)
        self.limitsChanged.connect(self.drawLimits)

        self.detectionsChanged.connect(self.updateOutputs)
        self.detectionsChanged.connect(self.drawDetections)

        self.options = options
        self.options.dwelltime.valueChanged.connect(lambda: self.updateLimits(None))
        self.options.method.currentTextChanged.connect(lambda: self.updateLimits(None))
        self.options.window_size.editingFinished.connect(
            lambda: self.updateLimits(None)
        )
        self.options.check_use_window.toggled.connect(lambda: self.updateLimits(None))
        self.options.sigma.editingFinished.connect(lambda: self.updateLimits(None))
        self.options.manual.editingFinished.connect(lambda: self.updateLimits(None))
        self.options.error_rate_alpha.editingFinished.connect(
            lambda: self.updateLimits(None)
        )
        self.options.error_rate_beta.editingFinished.connect(
            lambda: self.updateLimits(None)
        )
        self.options.efficiency_method.currentTextChanged.connect(self.onEfficiencyMethodChanged)

        self.responses = np.array([])
        self.events = np.array([])
        self.detections: Dict[str, np.ndarray] = {}
        self.labels: Dict[str, np.ndarray] = {}
        self.regions: Dict[str, np.ndarray] = {}
        self.limits: Dict[str, Tuple[str, Dict[str, float], np.ndarray]] = {}

        self.button_file = QtWidgets.QPushButton("Open File")
        self.button_file.pressed.connect(self.dialogLoadFile)

        self.label_file = ElidedLabel()

        # Actions

        self.action_graph_overlay = create_action(
            "user-desktop",
            "Histogram",
            "Overlay all signals.",
            lambda: self.setDrawMode("Overlay"),
            checkable=True,
        )
        self.action_graph_overlay.setChecked(True)
        self.action_graph_stacked = create_action(
            "object-rows",
            "Stacked",
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
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
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

    def dialogLoadFile(self, file: Optional[str] = None) -> None:
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
        dlg.dwelltimeImported.connect(self.options.dwelltime.setBaseValue)
        dlg.accepted.connect(lambda: self.label_file.setText(dlg.file_path.name))
        dlg.open()

    def loadData(self, data: np.ndarray) -> None:
        self.responses = data
        self.events = np.arange(data.size)

        self.io.repopulate(data.dtype.names)
        # Update graph, limits and detections
        self.drawGraph()
        self.updateLimits()

    def trimRegion(self, name: str) -> Tuple[int, int]:
        if self.draw_mode == "Overlay":
            plot = self.graph.plots["Overlay"]
        else:
            plot = self.graph.plots[name]
        return plot.region_start, plot.region_end

    def updateDetections(self, _name: Optional[str] = None) -> None:
        if _name is None or _name == "Overlay":
            names = self.responses.dtype.names
        else:
            names = [_name]

        for name in names:
            trim = self.trimRegion(name)
            responses = self.responses[name][trim[0] : trim[1]]
            if responses.size > 0 and name in self.limits:
                limits = self.limits[name][2]
                (
                    self.detections[name],
                    self.labels[name],
                    self.regions[name],
                ) = spcal.accumulate_detections(responses, limits["lc"], limits["ld"])
            else:
                self.detections.pop(name)
                self.labels.pop(name)
                self.regions.pop(name)

            self.detectionsChanged.emit(name)

    def updateLimits(self, _name: Optional[str] = None) -> None:
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

        if _name is None or _name == "Overlay":
            names = self.responses.dtype.names
        else:
            names = [_name]

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
            self.limitsChanged.emit(name)

    def updateOutputs(self, _name: Optional[str] = None) -> None:
        if _name is None or _name == "Overlay":
            names = self.responses.dtype.names
        else:
            names = [_name]

        for name in names:
            io = self.io[name]
            if name not in self.detections:
                io.clearOutputs()
            else:
                trim = self.trimRegion(name)
                io.updateOutputs(
                    self.responses[name][trim[0] : trim[1]],
                    self.detections[name],
                    self.labels[name],
                    self.limits[name],
                )

    def drawGraph(self) -> None:
        self.graph.clear()
        if len(self.responses) == 0:
            return

        dwell = self.options.dwelltime.baseValue()
        if dwell is None:
            raise ValueError("dwell is None")

        if self.draw_mode == "Stacked":
            for name in self.responses.dtype.names:
                ys = self.responses[name]

                plot = self.graph.addParticlePlot(name, xscale=dwell)
                self.graph.layout.nextRow()
                plot.drawSignal(self.events, self.responses[name])
        elif self.draw_mode == "Overlay":
            plot = self.graph.addParticlePlot("Overlay", xscale=dwell)
            for name, color in zip(self.responses.dtype.names, graph_colors):
                ys = self.responses[name]

                pen = QtGui.QPen(color, 1.0)
                pen.setCosmetic(True)
                plot.drawSignal(self.events, ys, label=name, pen=pen)
        else:
            raise ValueError("drawGraph: draw_mode must be 'Stacked', 'Overlay'.")

    def drawDetections(self, name: str) -> None:
        if self.draw_mode == "Overlay":
            plot = self.graph.plots["Overlay"]
            name_idx = list(self.responses.dtype.names).index(name)
            color = graph_colors[name_idx]
            if name_idx == 0:
                plot.clearScatters()
        else:
            plot = self.graph.plots[name]
            color = QtCore.Qt.red
            plot.clearScatters()

        if name in self.regions and self.regions[name].size > 0:
            maxima = detection_maxima(
                self.responses[name], self.regions[name] + self.trimRegion(name)[0]
            )
            plot.drawMaxima(
                self.events[maxima],
                self.responses[name][maxima],
                brush=QtGui.QBrush(color),
            )

    def drawLimits(self, name: str) -> None:
        if self.draw_mode == "Overlay":
            return
        plot = self.graph.plots[name]
        plot.clearLimits()

        if name in self.limits:
            plot.drawLimits(self.events, self.limits[name][2])

    def resetInputs(self) -> None:
        self.blockSignals(True)
        for i in range(self.io.stack.count()):
            self.io.stack.widget(i).clearInputs()
        self.blockSignals(False)

    def onEfficiencyMethodChanged(self, method: str) -> None:
        for io in self.io.widgets():
            io.response.setEnabled(method != "Mass Response")

    def isComplete(self) -> bool:
        return len(self.detections) > 0 and any(
            self.detections[name].size > 0 for name in self.detections
        )


class SampleWidget(InputWidget):
    def __init__(
        self, options: OptionsWidget, parent: Optional[QtWidgets.QWidget] = None
    ):
        super().__init__(SampleIOStack(), options, parent=parent)
        assert isinstance(self.io, SampleIOStack)


class ReferenceWidget(InputWidget):
    def __init__(
        self, options: OptionsWidget, parent: Optional[QtWidgets.QWidget] = None
    ):
        super().__init__(ReferenceIOStack(), options, parent=parent)

        # dwelltime covered by detectionsChanged
        self.options.uptake.valueChanged.connect(lambda: self.updateEfficiency(None))
        self.io.optionsChanged.connect(self.updateEfficiency)
        self.detectionsChanged.connect(self.updateEfficiency)

    def updateEfficiency(self, _name: Optional[str] = None) -> None:
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
            self.io[name].updateEfficiency(
                self.detections[name], dwell, time, uptake
            )
