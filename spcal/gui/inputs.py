import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

import spcal
from spcal.detection import combine_detections, detection_maxima
from spcal.gui.dialogs._import import _ImportDialogBase
from spcal.gui.graphs import color_schemes, symbols
from spcal.gui.graphs.particle import ParticleView
from spcal.gui.io import get_import_dialog_for_path, get_open_spcal_path, is_spcal_path
from spcal.gui.iowidgets import IOStack, ReferenceIOStack, SampleIOStack
from spcal.gui.options import OptionsWidget
from spcal.gui.util import create_action
from spcal.gui.widgets import ElidedLabel
from spcal.limit import SPCalLimit
from spcal.result import SPCalResult

logger = logging.getLogger(__name__)


class InputWidget(QtWidgets.QWidget):
    optionsChanged = QtCore.Signal()
    detectionsChanged = QtCore.Signal()
    limitsChanged = QtCore.Signal()

    dataLoaded = QtCore.Signal(Path)

    def __init__(
        self,
        io_stack: IOStack,
        options: OptionsWidget,
        color_scheme: str = "IBM Carbon",
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setAcceptDrops(True)

        self.import_options: dict = {}

        self.responses = np.array([])
        self.events = np.array([])
        self.detections = np.array([])
        self.labels = np.array([])
        self.regions = np.array([])
        self.limits: Dict[str, SPCalLimit] = {}

        self.draw_mode = "overlay"

        self.graph_toolbar = QtWidgets.QToolBar()
        self.graph_toolbar.setOrientation(QtCore.Qt.Vertical)
        self.graph_toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)

        self.graph = ParticleView()
        self.graph.regionChanged.connect(self.saveTrimRegion)
        self.graph.regionChanged.connect(self.updateLimits)
        self.last_region: Tuple[int, int] | None = None

        self.io = io_stack
        self.io.nameChanged.connect(self.updateGraphsForName)
        self.io.limitsChanged.connect(self.updateLimits)

        self.limitsChanged.connect(self.updateDetections)
        self.limitsChanged.connect(self.drawLimits)

        self.detectionsChanged.connect(self.updateOutputs)
        self.detectionsChanged.connect(self.drawDetections)

        self.options = options
        # self.options.dwelltime.valueChanged.connect(self.updateLimits)
        self.options.limitOptionsChanged.connect(self.updateLimits)
        self.options.efficiency_method.currentTextChanged.connect(
            self.onEfficiencyMethodChanged
        )

        self.button_file = QtWidgets.QPushButton("Open File")
        self.button_file.pressed.connect(self.dialogLoadFile)

        self.label_file = ElidedLabel()

        # Actions

        self.action_graph_overlay = create_action(
            "labplot-xy-curve",
            "Overlay Signals",
            "Overlay all element signals in a single plot.",
            lambda: self.setDrawMode("overlay"),
            checkable=True,
        )
        self.action_graph_overlay.setChecked(True)
        self.action_graph_single = create_action(
            "labplot-xy-curve-segments",
            "Individual Signals",
            "Draw the current element signal.",
            lambda: self.setDrawMode("single"),
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
        action_group_graph_view.addAction(self.action_graph_single)
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

    @property
    def names(self) -> Tuple[str, ...]:
        if self.responses.dtype.names is None:
            return tuple()
        else:
            return self.responses.dtype.names

    @property
    def detection_names(self) -> Tuple[str, ...]:
        if self.detections.dtype.names is None:
            return tuple()
        else:
            return self.detections.dtype.names

    @property
    def draw_names(self) -> Tuple[str, ...]:
        if self.draw_mode == "single":
            name = self.io.combo_name.currentText()
            if name != "<element>":
                return (name,)
            else:
                return tuple()
        return self.names

    def colorForName(self, name: str) -> QtGui.QColor:
        scheme = color_schemes[QtCore.QSettings().value("colorscheme", "IBM Carbon")]
        return QtGui.QColor(scheme[self.names.index(name) % len(scheme)])

    def setDrawMode(self, mode: str) -> None:
        self.draw_mode = mode
        self.redraw()

    def redraw(self) -> None:
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
            # Todo, nu import check
            for url in event.mimeData().urls():
                path = Path(url.toLocalFile())
                if is_spcal_path(path):
                    self.dialogLoadFile(path)
                    break
            event.acceptProposedAction()
        elif event.mimeData().hasHtml():
            pass
        else:
            super().dropEvent(event)

    def dialogLoadFile(
        self, path: str | Path | None = None
    ) -> _ImportDialogBase | None:
        if path is None:
            path = get_open_spcal_path(self)
            if path is None:
                return None
        else:
            path = Path(path)

        dlg = get_import_dialog_for_path(self, path, self.import_options)
        dlg.dataImported.connect(self.loadData)
        dlg.open()
        return dlg

    def loadData(self, data: np.ndarray, options: dict) -> None:
        # Load any values that need to be set from the import dialog inputs
        self.import_options = options
        self.options.dwelltime.setBaseValue(options["dwelltime"])
        self.label_file.setText(str(options["path"]))

        self.responses = data
        self.events = np.arange(data.size)

        self.io.repopulate(data.dtype.names)

        if len(data.dtype.names) < 2:
            self.action_graph_overlay.setEnabled(False)
            self.action_graph_single.setChecked(True)
            self.draw_mode = "single"
        else:
            self.action_graph_overlay.setEnabled(True)
        # Update graph, limits and detections
        self.last_region = None

        self.drawGraph()
        self.updateLimits()

        # New widgets, set editable
        if self.options.limit_method.currentText() == "Manual Input":
            self.io.setLimitsEditable(True)
        self.graph.action_export_data.setVisible(True)

        self.dataLoaded.emit(options["path"])

    def saveTrimRegion(self) -> None:
        # plot = next(iter(self.graph.plots.values()))
        self.last_region = self.graph.region_start, self.graph.region_end

    def trimRegion(self, name: str) -> Tuple[int, int]:
        return self.graph.region_start, self.graph.region_end

    def trimmedResponse(self, name: str) -> np.ndarray:
        trim = self.trimRegion(name)
        return self.responses[name][trim[0] : trim[1]]

    def asResult(self, name: str) -> SPCalResult:
        return SPCalResult(
            self.label_file.text(),
            self.trimmedResponse(name),
            self.detections[name],
            self.labels,
            self.limits[name],
        )

    def updateDetections(self) -> None:
        d, l, r = {}, {}, {}
        for name in self.names:
            responses = self.trimmedResponse(name)
            if responses.size > 0 and name in self.limits:
                (d[name], l[name], r[name],) = spcal.accumulate_detections(
                    responses,
                    np.minimum(
                        self.limits[name].mean_background,
                        self.limits[name].detection_threshold,
                    ),
                    self.limits[name].detection_threshold,
                    integrate=True,
                )

        self.detections, self.labels, self.regions = combine_detections(d, l, r)

        self.detectionsChanged.emit()

    def updateLimits(self) -> None:
        if self.responses.size == 0:
            return

        method = self.options.limit_method.currentText()
        poisson_alpha = self.options.error_rate_poisson.value() or 0.001
        gaussian_alpha = self.options.error_rate_gaussian.value() or 1e-6
        if self.options.window_size.isEnabled():
            window_size = self.options.window_size.value() or 0
        else:
            window_size = 0
        window_size = int(window_size or 0)
        max_iter = 10 if self.options.check_iterative.isChecked() else 1

        self.limits.clear()

        for name in self.names:
            response = self.trimmedResponse(name)
            if response.size == 0:
                continue

            if method == "Manual Input":
                limit = self.io[name].lod_count.value()
                if limit is not None:
                    self.limits[name] = SPCalLimit(
                        np.mean(response),
                        limit,
                        name="Manual Input",
                        params={},
                    )
                else:  # If empty limit then fill with best estimate
                    self.limits[name] = SPCalLimit.fromBest(
                        response,
                        poisson_alpha=poisson_alpha,
                        gaussian_alpha=gaussian_alpha,
                        window_size=window_size,
                        max_iters=max_iter,
                    )
            else:
                self.limits[name] = SPCalLimit.fromMethodString(
                    method,
                    response,
                    poisson_alpha=poisson_alpha,
                    gaussian_alpha=gaussian_alpha,
                    window_size=window_size,
                    max_iters=max_iter,
                )
        self.limitsChanged.emit()

    def updateOutputs(self) -> None:
        for name in self.names:
            io = self.io[name]
            if name not in self.detection_names:
                io.clearOutputs()
            else:
                io.updateOutputs(
                    self.trimmedResponse(name),
                    self.detections[name],
                    self.labels,
                    np.mean(self.limits[name].detection_threshold),
                    self.limits[name].name,
                    self.limits[name].params,
                )

    def updateFormat(self) -> None:
        for io in self.io:
            io.updateFormat()

    def updateGraphsForName(self, name: str) -> None:
        if self.draw_mode == "single":
            self.redraw()

    def drawGraph(self) -> None:
        self.graph.clear()
        if len(self.responses) == 0:
            return

        dwell = self.options.dwelltime.baseValue()
        if dwell is None:
            raise ValueError("dwell is None")
        self.graph.xaxis.setScale(dwell)

        for i, name in enumerate(self.draw_names):
            ys = self.responses[name]
            pen = QtGui.QPen(self.colorForName(name), 1.0)
            pen.setCosmetic(True)
            self.graph.drawSignal(name, self.events, ys, pen=pen)

        self.graph.region.setBounds((self.events[0], self.events[-1]))
        self.graph.setDataLimits(xMin=0.0, xMax=1.0, yMax=1.05)

        region = (
            (self.events[0], self.events[-1])
            if self.last_region is None
            else self.last_region
        )
        self.graph.region.blockSignals(True)
        self.graph.region.setRegion(region)
        self.graph.region.blockSignals(False)
        self.last_region = region

    def drawDetections(self) -> None:
        self.graph.clearScatters()
        if self.detections.size == 0:
            return

        for i, name in enumerate(self.draw_names):
            color = self.colorForName(name)
            symbol = symbols[i % len(symbols)]

            detected = np.flatnonzero(self.detections[name])

            if detected.size > 0:
                trim = self.trimRegion(name)
                maxima = (
                    detection_maxima(
                        self.trimmedResponse(name),
                        self.regions[detected],
                    )
                    + trim[0]
                )
                self.graph.drawMaxima(
                    name,
                    self.events[maxima],
                    self.responses[name][maxima],
                    brush=QtGui.QBrush(color),
                    symbol=symbol,
                )
        self.graph.zoomReset()

    def drawLimits(self) -> None:
        if self.draw_mode != "single":
            return

        self.graph.clearLimits()

        for name in self.draw_names:
            pen = QtGui.QPen(self.colorForName(name).darker(), 1.0)
            pen.setCosmetic(True)

            trim = self.trimRegion(name)

            self.graph.drawLimits(
                self.events[trim[0] : trim[1]],
                self.limits[name].mean_background,
                self.limits[name].detection_threshold,
                pen=pen,
            )

    def resetInputs(self) -> None:
        self.blockSignals(True)
        self.graph.clear()
        for i in range(self.io.stack.count()):
            self.io.stack.widget(i).clearInputs()
            self.io.stack.widget(i).clearOutputs()
        self.blockSignals(False)

    def onEfficiencyMethodChanged(self, method: str) -> None:
        for io in self.io.widgets():
            io.response.setEnabled(method != "Mass Response")

    def isComplete(self) -> bool:
        return self.detections.size > 0 and len(self.detection_names) > 0


class SampleWidget(InputWidget):
    def __init__(self, options: OptionsWidget, parent: QtWidgets.QWidget | None = None):
        super().__init__(SampleIOStack(), options, parent=parent)
        assert isinstance(self.io, SampleIOStack)


class ReferenceWidget(InputWidget):
    def __init__(self, options: OptionsWidget, parent: QtWidgets.QWidget | None = None):
        super().__init__(ReferenceIOStack(), options, parent=parent)

        # dwelltime covered by detectionsChanged
        self.options.uptake.baseValueChanged.connect(
            lambda: self.updateEfficiency(None)
        )
        self.io.optionsChanged.connect(self.updateEfficiency)
        self.detectionsChanged.connect(self.updateEfficiency)
        self.dataLoaded.connect(self.onDataLoaded)

    def onDataLoaded(self) -> None:
        if len(self.names) == 1:  # Default to use if only element
            self.io[self.names[0]].check_use_efficiency_for_all.setChecked(True)

    def updateEfficiency(self, name: str | None = None) -> None:
        dwell = self.options.dwelltime.baseValue()
        if self.responses.size == 0 or dwell is None:
            return

        if name is None:
            names = self.names
        else:
            names = (name,)

        time = self.events.size * dwell
        uptake = self.options.uptake.baseValue()
        for _name in names:
            self.io[_name].updateEfficiency(self.detections[_name], dwell, time, uptake)

    def getEfficiency(self, name: str) -> float | None:
        use_all = None
        for _name in self.io.names():
            if (
                self.io[_name].check_use_efficiency_for_all.checkState()
                == QtCore.Qt.CheckState.Checked
            ):
                use_all = _name
                break

        if use_all is not None:
            efficiency = self.io[use_all].efficiency.value()
        elif name in self.io:
            efficiency = self.io[name].efficiency.value()
        else:
            return None

        return efficiency
