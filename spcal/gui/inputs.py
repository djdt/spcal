import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

import spcal
from spcal.detection import combine_detections, detection_maxima
from spcal.gui.dialogs._import import ImportDialog, NuImportDialog
from spcal.gui.graphs import color_schemes, symbols
from spcal.gui.graphs.views import ParticleView
from spcal.gui.iowidgets import IOStack, ReferenceIOStack, SampleIOStack
from spcal.gui.options import OptionsWidget
from spcal.gui.util import create_action
from spcal.gui.widgets import ElidedLabel
from spcal.io.nu import is_nu_directory
from spcal.limit import SPCalLimit
from spcal.result import SPCalResult

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

        self.import_options: dict = {}

        self.responses = np.array([])
        self.events = np.array([])
        self.detections = np.array([])
        self.labels = np.array([])
        self.regions = np.array([])
        self.limits: Dict[str, SPCalLimit] = {}

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
            # Todo, nu import check
            for url in event.mimeData().urls():
                self.dialogLoadFile(url.toLocalFile())
                break
            event.acceptProposedAction()
        elif event.mimeData().hasHtml():
            pass
        else:
            super().dropEvent(event)

    def dialogLoadFile(
        self, path: str | Path | None = None
    ) -> ImportDialog | NuImportDialog | None:
        if path is None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Open",
                "",
                (
                    "NP Data Files (*.csv *.info);;CSV Documents(*.csv *.txt *.text);;"
                    "Nu Instruments(*.info);;All files(*)"
                ),
            )
            if path == "":
                return None

        path = Path(path)

        if path.suffix == ".info":
            if not is_nu_directory(path.parent):
                raise FileNotFoundError("dialogLoadFile: invalid Nu directory.")
            dlg = NuImportDialog(path.parent, self)
        else:
            dlg = ImportDialog(path, self)
        dlg.dataImported.connect(self.loadData)
        dlg.open()
        return dlg

    # def dialogLoadNuDirectory(self, dir: str | None = None) -> NuImportDialog | None:
    #     if dir is None:
    #         dir, _ = QtWidgets.QFileDialog.getExistingDirectory(self, "Open", "")
    #     if dir == "" or dir is None:
    #         return None

    #     dlg = NuImportDialog(dir, self)
    #     dlg.dataImported.connect(self.loadData)
    #     dlg.open()
    #     return dlg

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

    def asResult(self, name: str) -> SPCalResult:
        trim = self.trimRegion(name)
        return SPCalResult(
            self.label_file.text(),
            self.responses[name][trim[0] : trim[1]],
            self.detections[name],
            self.labels,
            self.limits[name],
        )

    def updateDetections(self) -> None:
        d, l, r = {}, {}, {}
        assert self.responses.dtype.names is not None
        for name in self.responses.dtype.names:
            trim = self.trimRegion(name)
            responses = self.responses[name][trim[0] : trim[1]]
            if responses.size > 0 and name in self.limits:
                (d[name], l[name], r[name],) = spcal.accumulate_detections(
                    responses,
                    self.limits[name].detection_threshold,
                )

        self.detections, self.labels, self.regions = combine_detections(d, l, r)

        self.detectionsChanged.emit()

    def updateLimits(self) -> None:
        if self.responses.size == 0:
            return

        method = self.options.method.currentText()
        poisson_alpha = (
            float(self.options.error_rate_poisson.text())
            if self.options.error_rate_poisson.hasAcceptableInput()
            else 0.001
        )
        gaussian_alpha = (
            float(self.options.error_rate_gaussian.text())
            if self.options.error_rate_gaussian.hasAcceptableInput()
            else 1e-6
        )
        window_size = (
            int(self.options.window_size.text())
            if self.options.window_size.hasAcceptableInput()
            and self.options.window_size.isEnabled()
            else 0
        )
        max_iter = 10 if self.options.check_iterative.isChecked() else 1

        self.limits.clear()

        assert self.responses.dtype.names is not None
        for name in self.responses.dtype.names:
            trim = self.trimRegion(name)
            response = self.responses[name][trim[0] : trim[1]]
            if response.size == 0:
                continue

            if method == "Manual Input":
                limit = float(self.options.manual.text())
                self.limits[name] = SPCalLimit(
                    np.mean(response), limit, name="Manual Input", params={}
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
        assert self.responses.dtype.names is not None
        assert self.detections.dtype.names is not None
        for name in self.responses.dtype.names:
            io = self.io[name]
            if name not in self.detections.dtype.names:
                io.clearOutputs()
            else:
                trim = self.trimRegion(name)
                io.updateOutputs(
                    self.responses[name][trim[0] : trim[1]],
                    self.detections[name],
                    self.labels,
                    np.mean(self.limits[name].detection_threshold),
                    self.limits[name].name,
                    self.limits[name].params,
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
        assert self.responses.dtype.names is not None
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
            plot.drawSignal(name, self.events, ys, pen=pen)

        self.graph.setDataLimits(yMax=1.05)
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
                    name,
                    self.events[maxima],
                    self.responses[name][maxima],
                    brush=QtGui.QBrush(color),
                    symbol=symbol,
                )

    def drawLimits(self) -> None:
        if self.draw_mode == "Overlay":
            return

        for plot in self.graph.plots.values():
            plot.clearLimits()

        for i, (name, limits) in enumerate(self.limits.items()):
            pen = QtGui.QPen(QtCore.Qt.black, 1.0, QtCore.Qt.DashLine)
            pen.setCosmetic(True)

            trim = self.trimRegion(name)

            if self.draw_mode == "Overlay":
                plot = self.graph.plots["Overlay"]
            else:
                plot = self.graph.plots[name]

            plot.drawLimits(
                self.events[trim[0] : trim[1]],
                # limits.mean_background,
                limits.detection_threshold,
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
        dwell = self.options.dwelltime.baseValue()
        if self.responses.size == 0 or dwell is None:
            return
        if _name is None or _name == "Overlay":
            names = self.responses.dtype.names
        else:
            names = (_name,)

        time = self.events.size * dwell
        uptake = self.options.uptake.baseValue()

        assert names is not None
        for name in names:
            self.io[name].updateEfficiency(self.detections[name], dwell, time, uptake)

    def getEfficiency(self, name: str) -> float | None:
        use_all = None
        for name in self.io.names():
            if self.io[name].check_use_efficiency_for_all.isChecked():
                use_all = name
                break

        if use_all is not None:
            efficiency = self.io[use_all].efficiency.text()
        elif name in self.io:
            efficiency = self.io[name].efficiency.text()
        else:
            return None

        try:
            return float(efficiency)
        except ValueError:
            return None
