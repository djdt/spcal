import logging
from pathlib import Path

import numpy as np
import numpy.lib.recfunctions as rfn
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.detection import accumulate_detections, combine_detections, detection_maxima
from spcal.gui.dialogs._import import _ImportDialogBase
from spcal.gui.dialogs.calculator import CalculatorDialog
from spcal.gui.graphs import color_schemes, symbols
from spcal.gui.graphs.draw import draw_particle_view
from spcal.gui.graphs.particle import ParticleView
from spcal.gui.io import get_import_dialog_for_path, get_open_spcal_path, is_spcal_path
from spcal.gui.iowidgets import IOStack, ReferenceIOStack, SampleIOStack
from spcal.gui.options import OptionsWidget
from spcal.gui.util import create_action
from spcal.gui.widgets import ElidedLabel
from spcal.limit import SPCalLimit
from spcal.pratt import Reducer, ReducerException
from spcal.result import SPCalResult

logger = logging.getLogger(__name__)


class InputWidget(QtWidgets.QWidget):
    optionsChanged = QtCore.Signal()
    detectionsChanged = QtCore.Signal()
    limitsChanged = QtCore.Signal()

    dataLoaded = QtCore.Signal(Path)
    namesEdited = QtCore.Signal(dict)

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
        self.current_expr: dict[str, str] = {}

        self.responses = np.array([])
        self.events = np.array([])
        self.detections = np.array([])
        self.labels = np.array([])
        self.regions = np.array([])
        self.original_regions: dict[str, np.ndarray] = {}  # regions before combination
        self.limits: dict[str, SPCalLimit] = {}

        self.draw_mode = "overlay"

        self.graph_toolbar = QtWidgets.QToolBar()
        self.graph_toolbar.setOrientation(QtCore.Qt.Orientation.Vertical)
        self.graph_toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)

        settings = QtCore.QSettings()
        font = QtGui.QFont(
            str(settings.value("GraphFont/Family", "SansSerif")),
            pointSize=int(settings.value("GraphFont/PointSize", 10)),
        )
        self.graph = ParticleView(font=font)
        self.graph.regionChanged.connect(self.saveTrimRegion)
        self.graph.regionChanged.connect(self.updateLimits)
        self.graph.requestPeakProperties.connect(self.dialogDataProperties)
        self.graph.requestImageExport.connect(self.dialogExportGraphImage)
        self.last_region: tuple[int, int] | None = None

        self.io = io_stack
        self.io.nameChanged.connect(self.updateGraphsForName)
        self.io.namesEdited.connect(self.updateNames)
        self.io.enabledNamesChanged.connect(self.redraw)
        self.io.enabledNamesChanged.connect(
            self.detectionsChanged
        )  # force results update
        self.io.namesEdited.connect(self.namesEdited)  # re-emit
        self.io.limitsChanged.connect(self.updateLimits)

        self.io.optionsChanged.connect(self.optionsChanged)

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
            QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Expanding
        )
        self.graph_toolbar.addWidget(spacer)
        self.graph_toolbar.addAction(self.action_graph_zoomout)

        # Layouts

        self.io.layout_top.insertWidget(0, self.button_file, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        self.io.layout_top.insertWidget(1, self.label_file, 1, QtCore.Qt.AlignmentFlag.AlignLeft)

        layout_graph = QtWidgets.QHBoxLayout()
        layout_graph.addWidget(self.graph_toolbar, 0)
        layout_graph.addWidget(self.graph, 1)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.io, 0)
        layout.addLayout(layout_graph, 1)

        self.setLayout(layout)

    @property
    def names(self) -> tuple[str, ...]:
        if self.responses.dtype.names is None:
            return tuple()
        else:
            return self.responses.dtype.names

    @property
    def detection_names(self) -> tuple[str, ...]:
        if self.detections.dtype.names is None:
            return tuple()
        else:
            return self.detections.dtype.names

    @property
    def enabled_names(self) -> list[str]:
        return self.io.enabledNames()

    @property
    def draw_names(self) -> list[str]:
        if self.draw_mode == "single":
            name = self.io.combo_name.currentText()
            if name != "<element>":
                return [name]
            else:
                return []
        return self.enabled_names

    def colorForName(self, name: str) -> QtGui.QColor:
        scheme = color_schemes[str(QtCore.QSettings().value("colorscheme", "IBM Carbon"))]
        return QtGui.QColor(scheme[self.names.index(name) % len(scheme)])

    def updateNames(self, names: dict[str, str]) -> None:
        if self.responses.dtype.names is not None:
            self.responses = rfn.rename_fields(self.responses, names)
        if self.detections.dtype.names is not None:
            self.detections = rfn.rename_fields(self.detections, names)

        for old, new in names.items():
            if old == new:
                continue
            if old in self.limits:
                self.limits[new] = self.limits.pop(old)
            if old in self.current_expr:
                self.current_expr[new] = self.current_expr.pop(old)
            if old in self.io:
                index = self.io.combo_name.findText(old)
                self.io.combo_name.setItemText(index, new)
            if "names" in self.import_options:
                if old in self.import_options["names"].values():
                    key = next(
                        k for k, v in self.import_options["names"].items() if v == old
                    )
                    self.import_options["names"][key] = new

        self.redraw()

    def setDrawMode(self, mode: str) -> None:
        self.draw_mode = mode
        self.redraw()

    def setGraphFont(self, font: QtGui.QFont) -> None:
        self.graph.setFont(font)
        self.redraw()  # fixes legend

    def redraw(self, save_range: bool = True) -> None:
        (xmin, xmax), (ymin, ymax) = self.graph.plot.viewRange()
        self.drawGraph()
        self.drawDetections()
        self.drawLimits()
        if save_range:
            if self.graph.plot.vb.state["autoVisibleOnly"][1]:
                self.graph.plot.vb.setRange(xRange=(xmin, xmax), padding=0.0)
            else:
                self.graph.plot.vb.setRange(
                    xRange=(xmin, xmax), yRange=(ymin, ymax), padding=0.0
                )

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

        dlg = get_import_dialog_for_path(
            self,
            path,
            self.import_options,
            screening_options={
                "poisson_kws": dict(self.options.poisson.state()),
                "gaussian_kws": dict(self.options.gaussian.state()),
                "compound_kws": dict(self.options.compound_poisson.state()),
            },
        )
        dlg.dataImported.connect(self.loadData)
        dlg.open()
        return dlg

    def dialogDataProperties(self) -> QtWidgets.QDialog | None:
        from spcal.gui.dialogs.peakproperties import PeakPropertiesDialog

        if len(self.detection_names) == 0:
            return

        dlg = PeakPropertiesDialog(self, self.io.combo_name.currentText())
        dlg.exec()

    def dialogExportGraphImage(self) -> None:
        from spcal.gui.dialogs.imageexport import ImageExportDialog

        def get_path_and_export(size: QtCore.QSize, dpi: int, options: dict) -> None:
            path = Path(self.import_options["path"])
            path = path.with_name(path.stem + "_signal.png")
            path, ok = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export Image", str(path.absolute()), "PNG Images (*.png)"
            )
            if not ok:
                return

            # Save defaults
            settings = QtCore.QSettings()
            settings.setValue("ImageExport/SizeX", size.width())
            settings.setValue("ImageExport/SizeY", size.height())
            settings.setValue("ImageExport/DPI", dpi)

            self.exportGraphImage(Path(path), size, dpi, options)

        settings = QtCore.QSettings()
        size = QtCore.QSize(
            int(settings.value("ImageExport/SizeX", 800)),
            int(settings.value("ImageExport/SizeY", 600)),
        )
        dpi = int(settings.value("ImageExport/DPI", 96))

        dlg = ImageExportDialog(
            size=size,
            dpi=dpi,
            options={
                "show legend": self.graph.plot.legend.isVisible(),
                "show detections": True,
                "transparent background": False,
            },
            parent=self,
        )
        dlg.exportSettingsSelected.connect(get_path_and_export)
        dlg.exec()

    def exportGraphImage(
        self,
        path: Path,
        size: QtCore.QSize,
        dpi: float,
        options: dict[str, bool],
    ) -> None:
        dpi_scale = dpi / 96.0
        xrange, yrange = self.graph.plot.viewRange()
        resized_font = QtGui.QFont(self.graph.font)
        resized_font.setPointSizeF(resized_font.pointSizeF() * dpi_scale)

        graph = draw_particle_view(
            None,
            {name: self.asResult(name) for name in self.draw_names},
            self.regions,
            dwell=float(self.options.dwelltime.value() or 1.0),
            show_markers=options.get("show detections", True),
            font=resized_font,
            scale=dpi_scale,
        )

        view_range = self.graph.plot.vb.state["viewRange"]
        graph.plot.vb.setRange(xRange=view_range[0], yRange=view_range[1], padding=0.0)
        graph.plot.legend.setVisible(options.get("show legend", True))
        graph.resize(size)
        graph.show()

        if options.get("transparent background", False):
            background = QtCore.Qt.GlobalColor.transparent
        else:
            background = QtCore.Qt.GlobalColor.white

        graph.exportImage(path, background=background)

    def addExpression(self, name: str, expr: str) -> None:
        self.current_expr[name] = expr
        self.reloadData()
        try:  # attempt to set response of new variable
            reducer = Reducer({n: self.io[n].response.baseValue() for n in self.names})
            data = reducer.reduce(expr)
            self.io[name].response.setBaseValue(data)
        except ReducerException:
            pass

    def removeExpression(self, name: str) -> None:
        self.current_expr.pop(name)
        if name in self.names:
            self.responses = rfn.drop_fields(self.responses, name, usemask=False)
        self.reloadData()

    def loadData(self, data: np.ndarray, options: dict) -> None:
        data = CalculatorDialog.reduceForData(data, self.current_expr)

        # Load any values that need to be set from the import dialog inputs
        self.import_options = options
        if "names" not in self.import_options:
            self.import_options["names"] = {str(n): str(n) for n in data.dtype.names}
        self.label_file.setText(str(options["path"]))

        self.options.blockSignals(True)
        self.options.dwelltime.setBaseValue(options["dwelltime"])

        self.options.blockSignals(False)

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

    def reloadData(self) -> None:
        if self.responses.size > 0:
            self.loadData(self.responses, self.import_options)

    def saveTrimRegion(self) -> None:
        # plot = next(iter(self.graph.plots.values()))
        self.last_region = self.graph.region_start, self.graph.region_end

    def trimRegion(self, name: str) -> tuple[int, int]:
        return self.graph.region_start, self.graph.region_end

    def trimmedResponse(self, name: str) -> np.ndarray:
        trim = self.trimRegion(name)
        return self.responses[name][trim[0] : trim[1]]

    def asResult(self, name: str, calibration_mode: str | None = None) -> SPCalResult:
        if calibration_mode is None:
            method = self.options.efficiency_method.currentText()
            if method in ["Manual Input", "Reference Particle"]:
                calibration_mode = "efficiency"
            elif method == "Mass Response":
                calibration_mode = "mass response"
            else:
                raise ValueError("unable to determine calibration mode")
        return SPCalResult(
            self.label_file.text(),
            self.trimmedResponse(name),
            self.detections[name],
            self.labels,
            self.limits[name],
            calibration_mode=calibration_mode,
        )

    def updateDetections(self) -> None:
        d, l, self.original_regions = {}, {}, {}
        acc_method = self.options.limit_accumulation
        points_req = self.options.points_required
        prominence_req = self.options.prominence_required
        for name in self.names:
            limit_accumulation = self.limits[name].accumulationLimit(acc_method)
            limit_detection = self.limits[name].detection_threshold
            # Ensure limit of accumulation is never greater than the detection limit
            limit_accumulation = np.minimum(limit_accumulation, limit_detection)
            responses = self.trimmedResponse(name)
            if responses.size > 0 and name in self.limits:
                d[name], l[name], self.original_regions[name] = accumulate_detections(
                    responses,
                    limit_accumulation,
                    limit_detection,
                    points_required=points_req,
                    prominence_required=prominence_req,
                    integrate=True,
                )

        self.detections, self.labels, self.regions = combine_detections(
            d, l, self.original_regions
        )

        self.detectionsChanged.emit()

    def updateLimits(self) -> None:
        if self.responses.size == 0:
            return

        method = self.options.limit_method.currentText()
        if method not in ["Compound Poisson", "Poisson"]:  # not Gaussian
            if not self.options.gaussian.isComplete():
                return
        if method not in ["Compound Poisson", "Gaussian"]:  # not Poisson
            if not self.options.poisson.isComplete():
                return
        if method not in ["Poisson", "Gaussian"]:  # not Compound Poisson
            if not self.options.compound_poisson.isComplete():
                return

        if self.options.window_size.isEnabled():
            window_size = self.options.window_size.value() or 0
        else:
            window_size = 0
        window_size = int(window_size or 0)
        max_iter = 100 if self.options.check_iterative.isChecked() else 1

        self.limits.clear()

        compound_kws = self.options.compound_poisson.state()
        gaussian_kws = self.options.gaussian.state()
        poisson_kws = self.options.poisson.state()

        for name in self.names:
            response = self.trimmedResponse(name)
            if response.size == 0:
                continue

            if method == "Manual Input":
                limit = self.io[name].lod_count.value()
                if limit is not None:
                    self.limits[name] = SPCalLimit(
                        np.nanmean(response),
                        limit,
                        name="Manual Input",
                        params={},
                    )
                else:  # If empty limit then fill with best estimate
                    self.limits[name] = SPCalLimit.fromBest(
                        response,
                        compound_kws=compound_kws,
                        poisson_kws=poisson_kws,
                        gaussian_kws=gaussian_kws,
                        window_size=window_size,
                        max_iters=max_iter,
                    )
            else:
                self.limits[name] = SPCalLimit.fromMethodString(
                    method,
                    response,
                    compound_kws=compound_kws,
                    poisson_kws=poisson_kws,
                    gaussian_kws=gaussian_kws,
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
                    str(self.limits[name]),
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

        ymax = 0.0
        for i, name in enumerate(self.draw_names):
            ys = self.responses[name]
            ymax = max(ymax, np.amax(ys))
            pen = QtGui.QPen(self.colorForName(name), 1.0)
            pen.setCosmetic(True)
            self.graph.drawSignal(name, self.events, ys, pen=pen)

        self.graph.region.setBounds((self.events[0], self.events[-1]))
        self.graph.setDataLimits(xMin=0.0, xMax=1.0)

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
                self.limits[name].mean_signal,
                self.limits[name].detection_threshold,
                pen=pen,
            )

    def resetInputs(self) -> None:
        self.blockSignals(True)
        self.io.blockSignals(True)
        self.import_options = {}
        self.responses = np.array([])
        self.events = np.array([])
        self.detections = np.array([])
        self.labels = np.array([])
        self.regions = np.array([])
        self.limits = {}

        self.graph.clear()
        for i in range(self.io.stack.count()):
            self.io.stack.widget(i).clearInputs()
            self.io.stack.widget(i).clearOutputs()
        self.io.combo_name.clear()
        self.io.blockSignals(False)
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
