from PySide6 import QtCore, QtGui, QtWidgets

import numpy as np
import logging

import spcal
from spcal import npdata

from spcal.calc import calculate_limits
from spcal.util import detection_maxima

from spcal.gui.dialogs import ImportDialog
from spcal.gui.iowidgets import SampleIOStack, ReferenceIOStack
from spcal.gui.graphs import ParticleView, graph_colors
from spcal.gui.options import OptionsWidget
from spcal.gui.units import UnitsWidget
from spcal.gui.widgets import ElidedLabel, ValidColorLineEdit

from typing import Dict, Generic, List, Optional, Tuple, TypeVar

logger = logging.getLogger(__name__)


class SampleIOWidget(QtWidgets.QWidget):
    optionsChanged = QtCore.Signal(str)

    def __init__(self, name: str, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.name = name

        self.inputs = QtWidgets.QGroupBox("Inputs")
        self.inputs.setLayout(QtWidgets.QFormLayout())

        self.element = ValidColorLineEdit(color_bad=QtGui.QColor(255, 255, 172))
        self.element.setValid(False)
        self.element.setCompleter(QtWidgets.QCompleter(list(npdata.data.keys())))
        self.element.textChanged.connect(self.elementChanged)

        self.density = UnitsWidget(
            {"g/cm³": 1e-3 * 1e6, "kg/m³": 1.0},
            default_unit="g/cm³",
        )
        self.molarmass = UnitsWidget(
            {"g/mol": 1e-3, "kg/mol": 1.0},
            default_unit="g/mol",
            invalid_color=QtGui.QColor(255, 255, 172),
        )
        self.massfraction = ValidColorLineEdit("1.0")
        self.massfraction.setValidator(QtGui.QDoubleValidator(0.0, 1.0, 4))

        self.element.setToolTip(
            "Input formula for density, molarmass and massfraction."
        )
        self.density.setToolTip("Sample particle density.")
        self.molarmass.setToolTip(
            "Molecular weight, required to calculate intracellular concentrations."
        )
        self.massfraction.setToolTip(
            "Ratio of the mass of the analyte over the mass of the particle."
        )

        self.density.valueChanged.connect(lambda: self.optionsChanged.emit(self.name))
        self.molarmass.valueChanged.connect(lambda: self.optionsChanged.emit(self.name))
        self.massfraction.textChanged.connect(
            lambda: self.optionsChanged.emit(self.name)
        )

        self.inputs.layout().addRow("Formula:", self.element)
        self.inputs.layout().addRow("Density:", self.density)
        self.inputs.layout().addRow("Molar mass:", self.molarmass)
        self.inputs.layout().addRow("Molar ratio:", self.massfraction)

        self.count = QtWidgets.QLineEdit("0")
        self.count.setReadOnly(True)
        self.background_count = QtWidgets.QLineEdit()
        self.background_count.setReadOnly(True)
        self.lod_count = QtWidgets.QLineEdit()
        self.lod_count.setReadOnly(True)

        self.outputs = QtWidgets.QGroupBox("Outputs")
        self.outputs.setLayout(QtWidgets.QFormLayout())
        self.outputs.layout().addRow("Particle count:", self.count)
        self.outputs.layout().addRow("Background count:", self.background_count)
        self.outputs.layout().addRow("LOD count:", self.lod_count)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.inputs)
        layout.addWidget(self.outputs)

        self.setLayout(layout)

    def elementChanged(self, text: str) -> None:
        if text in npdata.data:
            density, mw, mr = npdata.data[text]
            self.element.setValid(True)
            self.density.setValue(density)
            self.density.setUnit("g/cm³")
            self.density.setEnabled(False)
            self.molarmass.setValue(mw)
            self.molarmass.setUnit("g/mol")
            self.molarmass.setEnabled(False)
            self.massfraction.setText(str(mr))
            self.massfraction.setEnabled(False)
        else:
            self.element.setValid(False)
            self.density.setEnabled(True)
            self.molarmass.setEnabled(True)
            self.massfraction.setEnabled(True)

    def resetInputs(self) -> None:
        self.blockSignals(True)
        self.element.setText("")
        self.density.setValue(None)
        self.molarmass.setValue(None)
        self.massfraction.setText("1.0")
        self.blockSignals(False)

    def clearOutputs(self) -> None:
        self.count.setText("")
        self.background_count.setText("")
        self.lod_count.setText("")

    def updateOutputs(
        self,
        responses: np.ndarray,
        detections: np.ndarray,
        labels: np.ndarray,
        limits: Tuple[str, Dict[str, float], np.ndarray],
    ) -> None:
        background = np.mean(responses[labels == 0])
        background_std = np.std(responses[labels == 0])
        lod = np.mean(limits[2]["ld"])  # + self.background

        self.count.setText(f"{detections.size} ± {np.sqrt(detections.size):.1f}")
        self.background_count.setText(f"{background:.4g} ± {background_std:.4g}")
        self.lod_count.setText(
            f"{lod:.4g} ({limits[0]}, {','.join(f'{k}={v}' for k,v in limits[1].items())})"
        )

    def isComplete(self) -> bool:
        return (
            self.density.hasAcceptableInput() and self.massfraction.hasAcceptableInput()
        )


class ReferenceIOWidget(SampleIOWidget):
    def __init__(self, name: str, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(name, parent=parent)

        concentration_units = {
            "fg/L": 1e-18,
            "pg/L": 1e-15,
            "ng/L": 1e-12,
            "μg/L": 1e-9,
            "mg/L": 1e-6,
            "g/L": 1e-3,
            "kg/L": 1.0,
        }
        self.concentration = UnitsWidget(
            units=concentration_units,
            default_unit="ng/L",
            invalid_color=QtGui.QColor(255, 255, 172),
        )
        self.diameter = UnitsWidget(
            {"nm": 1e-9, "μm": 1e-6, "m": 1.0},
            default_unit="nm",
        )

        self.concentration.setToolTip("Reference particle concentration.")
        self.diameter.setToolTip("Reference particle diameter.")

        self.concentration.valueChanged.connect(
            lambda: self.optionsChanged.emit(self.name)
        )
        self.diameter.valueChanged.connect(lambda: self.optionsChanged.emit(self.name))

        self.inputs.layout().removeRow(self.molarmass)
        self.inputs.layout().insertRow(0, "Concentration:", self.concentration)
        self.inputs.layout().insertRow(1, "Diameter:", self.diameter)

        self.efficiency = QtWidgets.QLineEdit()
        self.efficiency.setValidator(QtGui.QDoubleValidator(0.0, 1.0, 10))
        self.efficiency.setReadOnly(True)

        self.massresponse = UnitsWidget(
            {
                "ag/count": 1e-21,
                "fg/count": 1e-18,
                "pg/count": 1e-15,
                "ng/count": 1e-12,
                "μg/count": 1e-9,
                "mg/count": 1e-6,
                "g/count": 1e-3,
                "kg/count": 1.0,
            },
            default_unit="ag/count",
        )
        self.massresponse.setReadOnly(True)

        self.outputs.layout().addRow("Trans. Efficiency:", self.efficiency)
        self.outputs.layout().addRow("Mass Response:", self.massresponse)

    def resetInputs(self) -> None:
        self.blockSignals(True)
        self.diameter.setValue(None)
        self.concentration.setValue(None)
        self.efficiency.setText("")
        self.massresponse.setValue(None)
        self.blockSignals(False)

        super().resetInputs()

    def updateEfficiency(
        self,
        detections: np.ndarray,
        dwell: float,
        response: Optional[float],
        time: float,
        uptake: Optional[float],
    ) -> None:
        self.efficiency.setText("")
        self.massresponse.setValue("")

        density = self.density.baseValue()
        diameter = self.diameter.baseValue()
        if density is None or diameter is None:
            return

        mass = spcal.reference_particle_mass(density, diameter)
        massfraction = (
            float(self.massfraction.text())
            if self.massfraction.hasAcceptableInput()
            else None
        )
        if massfraction is not None:
            self.massresponse.setBaseValue(mass * massfraction / np.mean(detections))

        # If concentration defined use conc method
        concentration = self.concentration.baseValue()
        if concentration is not None and uptake is not None:
            efficiency = spcal.nebulisation_efficiency_from_concentration(
                detections.size,
                concentration=concentration,
                mass=mass,
                flowrate=uptake,
                time=time,
            )
            self.efficiency.setText(f"{efficiency:.4g}")
        elif massfraction is not None and uptake is not None and response is not None:
            efficiency = spcal.nebulisation_efficiency_from_mass(
                detections,
                dwell=dwell,
                mass=mass,
                flowrate=uptake,
                response_factor=response,
                mass_fraction=massfraction,
            )
            self.efficiency.setText(f"{efficiency:.4g}")

    def isComplete(self) -> bool:
        return super().isComplete() and self.diameter.hasAcceptableInput()

IOWidget = TypeVar('IOWidget', bound=type)

class IOStack(QtWidgets.QWidget, Generic[IOWidget]):
    optionsChanged = QtCore.Signal(str)

    def __init__(
        self,
        widget_type: IOWidget,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)

        self.widget_type = widget_type

        self.combo_name = QtWidgets.QComboBox()
        self.stack = QtWidgets.QStackedWidget()

        self.combo_name.currentIndexChanged.connect(self.stack.setCurrentIndex)

        self.layout_top = QtWidgets.QHBoxLayout()
        self.layout_top.addWidget(self.combo_name, 0, QtCore.Qt.AlignRight)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(self.layout_top)
        layout.addWidget(self.stack, 1)
        self.setLayout(layout)

    def __getitem__(self, name: str) -> IOWidget:
        return self.stack.widget(self.combo_name.findText(name))  # type: ignore

    def widgets(self) -> List[IOWidget]:
        return [self.stack.widget(i) for i in range(self.stack.count())]  # type: ignore

    def repopulate(self, names: List[str]) -> None:
        self.blockSignals(True)
        self.combo_name.clear()
        while self.stack.count() > 0:
            self.stack.removeWidget(self.stack.widget(0))

        for name in names:
            self.combo_name.addItem(name)
            widget = self.widget_type(name)
            widget.optionsChanged.connect(self.optionsChanged)
            self.stack.addWidget(widget)
        self.blockSignals(False)

    def resetInputs(self) -> None:
        for widget in self.widgets():
            widget.resetInputs()


class InputWidget(QtWidgets.QWidget):
    optionsChanged = QtCore.Signal()
    detectionsChanged = QtCore.Signal(str)
    limitsChanged = QtCore.Signal(str)

    def __init__(
        self,
        io_stack_widget: IOStack,
        options: OptionsWidget,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.setAcceptDrops(True)

        self.graph = ParticleView()
        self.graph.regionChanged.connect(self.updateLimits)

        self.io = io_stack_widget

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

        self.responses = np.array([])
        self.events = np.array([])
        self.detections: Dict[str, np.ndarray] = {}
        self.labels: Dict[str, np.ndarray] = {}
        self.regions: Dict[str, np.ndarray] = {}
        self.limits: Dict[str, Tuple[str, Dict[str, float], np.ndarray]] = {}

        self.button_file = QtWidgets.QPushButton("Open File")
        self.button_file.pressed.connect(self.dialogLoadFile)

        self.label_file = ElidedLabel("text")

        self.io.layout_top.insertWidget(0, self.button_file, 0, QtCore.Qt.AlignLeft)
        self.io.layout_top.insertWidget(1, self.label_file, 1, QtCore.Qt.AlignLeft)

        layout_chart = QtWidgets.QVBoxLayout()
        layout_chart.addWidget(self.io)
        layout_chart.addWidget(self.graph, 1)

        layout = QtWidgets.QHBoxLayout()
        layout.addLayout(layout_chart, 1)

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
        for i in range(self.io_stack.count()):
            self.io_stack.widget(i).resetInputs()
        self.blockSignals(False)

    def isComplete(self) -> bool:
        return len(self.detections) > 0 and any(
            self.detections[name].size > 0 for name in self.detections
        )


class SampleWidget(InputWidget):
    def __init__(
        self, options: OptionsWidget, parent: Optional[QtWidgets.QWidget] = None
    ):
        super().__init__(SampleIOStack, options, parent=parent)


class ReferenceWidget(InputWidget):
    def __init__(
        self, options: OptionsWidget, parent: Optional[QtWidgets.QWidget] = None
    ):
        super().__init__(ReferenceIOStack, options, parent=parent)

        # dwelltime covered by detectionsChanged
        self.options.response.valueChanged.connect(lambda: self.updateEfficiency(None))
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
        response = self.options.response.baseValue()
        time = self.events.size * dwell
        uptake = self.options.uptake.baseValue()

        for name in names:
            self.io[name].updateEfficiency(
                self.detections[name], dwell, response, time, uptake
            )
