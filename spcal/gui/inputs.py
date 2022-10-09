from PySide6 import QtCore, QtGui, QtWidgets

import numpy as np
import logging

import spcal
from spcal import npdata

from spcal.calc import calculate_limits
from spcal.util import detection_maxima

from spcal.gui.dialogs import ImportDialog
from spcal.gui.graphs import ParticleView
from spcal.gui.options import OptionsWidget
from spcal.gui.units import UnitsWidget
from spcal.gui.widgets import ElidedLabel, ValidColorLineEdit

from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class SampleIOWidget(QtWidgets.QWidget):
    optionsChanged = QtCore.Signal()

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

        self.density.valueChanged.connect(self.optionsChanged)
        self.molarmass.valueChanged.connect(self.optionsChanged)
        self.massfraction.textChanged.connect(self.optionsChanged)

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

        self.concentration.valueChanged.connect(self.optionsChanged)
        self.diameter.valueChanged.connect(self.optionsChanged)

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

    def recalculate(
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

class SampleIOStack(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)

        self.combo_name = QtWidgets.QComboBox()
        self.stack = QtWidgets.QStackedWidget()

        self.combo_name.currentIndexChanged.connect(self.stack.setCurrentIndex)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.combo_name, 0, QtCore.Qt.AlignRight)
        layout.addWidget(self.stack, 1)
        self.setLayout(layout)

    def nameChanged(self) -> None:


    def repopulate(self, names: List[str]) -> None:
        self.combo_name.clear()
        for i in range(self.stack.count()):
            self.stack.removeWidget(self.stack.widget(i))

        for name in names:
            self.combo_name.addText(name)
            self.stack.addWidget(SampleIOWidget(name))


class InputWidget(QtWidgets.QWidget):
    optionsChanged = QtCore.Signal()
    detectionsChanged = QtCore.Signal(str)
    limitsChanged = QtCore.Signal(str)

    def __init__(
        self, options: OptionsWidget, parent: Optional[QtWidgets.QWidget] = None
    ):
        super().__init__(parent)
        self.setAcceptDrops(True)

        self.graph = ParticleView()
        self.graph.regionChanged.connect(self.updateLimits)

        self.redraw_graph_requested = False
        self.draw_mode = "Overlay"

        self.limitsChanged.connect(self.updateDetections)
        self.limitsChanged.connect(self.drawLimits)

        self.detectionsChanged.connect(self.updateTexts)
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

        self.label_file = ElidedLabel()
        self.label_file.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored
        )

        layout_table_file = QtWidgets.QHBoxLayout()
        layout_table_file.addWidget(self.label_file, 1, QtCore.Qt.AlignRight)
        layout_table_file.addWidget(self.button_file, 0, QtCore.Qt.AlignLeft)

        self.combo_name = QtWidgets.QComboBox()
        self.combo_name.currentTextChanged.connect(self.updateTexts)

        self.io_stack = QtWidgets.QStackedWidget()

        layout_io = QtWidgets.QVBoxLayout()
        layout_io.addWidget(self.combo_name, 0, QtCore.Qt.AlignRight)
        layout_io.addWidget(self.io_stack, 1)

        layout_chart = QtWidgets.QVBoxLayout()
        layout_chart.addLayout(layout_table_file, 0)
        layout_chart.addLayout(layout_io)
        layout_chart.addWidget(self.graph, 1)

        layout = QtWidgets.QHBoxLayout()
        layout.addLayout(layout_chart, 1)

        self.setLayout(layout)

    def getIOForName(self, name: str) -> ReferenceIOWidget:
        return self.io_stack.widget(self.combo_name.findText(name))

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

        self.combo_name.blockSignals(True)
        self.combo_name.clear()
        self.combo_name.addItems([name for name in self.responses.dtype.names])
        self.combo_name.adjustSize()
        self.combo_name.blockSignals(False)

        self.addIOWidgets()

        # Update graph, limits and detections
        self.drawGraph()
        self.updateLimits()

    def addIOWidgets(self):
        for i in range(self.io_stack.count()):
            self.io_stack.removeWidget(self.io_stack.widget(i))
        for name in self.responses.dtype.names:
            io = SampleIOWidget(name)
            io.optionsChanged.connect(self.optionsChanged)
            self.io_stack.addWidget(io)

    def trimRegion(self, name: str) -> Tuple[int, int]:
        if self.draw_mode == "Overlay":
            plot = self.graph.plots["Overlay"]
        else:
            plot = self.graph.plots[name]
        return plot.region_start, plot.region_end

    def updateDetections(self, name: Optional[str] = None) -> None:
        if name is None or name == "Overlay":
            names = self.responses.dtype.names
        else:
            names = [name]

        for n in names:
            trim = self.trimRegion(n)
            responses = self.responses[n][trim[0] : trim[1]]
            if responses.size > 0 and n in self.limits:
                limits = self.limits[n][2]
                (
                    self.detections[n],
                    self.labels[n],
                    self.regions[n],
                ) = spcal.accumulate_detections(responses, limits["lc"], limits["ld"])
            else:
                self.detections.pop(n)
                self.labels.pop(n)
                self.regions.pop(n)

            self.detectionsChanged.emit(n)

    def updateLimits(self, name: Optional[str] = None) -> None:
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

        if name is None or name == "Overlay":
            names = self.responses.dtype.names
        else:
            names = [name]

        for n in names:
            trim = self.trimRegion(n)
            response = self.responses[n][trim[0] : trim[1]]
            if response.size == 0:
                self.limits.pop(n)
                continue

            if method == "Manual Input":
                limit = float(self.options.manual.text())
                self.limits[n] = (
                    method,
                    {},
                    np.array(
                        [(np.mean(response), limit, limit)],
                        dtype=calculate_limits.dtype,
                    ),
                )
            else:
                self.limits[n] = calculate_limits(
                    response, method, sigma, (alpha, beta), window=window_size
                )
            self.limitsChanged.emit(n)

    def updateTexts(self) -> None:
        name = self.combo_name.currentText()
        widget = self.getIOForName(name)

        if name not in self.detections:
            widget.count.setText("")
            widget.background_count.setText("")
            widget.lod_count.setText("")
        else:
            trim = self.trimRegion(name)
            responses = self.responses[name][trim[0] : trim[1]]
            background = np.mean(responses[self.labels[name] == 0])
            background_std = np.std(responses[self.labels[name] == 0])
            lod = np.mean(self.limits[name][2]["ld"])  # + self.background

            widget.count.setText(
                f"{self.detections[name].size} ± {np.sqrt(self.detections[name].size):.1f}"
            )
            widget.background_count.setText(f"{background:.4g} ± {background_std:.4g}")
            widget.lod_count.setText(
                f"{lod:.4g} ({self.limits[name][0]}, {','.join(f'{k}={v}' for k,v in self.limits[name][1].items())})"
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
            for name, color in zip(self.responses.dtype.names, self.graph.plot_colors):
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
            color = self.graph.plot_colors[name_idx]
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

        self.optionsChanged.emit()

    def isComplete(self) -> bool:
        return len(self.detections) > 0 and any(
            self.detections[name].size > 0 for name in self.detections
        )


class SampleWidget(InputWidget):
    pass

    # def __init__(self, options: OptionsWidget, parent: QtWidgets.QWidget = None):
    #     super().__init__(options, parent=parent)

    # def isComplete(self) -> bool:
    #     return (
    #         len(self.detections) > 0
    #         and any(self.detections[name].size > 0 for name in self.detections)
    #         and self.massfraction.hasAcceptableInput()
    #         and self.density.hasAcceptableInput()
    #     )


#     def elementChanged(self, text: str) -> None:
#         if text in npdata.data:
#             density, mw, mr = npdata.data[text]
#             self.element.setValid(True)
#             self.density.setValue(density)
#             self.density.setUnit("g/cm³")
#             self.density.setEnabled(False)
#             self.molarmass.setValue(mw)
#             self.molarmass.setUnit("g/mol")
#             self.molarmass.setEnabled(False)
#             self.massfraction.setText(str(mr))
#             self.massfraction.setEnabled(False)
#         else:
#             self.element.setValid(False)
#             self.density.setEnabled(True)
#             self.molarmass.setEnabled(True)
#             self.massfraction.setEnabled(True)

#     def resetInputs(self) -> None:
#         self.blockSignals(True)
#         self.element.setText("")
#         self.density.setValue(None)
#         self.molarmass.setValue(None)
#         self.massfraction.setText("1.0")
#         self.blockSignals(False)
#         super().resetInputs()


class ReferenceWidget(InputWidget):
    def __init__(self, options: OptionsWidget, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(options, parent=parent)

        self.options.dwelltime.valueChanged.connect(self.recalculate)
        self.options.response.valueChanged.connect(self.recalculate)
        self.options.uptake.valueChanged.connect(self.recalculate)
        self.optionsChanged.connect(self.recalculate)
        self.detectionsChanged.connect(self.recalculate)

    def addIOWidgets(self):
        for i in range(self.io_stack.count()):
            self.io_stack.removeWidget(self.io_stack.widget(i))
        for name in self.responses.dtype.names:
            io = ReferenceIOWidget(name)
            io.optionsChanged.connect(self.optionsChanged)
            self.io_stack.addWidget(io)

    def recalculate(self) -> None:
        name = self.combo_name.currentText()
        io = self.getIOForName(name)

        if io is None:
            return

        dwell = self.options.dwelltime.baseValue()
        assert dwell is not None
        response = self.options.response.baseValue()
        time = self.events.size * dwell
        uptake = self.options.uptake.baseValue()

        io.recalculate(self.detections[name], dwell, response, time, uptake)
