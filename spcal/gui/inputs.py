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
from spcal.gui.tables import ParticleTable
from spcal.gui.units import UnitsWidget
from spcal.gui.widgets import (
    ElidedLabel,
    # RangeSlider,
    ValidColorLineEdit,
)

from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class InputWidget(QtWidgets.QWidget):
    optionsChanged = QtCore.Signal()
    detectionsChanged = QtCore.Signal(int)
    limitsChanged = QtCore.Signal()

    def __init__(
        self, options: OptionsWidget, parent: Optional[QtWidgets.QWidget] = None
    ):
        super().__init__(parent)
        self.setAcceptDrops(True)

        self.graph = ParticleView()

        self.redraw_graph_requested = False
        self.draw_mode = "All"

        self.limitsChanged.connect(self.updateDetections)
        self.limitsChanged.connect(self.requestRedraw)

        self.detectionsChanged.connect(self.updateTexts)

        self.options = options
        self.options.dwelltime.valueChanged.connect(self.updateLimits)
        self.options.method.currentTextChanged.connect(self.updateLimits)
        self.options.window_size.editingFinished.connect(self.updateLimits)
        self.options.check_use_window.toggled.connect(self.updateLimits)
        self.options.sigma.editingFinished.connect(self.updateLimits)
        self.options.manual.editingFinished.connect(self.updateLimits)
        self.options.error_rate_alpha.editingFinished.connect(self.updateLimits)
        self.options.error_rate_beta.editingFinished.connect(self.updateLimits)

        self.responses = np.array([])
        self.detections: Dict[str, np.ndarray] = {}
        self.labels: Dict[str, np.ndarray] = {}
        self.regions: Dict[str, np.ndarray] = {}
        self.limits: Dict[str, Tuple[str, Dict[str, float], np.ndarray]] = {}

        self.combo_name = QtWidgets.QComboBox()
        self.combo_name.currentTextChanged.connect(self.updateTexts)

        self.button_file = QtWidgets.QPushButton("Open File")
        self.button_file.pressed.connect(self.dialogLoadFile)

        self.label_file = ElidedLabel()
        self.label_file.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored
        )

        self.table = ParticleTable()
        self.table.model().dataChanged.connect(self.updateLimits)

        # self.slider = RangeSlider()
        # self.slider.setRange(0, 1)
        # self.slider.setValues(0, 1)
        # self.slider.valueChanged.connect(self.updateTrim)
        # self.slider.value2Changed.connect(self.updateTrim)
        # self.slider.sliderReleased.connect(self.updateLimits)

        # Sample options

        self.inputs = QtWidgets.QGroupBox("Inputs")
        self.inputs.setLayout(QtWidgets.QFormLayout())

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

        layout_table_file = QtWidgets.QHBoxLayout()
        layout_table_file.addWidget(self.label_file, 1, QtCore.Qt.AlignRight)
        layout_table_file.addWidget(self.button_file, 0, QtCore.Qt.AlignLeft)

        # layout_slider = QtWidgets.QHBoxLayout()
        # layout_slider.addWidget(QtWidgets.QLabel("Trim:"))
        # layout_slider.addWidget(self.slider, QtCore.Qt.AlignRight)

        layout_io = QtWidgets.QHBoxLayout()
        layout_io.addWidget(self.inputs)
        layout_io.addWidget(self.outputs)

        layout_chart = QtWidgets.QVBoxLayout()
        layout_chart.addLayout(layout_table_file, 0)
        layout_chart.addLayout(layout_io)
        layout_chart.addWidget(self.graph, 1)
        # layout_chart.addLayout(layout_slider)

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

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        if self.redraw_graph_requested:
            self.requestRedraw()

    # def numberOfEvents(self) -> int:
    #     return self.slider.right() - self.slider.left()

    # def timeAsSeconds(self) -> float:
    #     dwell = self.options.dwelltime.baseValue()
    #     if dwell is None:
    #         raise ValueError("timeAsSeconds: dwelltime not defined!")
    #     return (self.slider.right() - self.slider.left()) * dwell

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

        self.combo_name.blockSignals(True)
        self.combo_name.clear()
        self.combo_name.addItems([name for name in self.responses.dtype.names])
        self.combo_name.adjustSize()
        self.combo_name.blockSignals(False)

        # # Update Chart and slider
        # offset = self.slider.maximum() - self.slider.right()
        # self.slider.setRange(0, self.responses.shape[0])

        # right = max(self.slider.maximum() - offset, 1)
        # left = min(self.slider.left(), right - 1)
        # self.slider.setValues(left, right)
        # self.chart.xaxis.setRange(self.slider.minimum(), self.slider.maximum())

        self.updateLimits()

    def updateDetections(self) -> None:
        responses = self.responses
        # [self.slider.left() : self.slider.right()]

        self.detections.clear()
        self.labels.clear()
        self.regions.clear()

        if responses.size > 0 and len(self.limits) > 0:
            for name in responses.dtype.names:
                limits = self.limits[name][2]
                (
                    self.detections[name],
                    self.labels[name],
                    self.regions[name],
                ) = spcal.accumulate_detections(
                    responses[name], limits["lc"], limits["ld"]
                )

        self.detectionsChanged.emit(len(self.detections))

    def updateLimits(self) -> None:
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

        self.limits = {}

        if len(self.responses) == 0:
            return
        for name in self.responses.dtype.names:
            if method == "Manual Input":
                limit = float(self.options.manual.text())
                self.limits[name] = (
                    method,
                    {},
                    np.array(
                        [(np.mean(self.responses[name]), limit, limit)],
                        dtype=calculate_limits.dtype,
                    ),
                )
            else:
                self.limits[name] = calculate_limits(
                    self.responses[name],
                    method,
                    sigma,
                    (alpha, beta),
                    window=window_size,
                )

        self.limitsChanged.emit()

    def updateTexts(self) -> None:
        name = self.combo_name.currentText()

        if name not in self.detections:
            self.count.setText("")
            self.background_count.setText("")
            self.lod_count.setText("")
        else:
            responses = self.responses[name]
            # [self.slider.left() : self.slider.right()]
            self.background = np.mean(responses[self.labels[name] == 0])
            self.background_std = np.std(responses[self.labels[name] == 0])
            lod = np.mean(self.limits[name][2]["ld"])  # + self.background

            self.count.setText(
                f"{self.detections[name].size} ± {np.sqrt(self.detections[name].size):.1f}"
            )
            self.background_count.setText(
                f"{self.background:.4g} ± {self.background_std:.4g}"
            )
            self.lod_count.setText(
                f"{lod:.4g} ({self.limits[name][0]}, {','.join(f'{k}={v}' for k,v in self.limits[name][1].items())})"
            )

    def drawGraph(self, name: Optional[str] = None) -> None:
        self.graph.clear()
        if len(self.responses) == 0:
            return

        dwell = self.options.dwelltime.baseValue()
        if dwell is None:
            raise ValueError("dwell is None")

        xs = np.arange(self.responses.size) * dwell

        if self.draw_mode == "stacked":
            for name in self.responses.dtype.names:
                assert name is not None
                ys = self.responses[name]

                self.graph.addParticlePlot(name)
                self.graph.drawParticleSignal(name, xs, ys)

                if name in self.regions and self.regions[name].size > 0:
                    maxima = detection_maxima(ys, self.regions[name])
                    self.graph.drawParticleMaxima(name, xs[maxima], ys[maxima])

                if name in self.limits:
                    self.graph.drawParticleLimits(name, xs, self.limits[name][2])
        else:
            self.graph.addParticlePlot("overlay")
            for name, color in zip(self.responses.dtype.names, self.graph.plot_colors):
                assert name is not None
                ys = self.responses[name]

                pen = QtGui.QPen(color, 1.0)
                pen.setCosmetic(True)
                brush = QtGui.QBrush(color)

                self.graph.drawParticleSignal("overlay", xs, ys, label=name, pen=pen)

                if name in self.regions and self.regions[name].size > 0:
                    maxima = detection_maxima(ys, self.regions[name])
                    self.graph.drawParticleMaxima(
                        "overlay", xs[maxima], ys[maxima], brush=brush
                    )

    def requestRedraw(self) -> None:
        if self.isVisible():
            self.drawGraph()
            self.redraw_graph_requested = False
        else:
            self.redraw_graph_requested = True

    def resetInputs(self) -> None:
        self.blockSignals(True)
        self.count.setText("0")
        self.background_count.setText("")
        self.lod_count.setText("")

        self.responses = np.array([])
        self.detections.clear()
        self.labels.clear()
        self.limits.clear()

        self.blockSignals(False)

        self.optionsChanged.emit()


class SampleWidget(InputWidget):
    def __init__(self, options: OptionsWidget, parent: QtWidgets.QWidget = None):
        super().__init__(options, parent=parent)

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

    def isComplete(self) -> bool:
        return (
            len(self.detections) > 0
            and any(self.detections[name].size > 0 for name in self.detections)
            and self.massfraction.hasAcceptableInput()
            and self.density.hasAcceptableInput()
        )

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
        super().resetInputs()


class ReferenceWidget(InputWidget):
    def __init__(self, options: OptionsWidget, parent: QtWidgets.QWidget = None):
        super().__init__(options, parent=parent)

        concentration_units = {
            "fg/L": 1e-18,
            "pg/L": 1e-15,
            "ng/L": 1e-12,
            "μg/L": 1e-9,
            "mg/L": 1e-6,
            "g/L": 1e-3,
            "kg/L": 1.0,
        }

        self.element = ValidColorLineEdit(color_bad=QtGui.QColor(255, 255, 172))
        self.element.setValid(False)
        self.element.setCompleter(QtWidgets.QCompleter(list(npdata.data.keys())))
        self.element.textChanged.connect(self.elementChanged)

        self.concentration = UnitsWidget(
            units=concentration_units,
            default_unit="ng/L",
            invalid_color=QtGui.QColor(255, 255, 172),
        )
        self.density = UnitsWidget(
            {"g/cm³": 1e-3 * 1e6, "kg/m³": 1.0},
            default_unit="g/cm³",
        )
        self.diameter = UnitsWidget(
            {"nm": 1e-9, "μm": 1e-6, "m": 1.0},
            default_unit="nm",
        )
        self.massfraction = ValidColorLineEdit(
            "1.0", color_bad=QtGui.QColor(255, 255, 172)
        )
        self.massfraction.setValidator(QtGui.QDoubleValidator(0.0, 1.0, 4))

        self.element.setToolTip("Input formula for density and massfraction.")
        self.concentration.setToolTip("Reference particle concentration.")
        self.density.setToolTip("Reference particle density.")
        self.diameter.setToolTip("Reference particle diameter.")
        self.massfraction.setToolTip(
            "Ratio of the mass of the particle to the analyte."
        )

        self.concentration.valueChanged.connect(self.optionsChanged)
        self.density.valueChanged.connect(self.optionsChanged)
        self.diameter.valueChanged.connect(self.optionsChanged)
        self.massfraction.textChanged.connect(self.optionsChanged)

        self.inputs.layout().addRow("Concentration:", self.concentration)
        self.inputs.layout().addRow("Diameter:", self.diameter)
        self.inputs.layout().addRow("Formula:", self.element)
        self.inputs.layout().addRow("Density:", self.density)
        self.inputs.layout().addRow("Mass fraction:", self.massfraction)

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

        self.options.dwelltime.valueChanged.connect(self.recalculate)
        self.options.response.valueChanged.connect(self.recalculate)
        self.options.uptake.valueChanged.connect(self.recalculate)
        self.optionsChanged.connect(self.recalculate)
        self.detectionsChanged.connect(self.recalculate)

    def recalculate(self) -> None:
        self.efficiency.setText("")
        self.massresponse.setValue("")

        density = self.density.baseValue()
        diameter = self.diameter.baseValue()
        if (
            len(self.detections) == 0
            or any(self.detections[name].size == 0 for name in self.detections)
            or density is None
            or diameter is None
        ):
            return

        mass = spcal.reference_particle_mass(density, diameter)
        massfraction = (
            float(self.massfraction.text())
            if self.massfraction.hasAcceptableInput()
            else None
        )
        if massfraction is not None:
            self.massresponse.setBaseValue(
                mass * massfraction / np.mean(self.detections)
            )

        # If concentration defined use conc method
        concentration = self.concentration.baseValue()
        uptake = self.options.uptake.baseValue()
        time = self.timeAsSeconds()
        if concentration is not None and uptake is not None and time is not None:
            efficiency = spcal.nebulisation_efficiency_from_concentration(
                self.detections.size,
                concentration=concentration,
                mass=mass,
                flowrate=uptake,
                time=time,
            )
            self.efficiency.setText(f"{efficiency:.4g}")
            return

        # Else use the other method
        dwell = self.options.dwelltime.baseValue()
        response = self.options.response.baseValue()
        if (
            dwell is not None
            and response is not None
            and uptake is not None
            and massfraction is not None
        ):
            efficiency = spcal.nebulisation_efficiency_from_mass(
                self.detections,
                dwell=dwell,
                mass=mass,
                flowrate=uptake,
                response_factor=response,
                mass_fraction=massfraction,
            )
            self.efficiency.setText(f"{efficiency:.4g}")

    def elementChanged(self, text: str) -> None:
        if text in npdata.data:
            density, _, mr = npdata.data[text]
            self.element.setValid(True)
            self.density.setValue(density)
            self.density.setUnit("g/cm³")
            self.density.setEnabled(False)
            self.massfraction.setText(str(mr))
            self.massfraction.setEnabled(False)
        else:
            self.element.setValid(False)
            self.density.setEnabled(True)
            self.massfraction.setEnabled(True)

    def isComplete(self) -> bool:
        return (
            len(self.detections) > 0
            and any(self.detections[name].size > 0 for name in self.detections)
            and self.diameter.hasAcceptableInput()
            and self.massfraction.hasAcceptableInput()
            and self.density.hasAcceptableInput()
        )

    def resetInputs(self) -> None:
        self.blockSignals(True)
        self.element.setText("")
        self.diameter.setValue(None)
        self.density.setValue(None)
        self.massfraction.setText("1.0")
        self.concentration.setValue(None)

        self.efficiency.setText("")
        self.massresponse.setValue(None)
        self.blockSignals(False)
        super().resetInputs()
