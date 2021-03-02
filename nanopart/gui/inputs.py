from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtCharts import QtCharts

import numpy as np
from pathlib import Path

import nanopart
from nanopart import npdata

from nanopart.gui.charts import ParticleChart
from nanopart.gui.tables import ParticleTable
from nanopart.gui.units import UnitsWidget
from nanopart.gui.widgets import ElidedLabel, RangeSlider, ValidColorLineEdit

from nanopart.gui.options import OptionsWidget

from typing import Tuple


# todo changing values in table doesnt update linits


class InputWidget(QtWidgets.QWidget):
    optionsChanged = QtCore.Signal()
    detectionsChanged = QtCore.Signal(int)
    limitsChanged = QtCore.Signal()

    def __init__(self, options: OptionsWidget, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.setAcceptDrops(True)

        # self.detectionsChanged.connect(self.redrawChart)
        self.limitsChanged.connect(self.updateDetections)
        self.limitsChanged.connect(self.redrawLimits)

        self.options = options
        self.options.dwelltime.valueChanged.connect(self.updateLimits)
        self.options.method.currentTextChanged.connect(self.updateLimits)

        self.limits: Tuple[str, float, float, float] = None
        self.detections = np.array([], dtype=np.float64)

        self.button_file = QtWidgets.QPushButton("Open File")
        self.button_file.pressed.connect(self.dialogLoadFile)

        self.label_file = ElidedLabel()
        self.label_file.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored
        )

        self.chart = ParticleChart()
        self.chartview = QtCharts.QChartView(self.chart)
        self.chartview.setRubberBand(QtCharts.QChartView.HorizontalRubberBand)

        self.table = ParticleTable()
        self.table.unitChanged.connect(self.updateLimits)
        self.table.model.dataChanged.connect(self.updateDetections)
        self.table.model.dataChanged.connect(self.redrawChart)

        self.slider = RangeSlider()
        self.slider.setRange(0, 100)
        self.slider.valueChanged.connect(self.updateTrim)
        self.slider.value2Changed.connect(self.updateTrim)
        self.slider.sliderReleased.connect(self.updateDetections)

        # Sample options

        self.inputs = QtWidgets.QGroupBox("Inputs")
        self.inputs.setLayout(QtWidgets.QFormLayout())

        self.count = QtWidgets.QLineEdit("0")
        self.count.setReadOnly(True)
        self.background_count = QtWidgets.QLineEdit()
        self.background_count.setReadOnly(True)

        self.outputs = QtWidgets.QGroupBox("Outputs")
        self.outputs.setLayout(QtWidgets.QFormLayout())
        self.outputs.layout().addRow("Particle count:", self.count)
        self.outputs.layout().addRow("Background count:", self.background_count)

        layout_file = QtWidgets.QHBoxLayout()
        layout_file.addWidget(self.button_file, 0, QtCore.Qt.AlignLeft)
        layout_file.addWidget(self.label_file, 1)

        layout_table = QtWidgets.QVBoxLayout()
        layout_table.addLayout(layout_file, 0)
        layout_table.addWidget(self.table, 1)

        layout_slider = QtWidgets.QHBoxLayout()
        layout_slider.addWidget(QtWidgets.QLabel("Trim:"))
        layout_slider.addWidget(self.slider, QtCore.Qt.AlignRight)

        layout_io = QtWidgets.QHBoxLayout()
        layout_io.addWidget(self.inputs)
        layout_io.addWidget(self.outputs)

        layout_chart = QtWidgets.QVBoxLayout()
        layout_chart.addLayout(layout_io)
        layout_chart.addWidget(self.chartview, 1)
        layout_chart.addLayout(layout_slider)

        layout = QtWidgets.QHBoxLayout()
        layout.addLayout(layout_table, 0)
        layout.addLayout(layout_chart, 1)

        self.setLayout(layout)

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
                self.loadFile(url.toLocalFile())
            event.acceptProposedAction()
        elif event.mimeData().hasHtml():
            pass
        elif event.mimeData().hasText():
            pass
        else:
            super().dropEvent(event)

    def timeAsSeconds(self) -> float:
        dwell = self.options.dwelltime.baseValue()
        if dwell is None:
            return None
        return (self.slider.right() - self.slider.left()) * dwell

    def dialogLoadFile(self) -> None:
        file, _filter = QtWidgets.QFileDialog.getOpenFileName(self, "Open", "")
        if file != "":
            self.loadFile(file)

    def loadFile(self, file: str) -> None:
        try:
            parameters = self.table.loadFile(file)
        except ValueError:
            return

        self.label_file.setText(Path(file).name)

        # Update Chart and slider
        self.slider.setRange(0, self.table.model.rowCount())
        self.slider.setValues(0, self.table.model.rowCount())
        self.chart.xaxis.setRange(self.slider.left(), self.slider.right())

        # Update dwell time
        if "dwelltime" in parameters:
            self.options.dwelltime.setBaseValue(parameters["dwelltime"])

        self.redrawChart()
        self.redrawLimits()

    def updateDetections(self) -> None:
        responses = self.table.asCounts(
            self.options.dwelltime.baseValue(),
            trim=(self.slider.left(), self.slider.right()),
        )
        if self.limits is None or responses is None or responses.size == 0:
            self.detections = np.array([])

            self.count.setText("")
            self.background_count.setText("")
        else:
            detections, labels = nanopart.accumulate_detections(
                responses, self.limits[3], self.limits[2]
            )

            self.detections = detections
            background = np.mean(responses[labels == 0])

            self.count.setText(str(detections.size))
            self.background_count.setText(f"{background:.8g}")

        self.detectionsChanged.emit(self.detections.size)

    def updateLimits(self) -> None:
        method = self.options.method.currentText()
        responses = self.table.asCounts(
            self.options.dwelltime.baseValue(),
            trim=(self.slider.left(), self.slider.right()),
        )

        self.limits = None

        if responses is None or responses.size == 0:
            return

        mean = np.mean(responses)
        gaussian = None
        poisson: Tuple[float, float] = None

        if method == "Automatic":
            method = "Poisson" if mean < 50.0 else "Gaussian"

        if method in ["Highest", "Gaussian"]:
            if self.options.sigma.hasAcceptableInput():
                sigma = float(self.options.sigma.text())
                gaussian = mean + sigma * np.std(responses)

        if method in ["Highest", "Poisson"]:
            if self.options.epsilon.hasAcceptableInput():
                epsilon = float(self.options.epsilon.text())
                yc, yd = nanopart.poisson_limits(mean, epsilon=epsilon)
                poisson = (mean + yc, mean + yd)

        if method == "Highest":
            if gaussian is not None and poisson is not None:
                method = "Gaussian" if gaussian > poisson[1] else "Poisson"

        if method == "Gaussian" and gaussian is not None:
            self.limits = (method, mean, sigma, gaussian)
        elif method == "Poisson" and poisson is not None:
            self.limits = (method, mean, poisson[0], poisson[1])

        self.limitsChanged.emit()

    def redrawChart(self) -> None:
        responses = self.table.asCounts(
            self.options.dwelltime.baseValue(),
        )
        if responses is None or responses.size == 0:
            return

        events = np.arange(responses.size)
        self.chart.setData(np.stack((events, responses), axis=1))

        self.chart.drawVerticalLines(
            [self.slider.left(), self.slider.right()],
            pens=[
                QtGui.QPen(QtGui.QColor(255, 0, 0), 2.0),
                QtGui.QPen(QtGui.QColor(255, 0, 0), 2.0),
            ],
            visible_in_legend=False,  # type: ignore
        )
        self.chart.updateYRange()

    def redrawLimits(self) -> None:
        if self.limits is None:
            self.chart.clearHorizontalLines()
        elif self.limits[0] == "Gaussian":
            if len(self.chart.hlines) == 2:
                self.chart.setHorizontalLines([self.limits[1], self.limits[3]])
            else:
                self.chart.drawHorizontalLines(
                    [self.limits[1], self.limits[3]],
                    names=["mean", f"{self.limits[2]}σ"],
                    pens=[
                        QtGui.QPen(QtGui.QColor(255, 0, 0), 1.0, QtCore.Qt.DashLine),
                        QtGui.QPen(QtGui.QColor(0, 0, 255), 1.0, QtCore.Qt.DashLine),
                    ],
                )
        else:
            if len(self.chart.hlines) == 3:
                self.chart.setHorizontalLines(
                    [self.limits[1], self.limits[2], self.limits[3]]
                )
            else:
                self.chart.drawHorizontalLines(
                    [self.limits[1], self.limits[2], self.limits[3]],
                    names=["mean", "Lc", "Ld"],
                    pens=[
                        QtGui.QPen(QtGui.QColor(255, 0, 0), 1.0, QtCore.Qt.DashLine),
                        QtGui.QPen(QtGui.QColor(0, 172, 0), 1.0, QtCore.Qt.DashLine),
                        QtGui.QPen(QtGui.QColor(0, 0, 255), 1.0, QtCore.Qt.DashLine),
                    ],
                )

    def updateTrim(self) -> None:
        values = [self.slider.left(), self.slider.right()]
        self.chart.setVerticalLines(values)  # type: ignore


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
        self.molarratio = ValidColorLineEdit("1.0")
        self.molarratio.setValidator(QtGui.QDoubleValidator(0.0, 1.0, 4))

        self.element.setToolTip("Input formula for density and molarratio.")
        self.density.setToolTip("Sample particle density.")
        self.molarmass.setToolTip(
            "Molecular weight, used to calcullate # atoms per particle."
        )
        self.molarratio.setToolTip("Ratio of the mass of the particle to the analyte.")

        self.density.valueChanged.connect(self.optionsChanged)
        self.molarmass.valueChanged.connect(self.optionsChanged)
        self.molarratio.textChanged.connect(self.optionsChanged)

        self.inputs.layout().addRow("Formula:", self.element)
        self.inputs.layout().addRow("Density:", self.density)
        self.inputs.layout().addRow("Molar mass:", self.molarmass)
        self.inputs.layout().addRow("Molar ratio:", self.molarratio)

    def isComplete(self) -> bool:
        return (
            self.detections is not None
            and self.detections.size > 0
            and self.molarratio.hasAcceptableInput()
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
            self.molarratio.setText(str(mr))
            self.molarratio.setEnabled(False)
        else:
            self.element.setValid(False)
            self.density.setEnabled(True)
            self.molarmass.setEnabled(True)
            self.molarratio.setEnabled(True)


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
        self.molarratio = ValidColorLineEdit(
            "1.0", color_bad=QtGui.QColor(255, 255, 172)
        )
        self.molarratio.setValidator(QtGui.QDoubleValidator(0.0, 1.0, 4))

        self.element.setToolTip("Input formula for density and molarratio.")
        self.concentration.setToolTip("Reference particle concentration.")
        self.density.setToolTip("Reference particle density.")
        self.diameter.setToolTip("Reference particle diameter.")
        self.molarratio.setToolTip("Ratio of the mass of the particle to the analyte.")

        self.concentration.valueChanged.connect(self.optionsChanged)
        self.density.valueChanged.connect(self.optionsChanged)
        self.diameter.valueChanged.connect(self.optionsChanged)
        self.molarratio.textChanged.connect(self.optionsChanged)

        self.inputs.layout().addRow("Concentration:", self.concentration)
        self.inputs.layout().addRow("Diameter:", self.diameter)
        self.inputs.layout().addRow("Formula:", self.element)
        self.inputs.layout().addRow("Density:", self.density)
        self.inputs.layout().addRow("Molar ratio:", self.molarratio)

        self.efficiency = ValidColorLineEdit()
        self.efficiency.setValidator(QtGui.QDoubleValidator(0.0, 1.0, 10))
        self.efficiency.setReadOnly(True)

        self.massresponse = UnitsWidget(
            {
                "ag": 1e-21,
                "fg": 1e-18,
                "pg": 1e-15,
                "ng": 1e-12,
                "μg": 1e-9,
                "mg": 1e-6,
                "g": 1e-3,
                "kg": 1.0,
            },
            default_unit="ag",
            update_value_with_unit=True,
        )
        self.massresponse.setReadOnly(True)

        self.outputs.layout().addRow("Neb. Efficiency:", self.efficiency)
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
        if self.detections.size == 0 or any(x is None for x in [density, diameter]):
            return

        mass = nanopart.reference_particle_mass(density, diameter)
        molarratio = (
            float(self.molarratio.text())
            if self.molarratio.hasAcceptableInput()
            else None
        )
        if molarratio is not None:
            self.massresponse.setBaseValue(mass * molarratio / np.mean(self.detections))

        # If concentration defined use conc method
        concentration = self.concentration.baseValue()
        uptake = self.options.uptake.baseValue()
        time = self.timeAsSeconds()
        if all(o is not None for o in [concentration, uptake, time]):
            efficiency = nanopart.nebulisation_efficiency_from_concentration(
                self.detections.size,
                concentration=concentration,
                mass=mass,
                flow=uptake,
                time=time,
            )
            self.efficiency.setText(f"{efficiency:.4g}")
            return

        # Else use the other method
        dwell = self.options.dwelltime.baseValue()
        response = self.options.response.baseValue()
        if all(o is not None for o in [dwell, response, molarratio, uptake]):
            efficiencies = nanopart.nebulisation_efficiency_from_mass(
                self.detections,
                dwell=dwell,
                mass=mass,
                flowrate=uptake,
                response_factor=response,
                molar_ratio=molarratio,
            )
            efficiency = np.mean(efficiencies)
            self.efficiency.setText(f"{efficiency:.4g}")

    def elementChanged(self, text: str) -> None:
        if text in npdata.data:
            density, mw, mr = npdata.data[text]
            self.element.setValid(True)
            self.density.setValue(density)
            self.density.setUnit("g/cm³")
            self.density.setEnabled(False)
            self.molarratio.setText(str(mr))
            self.molarratio.setEnabled(False)
        else:
            self.element.setValid(False)
            self.density.setEnabled(True)
            self.molarratio.setEnabled(True)
