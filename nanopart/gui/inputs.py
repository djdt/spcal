from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtCharts import QtCharts

import numpy as np
from pathlib import Path

import nanopart
from nanopart import npdata

from nanopart.calc import calculate_limits
from nanopart.io import read_nanoparticle_file

from nanopart.gui.charts import ParticleChart, ParticleChartView
from nanopart.gui.options import OptionsWidget
from nanopart.gui.tables import ParticleTable
from nanopart.gui.units import UnitsWidget
from nanopart.gui.widgets import (
    ElidedLabel,
    RangeSlider,
    ValidColorLineEdit,
)

from typing import Tuple


# todo changing values in table doesnt update linits


class InputWidget(QtWidgets.QWidget):
    optionsChanged = QtCore.Signal()
    detectionsChanged = QtCore.Signal(int)
    limitsChanged = QtCore.Signal()

    def __init__(self, options: OptionsWidget, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.setAcceptDrops(True)

        self.redraw_charts_requested = False

        # self.detectionsChanged.connect(self.redrawChart)
        self.limitsChanged.connect(self.updateDetections)
        self.limitsChanged.connect(self.requestRedraw)

        self.options = options
        self.options.dwelltime.valueChanged.connect(self.updateLimits)
        self.options.method.currentTextChanged.connect(self.updateLimits)
        self.options.window_size.editingFinished.connect(self.updateLimits)
        self.options.check_use_window.toggled.connect(self.updateLimits)
        self.options.epsilon.editingFinished.connect(self.updateLimits)
        self.options.sigma.editingFinished.connect(self.updateLimits)

        self.background = 0.0
        self.background_std = 0.0
        self.detections = np.array([], dtype=np.float64)
        self.detections_std = 0.0
        self.limits: Tuple[str, float, float, float] = None

        self.button_file = QtWidgets.QPushButton("Open File")
        self.button_file.pressed.connect(self.dialogLoadFile)

        self.label_file = ElidedLabel()
        self.label_file.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored
        )

        self.chart = ParticleChart()
        self.chartview = ParticleChartView(self.chart)
        self.chartview.setRubberBand(QtCharts.QChartView.HorizontalRubberBand)
        self.chartview.setAcceptDrops(False)

        self.table_units = QtWidgets.QComboBox()
        self.table_units.addItems(["Counts", "CPS"])
        self.table_units.currentTextChanged.connect(self.updateLimits)

        self.table = ParticleTable()
        self.table.model().dataChanged.connect(self.updateLimits)

        self.slider = RangeSlider()
        self.slider.setRange(0, 1)
        self.slider.setValues(0, 1)
        self.slider.valueChanged.connect(self.updateTrim)
        self.slider.value2Changed.connect(self.updateTrim)
        self.slider.sliderReleased.connect(self.updateLimits)

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
        layout_table_file.addWidget(self.button_file, 0, QtCore.Qt.AlignLeft)
        layout_table_file.addWidget(self.label_file, 1)

        layout_table_units = QtWidgets.QHBoxLayout()
        layout_table_units.addStretch(1)
        layout_table_units.addWidget(QtWidgets.QLabel("Response:"), 0)
        layout_table_units.addWidget(self.table_units, 0)

        layout_table = QtWidgets.QVBoxLayout()
        layout_table.addLayout(layout_table_units, 0)
        layout_table.addWidget(self.table, 1)

        layout_slider = QtWidgets.QHBoxLayout()
        layout_slider.addWidget(QtWidgets.QLabel("Trim:"))
        layout_slider.addWidget(self.slider, QtCore.Qt.AlignRight)

        layout_io = QtWidgets.QHBoxLayout()
        layout_io.addWidget(self.inputs)
        layout_io.addWidget(self.outputs)

        layout_chart = QtWidgets.QVBoxLayout()
        layout_chart.addLayout(layout_table_file, 0)
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
                break
            event.acceptProposedAction()
        elif event.mimeData().hasText():
            text = event.mimeData().text()
            data = np.genfromtxt(text.split("\n"), usecols=0, dtype=np.float64)
            data = data[~np.isnan(data)]
            if data.size == 0:
                event.ignore()
                return

            self.loadData(data)
            event.acceptProposedAction()
        elif event.mimeData().hasHtml():
            pass
        else:
            super().dropEvent(event)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        if self.redraw_charts_requested:
            self.redrawChart()
            self.redrawLimits()
            self.redraw_charts_requested = False

    def numberOfEvents(self) -> int:
        return self.slider.right() - self.slider.left()

    def responseAsCounts(self, trim: Tuple[int, int] = None) -> np.ndarray:
        if trim is None:
            trim = (self.slider.left(), self.slider.right())

        dwelltime = self.options.dwelltime.baseValue()
        response = self.table.model().array[trim[0] : trim[1], 0]

        if self.table_units.currentText() == "Counts":
            return response
        elif dwelltime is not None:
            return response * dwelltime
        else:
            return None

    def timeAsSeconds(self) -> float:
        dwell = self.options.dwelltime.baseValue()
        if dwell is None:
            return None
        return (self.slider.right() - self.slider.left()) * dwell

    def dialogLoadFile(self) -> None:
        file, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open",
            "",
            "CSV Documents(*.csv *.txt *.text);;All files(*)",
        )
        if file != "":
            self.loadFile(file)

    def loadFile(self, file: str) -> None:
        try:
            responses, parameters = read_nanoparticle_file(file, delimiter=",")
        except ValueError:
            return

        self.label_file.setText(Path(file).name)

        self.table_units.blockSignals(True)
        self.table_units.setCurrentText("CPS" if parameters["cps"] else "Counts")
        self.table_units.blockSignals(False)

        # Update dwell time
        if "dwelltime" in parameters:
            self.options.dwelltime.setBaseValue(parameters["dwelltime"])

        self.loadData(responses)

    def loadData(self, data: np.ndarray) -> None:
        self.table.model().beginResetModel()
        self.table.model().array = data[:, None]
        self.table.model().endResetModel()

        # Update Chart and slider
        offset = self.slider.maximum() - self.slider.right()
        self.slider.setRange(0, self.table.model().rowCount())

        right = max(self.slider.maximum() - offset, 1)
        left = min(self.slider.left(), right - 1)
        self.slider.setValues(left, right)
        self.chart.xaxis.setRange(self.slider.minimum(), self.slider.maximum())

        self.updateLimits()

    def updateDetections(self) -> None:
        responses = self.responseAsCounts()

        if self.limits is None or responses is None or responses.size == 0:
            self.detections = np.array([])
            self.background_std = 0.0
            self.detections_std = 0.0

            self.count.setText("")
            self.background_count.setText("")
            self.lod_count.setText("")
        else:
            detections, labels, regions = nanopart.accumulate_detections(
                responses, self.limits[2], self.limits[3], return_regions=True
            )
            centers = (regions[:, 0] + regions[:, 1]) // 2
            self.centers = centers
            values = np.linspace(0, responses.size, 3 + 1)
            indicies = np.searchsorted(self.centers, values, side="left")

            self.detections = detections
            self.detections_std = np.std(np.diff(indicies))
            self.background = np.nanmean(responses[labels == 0])
            self.background_std = np.nanstd(responses[labels == 0])
            lod = np.mean(self.limits[2])  # + self.background

            self.count.setText(f"{detections.size} ± {self.detections_std:.1f}")
            self.background_count.setText(
                f"{self.background:.4g} ± {self.background_std:.4g}"
            )
            self.lod_count.setText(f"{lod:.4g} ({self.limits[0]})")

        self.detectionsChanged.emit(self.detections.size)

    def updateLimits(self) -> None:
        method = self.options.method.currentText()
        responses = self.responseAsCounts()
        sigma = (
            float(self.options.sigma.text())
            if self.options.sigma.hasAcceptableInput()
            else None
        )
        epsilon = (
            float(self.options.epsilon.text())
            if self.options.epsilon.hasAcceptableInput()
            else None
        )
        window_size = (
            int(self.options.window_size.text())
            if self.options.window_size.hasAcceptableInput()
            and self.options.window_size.isEnabled()
            else None
        )

        self.limits = calculate_limits(
            responses, method, sigma, epsilon, window=window_size
        )
        self.limitsChanged.emit()

    def redrawChart(self) -> None:
        responses = self.responseAsCounts(trim=(0, self.table.model().rowCount()))
        if responses is None or responses.size == 0:
            return

        centers = self.centers + self.slider.left()

        self.chart.setData(np.nan_to_num(responses))
        self.chart.setScatter(centers, responses[centers])

        self.chart.drawVerticalLines(
            [self.slider.left(), self.slider.right()],
            pens=[
                QtGui.QPen(QtGui.QColor(255, 0, 0), 2.0),
                QtGui.QPen(QtGui.QColor(255, 0, 0), 2.0),
            ],
            visible_in_legend=[False, False],  # type: ignore
        )
        self.chart.updateYRange()

    def redrawLimits(self) -> None:
        if self.limits is None:
            self.chart.ub.clear()
            self.chart.lc.clear()
            self.chart.ld.clear()
            return

        xs = np.arange(self.slider.left(), self.slider.right())

        self.chart.setBackground(xs, self.limits[1])
        if self.limits[0] == "Poisson":
            self.chart.setLimitCritical(xs, self.limits[2])
        else:
            self.chart.lc.clear()
        self.chart.setLimitDetection(xs, self.limits[3])
        self.chart.updateGeometry()

    def requestRedraw(self) -> None:
        if self.isVisible():
            self.redrawChart()
            self.redrawLimits()
        else:
            self.redraw_charts_requested = True

    def updateTrim(self) -> None:
        values = [self.slider.left(), self.slider.right()]
        self.chart.setVerticalLines(values)  # type: ignore

    def resetInputs(self) -> None:
        self.blockSignals(True)
        self.slider.setRange(0, 100)
        self.slider.setValues(0, 100)
        self.count.setText("0")
        self.background_count.setText("")
        self.lod_count.setText("")

        self.background = 0.0
        self.background_std = 0.0
        self.detections = np.array([], dtype=np.float64)
        self.detections_std = 0.0
        self.limits = None

        self.table.model().beginResetModel()
        self.table.model().array = np.empty((0, 1), dtype=float)
        self.table.model().endResetModel()
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

    def resetInputs(self) -> None:
        self.blockSignals(True)
        self.element.setText("")
        self.density.setValue(None)
        self.molarmass.setValue(None)
        self.molarratio.setText("1.0")
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
                flowrate=uptake,
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

    def isComplete(self) -> bool:
        return (
            self.detections is not None
            and self.detections.size > 0
            and self.diameter.hasAcceptableInput()
            and self.molarratio.hasAcceptableInput()
            and self.density.hasAcceptableInput()
        )

    def resetInputs(self) -> None:
        self.blockSignals(True)
        self.element.setText("")
        self.diameter.setValue(None)
        self.density.setValue(None)
        self.molarratio.setText("1.0")
        self.concentration.setValue(None)

        self.efficiency.setText("")
        self.massresponse.setValue(None)
        self.blockSignals(False)
        super().resetInputs()
