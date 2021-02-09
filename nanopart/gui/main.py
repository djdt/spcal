import numpy as np

from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtCharts import QtCharts

import nanopart
from nanopart.io import read_nanoparticle_file

from nanopart.gui.charts import ParticleChart, ParticleResultsChart
from nanopart.gui.util import ParticleModel
from nanopart.gui.widgets import RangeSlider, UnitsWidget, ValidColorLineEdit

from typing import Dict

concentration_units = {
    "fg/L": 1e-18,
    "pg/L": 1e-15,
    "ng/L": 1e-12,
    "μg/L": 1e-9,
    "mg/L": 1e-6,
    "g/L": 1e-3,
    "kg/L": 1.0,
}
density_units = {"g/cm³": 1e-3 * 1e6, "kg/m³": 1.0}
response_units = {
    "counts/(pg/L)": 1e15,
    "counts/(ng/L)": 1e12,
    "counts/(μg/L)": 1e9,
    "counts/(mg/L)": 1e6,
}
uptake_units = {"ml/min": 1e-3 / 60.0, "ml/s": 1e-3, "L/min": 1.0 / 60.0, "L/s": 1.0}


class ParticleTable(QtWidgets.QWidget):
    unitChanged = QtCore.Signal(str)

    def __init__(self, model: ParticleModel, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.table = QtWidgets.QTableView()
        self.table.setModel(model)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)

        self.response = QtWidgets.QComboBox()
        self.response.addItems(["counts", "cps"])
        self.response.currentTextChanged.connect(self.unitChanged)

        layout_unit = QtWidgets.QHBoxLayout()
        layout_unit.addWidget(QtWidgets.QLabel("Response units:"), 1)
        layout_unit.addWidget(self.response, 0)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.table)
        layout.addLayout(layout_unit)
        self.setLayout(layout)


class ParticleWidget(QtWidgets.QWidget):
    completeChanged = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.setAcceptDrops(True)

        self.detections: np.ndarray = None
        # self.regions: np.ndarray = None
        self.limits = (0.0, 0.0, 0.0)
        self.true_background = 0.0

        self.button_file = QtWidgets.QPushButton("Open File")
        self.button_file.pressed.connect(self.dialogLoadfile)

        self.chart = ParticleChart()
        self.chartview = QtCharts.QChartView(self.chart)
        # self.chartview.setRenderHint(QtGui.QPainter.Antialiasing)
        self.chartview.setRubberBand(QtCharts.QChartView.HorizontalRubberBand)

        self.model = ParticleModel(np.ndarray((0, 1), dtype=np.float64))
        self.model.dataChanged.connect(self.updateOutputs)
        self.model.dataChanged.connect(self.updateChartData)

        self.table = ParticleTable(self.model)
        self.table.unitChanged.connect(self.updateOutputs)
        self.table.unitChanged.connect(self.updateChartLines)

        self.slider = RangeSlider()
        self.slider.setRange(0, 100)
        self.slider.valueChanged.connect(self.updateChartTrim)
        self.slider.value2Changed.connect(self.updateChartTrim)
        self.slider.sliderReleased.connect(self.updateOutputs)
        self.slider.sliderReleased.connect(self.updateChartLines)

        # Instrument wide settings
        self.dwelltime = UnitsWidget(units={"ms": 1e-3, "s": 1.0}, unit="s")
        self.uptake = UnitsWidget(units=uptake_units, unit="ml/min")
        self.response = UnitsWidget(units=response_units, unit="counts/(μg/L)")
        self.density = UnitsWidget(units=density_units, unit="g/cm³")
        self.efficiency = ValidColorLineEdit()
        self.efficiency.setValidator(QtGui.QDoubleValidator(0.0, 1.0, 4))

        self.dwelltime.le.setToolTip(
            "ICP-MS dwell-time. Read from imported file if time column exists."
        )
        self.uptake.le.setToolTip("ICP-MS sample flowrate.")
        self.response.le.setToolTip("ICP-MS response for ionic standard.")
        self.density.le.setToolTip("Sample and reference particle density.")
        self.efficiency.setToolTip(
            "Nebulisation efficiency. Can be calculated using a reference particle."
        )

        # Complete Changed
        self.model.rowsRemoved.connect(self.completeChanged)
        self.model.rowsInserted.connect(self.completeChanged)
        self.model.modelReset.connect(self.completeChanged)
        self.dwelltime.changed.connect(self.completeChanged)
        self.dwelltime.changed.connect(self.updateOutputs)
        self.dwelltime.changed.connect(self.updateChartLines)
        self.uptake.changed.connect(self.completeChanged)
        self.response.changed.connect(self.completeChanged)
        self.density.changed.connect(self.completeChanged)
        self.efficiency.textChanged.connect(self.completeChanged)

        line = QtWidgets.QFrame()
        line.setFrameStyle(QtWidgets.QFrame.HLine)

        self.inputs = QtWidgets.QGroupBox("Inputs")
        self.inputs.setLayout(QtWidgets.QFormLayout())
        self.inputs.layout().addRow("Uptake:", self.uptake)
        self.inputs.layout().addRow("Density", self.density)
        self.inputs.layout().addRow("Dwell time:", self.dwelltime)
        self.inputs.layout().addRow("Response:", self.response)
        self.inputs.layout().addRow("Neb. Efficiency:", self.efficiency)
        self.inputs.layout().addRow(line)

        self.outputs = QtWidgets.QGroupBox("outputs")
        self.outputs.setLayout(QtWidgets.QFormLayout())

        layout_table = QtWidgets.QVBoxLayout()
        layout_table.addWidget(self.button_file, 0, QtCore.Qt.AlignLeft)
        layout_table.addWidget(self.table)

        layout_slider = QtWidgets.QHBoxLayout()
        layout_slider.addWidget(QtWidgets.QLabel("Trim:"))
        layout_slider.addWidget(self.slider, QtCore.Qt.AlignRight)

        layout_io = QtWidgets.QHBoxLayout()
        layout_io.addWidget(self.inputs)
        layout_io.addWidget(self.outputs)
        layout_chart = QtWidgets.QVBoxLayout()
        layout_chart.addLayout(layout_io)
        # layout_chart.addWidget(self.inputs, 0, QtCore.Qt.AlignRight)
        # layout_chart.addWidget(self.outputs, 0, QtCore.Qt.AlignRight)
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

    def isComplete(self) -> bool:
        return (
            self.uptake.le.hasAcceptableInput()
            and self.density.le.hasAcceptableInput()
            and self.dwelltime.le.hasAcceptableInput()
            and self.response.le.hasAcceptableInput()
            and self.efficiency.hasAcceptableInput()
            and self.model.rowCount() > 0
        )

    def timeAsSeconds(self) -> np.ndarray:
        dwell = self.dwelltime.baseValue()
        return (self.slider.right() - self.slider.left()) * dwell

    def responseAsCounts(self) -> np.ndarray:
        factors = {"counts": 1.0, "cps": self.dwelltime.baseValue()}
        responses = self.model.array[self.slider.left() : self.slider.right(), 0]
        return responses * factors[self.table.response.currentText()]

    def dialogLoadfile(self) -> None:
        file, _filter = QtWidgets.QFileDialog.getOpenFileName(self, "Open", "")
        if file != "":
            self.loadFile(file)

    def loadFile(self, file: str) -> np.ndarray:
        try:
            responses, parameters = read_nanoparticle_file(file, delimiter=",")
        except ValueError:
            return

        # Update dwell time
        if "dwelltime" in parameters:
            self.dwelltime.le.setText(str(parameters["dwelltime"]))

        self.table.response.blockSignals(True)
        self.table.response.setCurrentText("cps" if parameters["cps"] else "counts")
        self.table.response.blockSignals(True)

        self.model.beginResetModel()
        self.model.array = responses[:, None]
        self.model.endResetModel()

        # Update Chart and slider
        self.slider.setRange(0, self.model.rowCount())
        self.slider.setValues(0, self.model.rowCount())

        self.updateOutputs()
        self.updateChartData()
        self.chart.xaxis.setRange(self.slider.left(), self.slider.right())

    def updateOutputs(self) -> None:
        if not self.dwelltime.le.hasAcceptableInput():
            return

        response = self.responseAsCounts()
        if response.size == 0:
            self.limits = None
            self.detections = None
            return

        # Update the ub, lc, ld
        ub = np.nanmean(response)
        yc, yd = nanopart.poisson_limits(ub)

        self.limits = (ub, yc + ub, yd + ub)
        self.detections, regions = nanopart.accumulate_detections(response - ub, yd, yc)
        self.true_background = np.nanmean(response[regions == 0])

        self.count.setText(str(self.detections.size))

    def updateChartData(self) -> None:
        responses = self.model.array[:, 0]
        events = np.arange(responses.size)
        self.chart.setData(np.stack((events, responses), axis=1))

        self.updateChartLines()
        self.updateChartTrim()

    def updateChartLines(self) -> None:
        if not self.dwelltime.le.hasAcceptableInput() or self.limits is None:
            return

        ub, lc, ld = self.limits
        if self.table.response.currentText() == "cps":
            dwell = self.dwelltime.baseValue()
            ub /= dwell
            lc /= dwell
            ld /= dwell

        self.chart.setLines(ub, lc, ld)

    def updateChartTrim(self) -> None:
        self.chart.setTrim(self.slider.left(), self.slider.right())


class ParticleSampleWidget(ParticleWidget):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.molarratio = ValidColorLineEdit("1.0")
        self.molarratio.setValidator(QtGui.QDoubleValidator(0.0, 1.0, 6))
        self.molarratio.setToolTip(
            "Ratio of the mass of the particle to the mass of the analyte."
        )

        self.molarratio.textChanged.connect(self.completeChanged)

        self.inputs.layout().addRow("Molar ratio:", self.molarratio)

        self.count = QtWidgets.QLineEdit()
        self.count.setReadOnly(True)
        self.outputs.layout().addRow("Particle Count:", self.count)

    def isComplete(self) -> bool:
        return super().isComplete() and self.molarratio.hasAcceptableInput()

    def parameters(self) -> Dict[str, float]:
        return {
            "density": self.density.baseValue(),
            "dwelltime": self.dwelltime.baseValue(),
            "efficiency": float(self.efficiency.text()),
            "molarratio": float(self.molarratio.text()),
            "response": self.response.baseValue(),
            "time": self.timeAsSeconds(),
            "uptake": self.uptake.baseValue(),
        }


class ParticleReferenceWidget(ParticleWidget):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.efficiency.setReadOnly(True)

        self.concentration = UnitsWidget(units=concentration_units, unit="ng/L")
        self.diameter = UnitsWidget(units={"nm": 1e-9, "μm": 1e-6, "m": 1.0}, unit="nm")

        self.concentration.le.setToolTip("The concentration of the reference used.")
        self.diameter.le.setToolTip("The diameter of the reference particle used.")

        self.inputs.layout().addRow("Concentration:", self.concentration)
        self.inputs.layout().addRow("Diameter:", self.diameter)

        self.uptake.changed.connect(self.calculateEfficiency)
        self.density.changed.connect(self.calculateEfficiency)
        self.concentration.changed.connect(self.calculateEfficiency)
        self.diameter.changed.connect(self.calculateEfficiency)

        self.count = QtWidgets.QLineEdit()
        self.count.setReadOnly(True)
        self.outputs.layout().addRow("Particle Count:", self.count)

        self.count.textChanged.connect(self.calculateEfficiency)

    def isComplete(self) -> bool:
        return (
            super().isComplete()
            and self.concentration.le.hasAcceptableInput()
            and self.diameter.le.hasAcceptableInput()
        )

    def canCalculateEfficiency(self) -> bool:
        return (
            self.uptake.le.hasAcceptableInput()
            and self.density.le.hasAcceptableInput()
            and self.concentration.le.hasAcceptableInput()
            and self.diameter.le.hasAcceptableInput()
            and self.count.text() != ""
            and int(self.count.text()) > 0
        )

    def calculateEfficiency(self) -> None:
        if not self.canCalculateEfficiency():
            return

        uptake = self.uptake.baseValue()
        count = int(self.count.text())
        conc = self.concentration.baseValue()
        mass = nanopart.reference_particle_mass(
            self.density.baseValue(), self.diameter.baseValue()
        )
        time = self.timeAsSeconds()
        efficiency = nanopart.nebulisation_efficiency(count, conc, mass, uptake, time)
        self.efficiency.setText(f"{efficiency:.4f}")


class ParticleResultsWidget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)

        self.masses: np.ndarray = None
        self.sizes: np.ndarray = None
        self.number_concentration = 0.0
        self.concentration = 0.0
        self.ionic_background = 0.0
        self.background_lod_mass = 0.0
        self.background_lod_size = 0.0

        self.chart = ParticleResultsChart()
        self.chartview = QtCharts.QChartView(self.chart)

        self.outputs = QtWidgets.QGroupBox("outputs")
        self.outputs.setLayout(QtWidgets.QFormLayout())

        self.count = QtWidgets.QLineEdit()
        self.count.setReadOnly(True)
        self.number = UnitsWidget(
            units={"#/ml": 1e3, "#/L": 1.0}, unit="#/L", update_value_with_unit=True
        )
        self.number.le.setReadOnly(True)
        self.conc = UnitsWidget(
            units=concentration_units, unit="ng/L", update_value_with_unit=True
        )
        self.conc.le.setReadOnly(True)
        self.background = UnitsWidget(
            units=concentration_units, unit="ng/L", update_value_with_unit=True
        )
        self.background.le.setReadOnly(True)

        self.outputs.layout().addRow("Detected particles:", self.count)
        self.outputs.layout().addRow("Number concentration:", self.number)
        self.outputs.layout().addRow("Concentration:", self.conc)
        self.outputs.layout().addRow("Ionic Background:", self.background)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.outputs)
        layout.addWidget(self.chartview)
        self.setLayout(layout)

    def updateChart(self) -> None:
        self.chart.setData(self.sizes * 1e9, bins=128)
        self.chart.setLines(np.mean(self.sizes) * 1e9, np.median(self.sizes) * 1e9)

    def updateForSample(
        self,
        detections: np.ndarray,
        background: float,
        limit_detection: float,
        params: Dict[str, float],
    ) -> None:
        self.masses = nanopart.particle_mass(
            detections,
            dwell=params["dwelltime"],
            efficiency=params["efficiency"],
            flowrate=params["uptake"],
            response_factor=params["response"],
            mass_fraction=params["molarratio"],
        )
        self.sizes = nanopart.particle_size(self.masses, density=params["density"])
        self.number_concentration = nanopart.particle_number_concentration(
            detections.size,
            efficiency=params["efficiency"],
            flowrate=params["uptake"],
            time=params["time"],
        )
        self.concentration = nanopart.particle_total_concentration(
            self.masses,
            efficiency=params["efficiency"],
            flowrate=params["uptake"],
            time=params["time"],
        )

        self.ionic_background = background / params["response"]
        self.background_lod_mass = nanopart.particle_mass(
            limit_detection,
            dwell=params["dwelltime"],
            efficiency=params["efficiency"],
            flowrate=params["uptake"],
            response_factor=params["response"],
            mass_fraction=params["molarratio"],
        )
        self.background_lod_size = nanopart.particle_size(
            self.background_lod_mass, density=params["density"]
        )

        self.count.setText(f"{detections.size}")
        self.number.setBaseValue(self.number_concentration)
        self.conc.setBaseValue(self.concentration)
        self.background.setBaseValue(self.ionic_background)


class NanoPartWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.currentChanged.connect(self.onTabChanged)

        self.sample = ParticleSampleWidget()
        self.reference = ParticleReferenceWidget()
        self.results = ParticleResultsWidget()

        self.sample.completeChanged.connect(self.onSampleComplete)

        # Sync instrument wide parameters
        self.sample.uptake.sync(self.reference.uptake)
        self.sample.dwelltime.sync(self.reference.dwelltime)
        self.sample.response.sync(self.reference.response)
        self.sample.density.sync(self.reference.density)
        self.reference.efficiency.textChanged.connect(self.sample.efficiency.setText)

        self.tabs.addTab(self.sample, "Sample")
        self.tabs.addTab(self.reference, "Reference")
        self.tabs.addTab(self.results, "Results")
        self.tabs.setTabEnabled(self.tabs.indexOf(self.results), False)

        widget = QtWidgets.QWidget()

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.tabs, 1)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def onSampleComplete(self) -> None:
        complete = self.sample.isComplete()
        self.tabs.setTabEnabled(self.tabs.indexOf(self.results), complete)

    def onTabChanged(self, index: int) -> None:
        if self.tabs.widget(index) == self.results:
            if self.sample.isComplete():
                self.results.updateForSample(
                    self.sample.detections,
                    self.sample.true_background,
                    self.sample.limits[2],
                    self.sample.parameters(),
                )
                self.results.updateChart()
