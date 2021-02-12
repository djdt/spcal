from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtCharts import QtCharts

import numpy as np

import nanopart

from nanopart.gui.charts import ParticleHistogram
from nanopart.gui.npoptions import NPOptionsWidget
from nanopart.gui.npinputs import NPSampleWidget
from nanopart.gui.units import UnitsWidget


class NPResultsWidget(QtWidgets.QWidget):
    def __init__(
        self,
        options: NPOptionsWidget,
        sample: NPSampleWidget,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)

        self.options = options
        self.sample = sample

        concentration_units = {
            "fg/L": 1e-18,
            "pg/L": 1e-15,
            "ng/L": 1e-12,
            "μg/L": 1e-9,
            "mg/L": 1e-6,
            "g/L": 1e-3,
            "kg/L": 1.0,
        }

        self.masses: np.ndarray = None
        self.sizes: np.ndarray = None
        self.number_concentration = 0.0
        self.concentration = 0.0
        self.ionic_background = 0.0
        self.background_lod_mass = 0.0
        self.background_lod_size = 0.0

        self.chart = ParticleHistogram()
        self.chart.drawVerticalLines(
            [0, 0],
            # [0, 0, 0],
            colors=[
                QtGui.QColor(255, 0, 0),
                QtGui.QColor(0, 0, 255),
                # QtGui.QColor(0, 255, 0),
            ],
            names=["mean", "median", "lod"],
            # names=["mean", "median", "lod"],
            styles=[QtCore.Qt.DashLine] * 2,
            # styles=[QtCore.Qt.DashLine] * 3,
        )
        self.chartview = QtCharts.QChartView(self.chart)

        self.method = QtWidgets.QComboBox()
        self.method.addItem(["Sizes", "Masses"])

        self.outputs = QtWidgets.QGroupBox("outputs")
        self.outputs.setLayout(QtWidgets.QFormLayout())

        self.count = QtWidgets.QLineEdit()
        self.count.setReadOnly(True)
        self.number = UnitsWidget(
            {"#/ml": 1e3, "#/L": 1.0}, default_unit="#/L", update_value_with_unit=True
        )
        self.number.setReadOnly(True)
        self.conc = UnitsWidget(
            concentration_units, default_unit="ng/L", update_value_with_unit=True
        )
        self.conc.setReadOnly(True)
        self.background = UnitsWidget(
            concentration_units, default_unit="ng/L", update_value_with_unit=True
        )
        self.background.setReadOnly(True)

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
        self.chart.setVerticalLines(
            [
                np.mean(self.sizes) * 1e9,
                np.median(self.sizes) * 1e9,
                # self.background_lod_size * 1e9,
            ]
        )

    def updateResults(self) -> None:
        dwelltime = self.options.dwelltime.baseValue()
        uptake = self.options.uptake.baseValue()
        response = self.options.response.baseValue()
        efficiency = float(self.options.efficiency.text())

        time = self.sample.timeAsSeconds()
        density = self.sample.density.baseValue()
        molarratio = float(self.sample.molarratio.text())

        self.masses = nanopart.particle_mass(
            self.sample.detections,
            dwell=dwelltime,
            efficiency=efficiency,
            flowrate=uptake,
            response_factor=response,
            mass_fraction=molarratio,
        )
        self.sizes = nanopart.particle_size(self.masses, density=density)
        self.number_concentration = nanopart.particle_number_concentration(
            self.sample.detections.size,
            efficiency=efficiency,
            flowrate=uptake,
            time=time,
        )
        self.concentration = nanopart.particle_total_concentration(
            self.masses,
            efficiency=efficiency,
            flowrate=uptake,
            time=time,
        )

        self.ionic_background = self.sample.background / response
        self.background_lod_mass = nanopart.particle_mass(
            self.sample.limits[3],
            dwell=dwelltime,
            efficiency=efficiency,
            flowrate=uptake,
            response_factor=response,
            mass_fraction=molarratio,
        )
        self.background_lod_size = nanopart.particle_size(
            self.background_lod_mass, density=density
        )

        self.count.setText(f"{self.sample.detections.size}")
        self.number.setBaseValue(self.number_concentration)
        self.conc.setBaseValue(self.concentration)
        self.background.setBaseValue(self.ionic_background)

        self.updateChart()
