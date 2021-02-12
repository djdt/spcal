from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtCharts import QtCharts

import numpy as np

import nanopart

from nanopart.gui.charts import ParticleHistogram
from nanopart.gui.npoptions import NPOptionsWidget
from nanopart.gui.npinputs import NPSampleWidget
from nanopart.gui.units import UnitsWidget

from typing import Dict


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
        self.chartview = QtCharts.QChartView(self.chart)

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
        self.chart.setLines(np.mean(self.sizes) * 1e9, np.median(self.sizes) * 1e9)

    def updateResults(self) -> None:
        dwelltime = self.options.dwelltime.baseValue()
        uptake = self.options.uptake.baseValue()
        response = self.options.response.baseValue()
        efficiency = float(self.options.efficiency)

        time = self.sample.timeAsSeconds()
        density = self.sample.density.baseValue()
        molarratio = float(self.sample.molarratio)

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
            detections.size,
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

        self.ionic_background = background / response
        self.background_lod_mass = nanopart.particle_mass(
            limit_detection,
            dwell=dwelltime,
            efficiency=efficiency,
            flowrate=uptake,
            response_factor=response,
            mass_fraction=molarratio,
        )
        self.background_lod_size = nanopart.particle_size(
            self.background_lod_mass, density=density
        )

        self.count.setText(f"{detections.size}")
        self.number.setBaseValue(self.number_concentration)
        self.conc.setBaseValue(self.concentration)
        self.background.setBaseValue(self.ionic_background)
