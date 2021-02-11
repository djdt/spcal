import numpy as np

from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtCharts import QtCharts

import nanopart
from nanopart.io import read_nanoparticle_file

from nanopart.gui.charts import ParticleChart, ParticleResultsChart
from nanopart.gui.widgets import RangeSlider, ValidColorLineEdit
from nanopart.gui.units import UnitsWidget

from nanopart.gui.npoptions import NPOptionsWidget
from nanopart.gui.npinputs import NPSampleWidget, NPReferenceWidget

from typing import Dict

# class ParticleReferenceWidget(ParticleWidget):
#     def __init__(self, parent: QtWidgets.QWidget = None):
#         super().__init__(parent)
#         self.efficiency.setReadOnly(True)

#         self.concentration = UnitsWidget(units=concentration_units, unit="ng/L")
#         self.diameter = UnitsWidget(units=, unit="nm")

#         self.concentration.le.setToolTip("The concentration of the reference used.")
#         self.diameter.le.setToolTip("The diameter of the reference particle used.")

#         self.inputs.layout().addRow("Concentration:", self.concentration)
#         self.inputs.layout().addRow("Diameter:", self.diameter)

#         self.uptake.changed.connect(self.calculateEfficiency)
#         self.density.changed.connect(self.calculateEfficiency)
#         self.concentration.changed.connect(self.calculateEfficiency)
#         self.diameter.changed.connect(self.calculateEfficiency)

#         self.count = QtWidgets.QLineEdit()
#         self.count.setReadOnly(True)
#         self.outputs.layout().addRow("Particle Count:", self.count)

#         self.count.textChanged.connect(self.calculateEfficiency)

#     def isComplete(self) -> bool:
#         return (
#             super().isComplete()
#             and self.concentration.le.hasAcceptableInput()
#             and self.diameter.le.hasAcceptableInput()
#         )

#     def canCalculateEfficiency(self) -> bool:
#         return (
#             self.uptake.le.hasAcceptableInput()
#             and self.density.le.hasAcceptableInput()
#             and self.concentration.le.hasAcceptableInput()
#             and self.diameter.le.hasAcceptableInput()
#             and self.count.text() != ""
#             and int(self.count.text()) > 0
#         )

#     def calculateEfficiency(self) -> None:
#         if not self.canCalculateEfficiency():
#             return

#         uptake = self.uptake.baseValue()
#         count = int(self.count.text())
#         conc = self.concentration.baseValue()
#         mass = nanopart.reference_particle_mass(
#             self.density.baseValue(), self.diameter.baseValue()
#         )
#         time = self.timeAsSeconds()
#         efficiency = nanopart.nebulisation_efficiency(count, conc, mass, uptake, time)
#         self.efficiency.setText(f"{efficiency:.4f}")


# class ParticleResultsWidget(QtWidgets.QWidget):
#     def __init__(self, parent: QtWidgets.QWidget = None):
#         super().__init__(parent)

#         self.masses: np.ndarray = None
#         self.sizes: np.ndarray = None
#         self.number_concentration = 0.0
#         self.concentration = 0.0
#         self.ionic_background = 0.0
#         self.background_lod_mass = 0.0
#         self.background_lod_size = 0.0

#         self.chart = ParticleResultsChart()
#         self.chartview = QtCharts.QChartView(self.chart)

#         self.outputs = QtWidgets.QGroupBox("outputs")
#         self.outputs.setLayout(QtWidgets.QFormLayout())

#         self.count = QtWidgets.QLineEdit()
#         self.count.setReadOnly(True)
#         self.number = UnitsWidget(
#             units={"#/ml": 1e3, "#/L": 1.0}, unit="#/L", update_value_with_unit=True
#         )
#         self.number.le.setReadOnly(True)
#         self.conc = UnitsWidget(
#             units=concentration_units, unit="ng/L", update_value_with_unit=True
#         )
#         self.conc.le.setReadOnly(True)
#         self.background = UnitsWidget(
#             units=concentration_units, unit="ng/L", update_value_with_unit=True
#         )
#         self.background.le.setReadOnly(True)

#         self.outputs.layout().addRow("Detected particles:", self.count)
#         self.outputs.layout().addRow("Number concentration:", self.number)
#         self.outputs.layout().addRow("Concentration:", self.conc)
#         self.outputs.layout().addRow("Ionic Background:", self.background)

#         layout = QtWidgets.QVBoxLayout()
#         layout.addWidget(self.outputs)
#         layout.addWidget(self.chartview)
#         self.setLayout(layout)

#     def updateChart(self) -> None:
#         self.chart.setData(self.sizes * 1e9, bins=128)
#         self.chart.setLines(np.mean(self.sizes) * 1e9, np.median(self.sizes) * 1e9)

#     def updateForSample(
#         self,
#         detections: np.ndarray,
#         background: float,
#         limit_detection: float,
#         params: Dict[str, float],
#     ) -> None:
#         self.masses = nanopart.particle_mass(
#             detections,
#             dwell=params["dwelltime"],
#             efficiency=params["efficiency"],
#             flowrate=params["uptake"],
#             response_factor=params["response"],
#             mass_fraction=params["molarratio"],
#         )
#         self.sizes = nanopart.particle_size(self.masses, density=params["density"])
#         self.number_concentration = nanopart.particle_number_concentration(
#             detections.size,
#             efficiency=params["efficiency"],
#             flowrate=params["uptake"],
#             time=params["time"],
#         )
#         self.concentration = nanopart.particle_total_concentration(
#             self.masses,
#             efficiency=params["efficiency"],
#             flowrate=params["uptake"],
#             time=params["time"],
#         )

#         self.ionic_background = background / params["response"]
#         self.background_lod_mass = nanopart.particle_mass(
#             limit_detection,
#             dwell=params["dwelltime"],
#             efficiency=params["efficiency"],
#             flowrate=params["uptake"],
#             response_factor=params["response"],
#             mass_fraction=params["molarratio"],
#         )
#         self.background_lod_size = nanopart.particle_size(
#             self.background_lod_mass, density=params["density"]
#         )

#         self.count.setText(f"{detections.size}")
#         self.number.setBaseValue(self.number_concentration)
#         self.conc.setBaseValue(self.concentration)
#         self.background.setBaseValue(self.ionic_background)


class NanoPartWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.tabs = QtWidgets.QTabWidget()
        # self.tabs.currentChanged.connect(self.onTabChanged)

        self.options = NPOptionsWidget()
        self.sample = NPSampleWidget(self.options)
        self.reference = NPReferenceWidget(self.options)
        # self.results = ParticleResultsWidget()

        self.reference.efficiency.textChanged.connect(self.options.setEfficiency)

        # self.options.optionsChanged.connect(self.sample.setOptions)
        # self.sample.completeChanged.connect(self.onSampleComplete)

        # Sync instrument wide parameters
        # self.sample.uptake.sync(self.reference.uptake)
        # self.sample.dwelltime.sync(self.reference.dwelltime)
        # self.sample.response.sync(self.reference.response)
        # self.sample.density.sync(self.reference.density)
        # self.reference.efficiency.textChanged.connect(self.sample.efficiency.setText)

        self.tabs.addTab(self.options, "Options")
        self.tabs.addTab(self.sample, "Sample")
        self.tabs.addTab(self.reference, "Reference")
        # self.tabs.addTab(self.results, "Results")
        # self.tabs.setTabEnabled(self.tabs.indexOf(self.results), False)

        widget = QtWidgets.QWidget()

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.tabs, 1)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    # def onSampleComplete(self) -> None:
    #     complete = self.sample.isComplete()
    #     self.tabs.setTabEnabled(self.tabs.indexOf(self.results), complete)

    # def onTabChanged(self, index: int) -> None:
        # if self.tabs.widget(index) == self.results:
        #     if self.sample.isComplete():
        #         self.results.updateForSample(
        #             self.sample.detections,
        #             self.sample.true_background,
        #             self.sample.poisson_limits[2],
        #             self.sample.parameters(),
        #         )
        #         self.results.updateChart()
