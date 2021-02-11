

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
