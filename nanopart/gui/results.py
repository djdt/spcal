from PySide2 import QtCore, QtGui, QtWidgets

import numpy as np
from pathlib import Path

from nanopart.calc import (
    results_from_mass_response,
    results_from_nebulisation_efficiency,
)
from nanopart.fit import fit_normal, fit_lognormal
from nanopart.io import export_nanoparticle_results

from nanopart.gui.charts import ParticleHistogram, ParticleChartView
from nanopart.gui.inputs import SampleWidget, ReferenceWidget
from nanopart.gui.options import OptionsWidget
from nanopart.gui.tables import ResultsTable
from nanopart.gui.units import UnitsWidget


class ResultsWidget(QtWidgets.QWidget):
    def __init__(
        self,
        options: OptionsWidget,
        sample: SampleWidget,
        reference: ReferenceWidget,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)

        self.options = options
        self.sample = sample
        self.reference = reference

        concentration_units = {
            "fg/L": 1e-18,
            "pg/L": 1e-15,
            "ng/L": 1e-12,
            "μg/L": 1e-9,
            "mg/L": 1e-6,
            "g/L": 1e-3,
            "kg/L": 1.0,
        }

        self.nbins = "auto"
        self.result = {}

        self.chart = ParticleHistogram()
        self.chart.drawVerticalLines(
            [0, 0, 0],
            names=["mean", "median", "lod"],
            pens=[
                QtGui.QPen(QtGui.QColor(255, 0, 0), 1.5, QtCore.Qt.DashLine),
                QtGui.QPen(QtGui.QColor(0, 0, 255), 1.5, QtCore.Qt.DashLine),
                QtGui.QPen(QtGui.QColor(0, 172, 0), 1.5, QtCore.Qt.DashLine),
            ],
        )
        self.chartview = ParticleChartView(self.chart)
        self.chartview.setRenderHint(QtGui.QPainter.Antialiasing)

        self.table = ResultsTable()

        self.fitmethod = QtWidgets.QComboBox()
        self.fitmethod.addItems(["None", "Normal", "Lognormal"])
        self.fitmethod.setCurrentText("Lognormal")

        self.fitmethod.currentIndexChanged.connect(self.updateChartFit)

        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["Signal", "Mass", "Size"])
        self.mode.setCurrentText("Size")
        self.mode.currentIndexChanged.connect(self.updateTable)
        self.mode.currentIndexChanged.connect(self.updateChart)

        self.outputs = QtWidgets.QGroupBox("Outputs")
        self.outputs.setLayout(QtWidgets.QHBoxLayout())

        self.count = QtWidgets.QLineEdit()
        self.count.setReadOnly(True)
        self.number = UnitsWidget(
            {"#/L": 1.0, "#/ml": 1e3}, default_unit="#/L", update_value_with_unit=True
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

        self.lod = UnitsWidget(
            {"nm": 1e-9, "μm": 1e-6, "m": 1.0},
            default_unit="nm",
            update_value_with_unit=True,
        )
        self.lod.setReadOnly(True)
        self.mean = UnitsWidget(
            {"nm": 1e-9, "μm": 1e-6, "m": 1.0},
            default_unit="nm",
            update_value_with_unit=True,
        )
        self.mean.setReadOnly(True)
        self.median = UnitsWidget(
            {"nm": 1e-9, "μm": 1e-6, "m": 1.0},
            default_unit="nm",
            update_value_with_unit=True,
        )
        self.median.setReadOnly(True)

        layout_outputs_left = QtWidgets.QFormLayout()
        layout_outputs_left.addRow("Detected particles:", self.count)
        layout_outputs_left.addRow("Number concentration:", self.number)
        layout_outputs_left.addRow("Concentration:", self.conc)
        layout_outputs_left.addRow("Ionic Background:", self.background)

        layout_outputs_right = QtWidgets.QFormLayout()
        layout_outputs_right.addRow("Mean size:", self.mean)
        layout_outputs_right.addRow("Median size:", self.median)
        layout_outputs_right.addRow("Size LOD:", self.lod)

        self.outputs.layout().addLayout(layout_outputs_left)
        self.outputs.layout().addLayout(layout_outputs_right)

        self.button_export = QtWidgets.QPushButton("Export")
        self.button_export.pressed.connect(self.dialogExportResults)

        layout_table_options = QtWidgets.QHBoxLayout()
        layout_table_options.addStretch(1)
        layout_table_options.addWidget(QtWidgets.QLabel("Mode:"), 0)
        layout_table_options.addWidget(self.mode)

        layout_table = QtWidgets.QVBoxLayout()
        layout_table.addLayout(layout_table_options)
        layout_table.addWidget(self.table, 1)
        layout_table.addWidget(self.button_export, 0, QtCore.Qt.AlignRight)

        layout_chart_options = QtWidgets.QHBoxLayout()
        layout_chart_options.addStretch(1)
        layout_chart_options.addWidget(QtWidgets.QLabel("Fit:"), 0)
        layout_chart_options.addWidget(self.fitmethod)

        layout_outputs = QtWidgets.QVBoxLayout()
        layout_outputs.addWidget(self.outputs)
        layout_outputs.addWidget(self.chartview)
        layout_outputs.addLayout(layout_chart_options)

        layout = QtWidgets.QHBoxLayout()
        layout.addLayout(layout_table, 0)
        layout.addLayout(layout_outputs, 1)
        self.setLayout(layout)

    def dialogExportResults(self) -> None:
        file, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export", "", "CSV Documents (.csv)"
        )
        if file != "":
            export_nanoparticle_results(Path(file), self.result)

    def updateChart(self) -> None:
        mode = self.mode.currentText()
        if mode == "Mass":
            data = self.result["masses"] * 1e21  # ag
            lod = self.result["lod_mass"] * 1e21
            self.chart.xaxis.setTitleText("Mass (ag)")
        elif mode == "Signal":
            data = self.result["detections"]  # counts
            lod = self.result["lod"]
            self.chart.xaxis.setTitleText("Signal (counts)")
        elif mode == "Size":
            data = self.result["sizes"] * 1e9  # nm
            lod = self.result["lod_size"] * 1e9
            self.chart.xaxis.setTitleText("Size (nm)")

        # Crush the LOD for chart
        # TODO use a range
        if isinstance(lod, np.ndarray):
            lod = np.amax(lod)

        bins = np.histogram_bin_edges(data, bins=self.nbins)
        if len(bins) - 1 < 16:
            bins = np.histogram_bin_edges(data, bins=16)
        elif len(bins) - 1 > 128:
            bins = np.histogram_bin_edges(data, bins=128)
        hist, _ = np.histogram(
            data, bins=bins, range=(data.min(), np.percentile(data, 99.9))
        )
        self.chart.setData(hist, bins, xmin=0.0)

        self.chart.setVerticalLines([np.mean(data), np.median(data), lod])

        self.updateChartFit()

    def updateChartFit(self) -> None:
        mode = self.mode.currentText()
        if mode == "Mass":
            data = self.result["masses"] * 1e21  # ag
        elif mode == "Signal":
            data = self.result["detections"]  # counts
        elif mode == "Size":
            data = self.result["sizes"] * 1e9  # nm

        method = self.fitmethod.currentText()
        if method == "None":
            self.chart.fit.clear()
            self.chart.label_fit.setVisible(False)
            return

        # vmin and vmax are for non-ednsity histogram
        vmin = data.min()
        vmax = np.percentile(data, 99.9)

        bins = np.histogram_bin_edges(data, bins=self.nbins)
        if len(bins) - 1 < 16:
            bins = np.histogram_bin_edges(data, bins=16)
        elif len(bins) - 1 > 128:
            bins = np.histogram_bin_edges(data, bins=128)
        hist, _ = np.histogram(
            data, bins=bins, range=(vmin, vmax), density=True
        )

        if method == "Normal":
            fit, err, opts = fit_normal(bins[1:], hist)
        elif method == "Lognormal":
            fit, err, opts = fit_lognormal(bins[1:], hist)

        # Fit to the non-density histogram
        binwidth = bins[1] - bins[0]
        fit = fit * binwidth * data.size
        self.chart.setFit(bins[1:], fit)
        self.chart.fit.setName(method)
        self.chart.label_fit.setVisible(True)

    def updateTable(self) -> None:
        mode = self.mode.currentText()
        if mode == "Mass":
            data = self.result["masses"]
        elif mode == "Signal":
            data = self.result["detections"]
        elif mode == "Size":
            data = self.result["sizes"]
        self.table.model().beginResetModel()
        self.table.model().array = data[:, None]
        self.table.model().endResetModel()

    def updateResults(self) -> None:
        method = self.options.efficiency_method.currentText()

        self.result = {
            "background": self.sample.background,
            "detections": self.sample.detections,
            "events": self.sample.numberOfEvents(),
            "file": self.sample.label_file.text(),
            "limit_method": self.sample.limits[0],
            "lod": self.sample.limits[3],
        }

        if method in ["Manual", "Reference"]:
            if method == "Manual":
                efficiency = float(self.options.efficiency.text())
            elif method == "Reference":
                efficiency = float(self.reference.efficiency.text())

            dwelltime = self.options.dwelltime.baseValue()
            density = self.sample.density.baseValue()
            molarratio = float(self.sample.molarratio.text())
            time = self.sample.timeAsSeconds()
            uptake = self.options.uptake.baseValue()
            response = self.options.response.baseValue()

            self.result.update(
                results_from_nebulisation_efficiency(
                    self.result["detections"],
                    self.result["background"],
                    self.result["lod"],
                    density=density,
                    dwelltime=dwelltime,
                    efficiency=efficiency,
                    molarratio=molarratio,
                    uptake=uptake,
                    response=response,
                    time=time,
                )
            )
        elif method == "Mass Response (None)":
            density = self.sample.density.baseValue()
            molarratio = float(self.sample.molarratio.text())
            massresponse = self.reference.massresponse.baseValue()

            self.result.update(
                results_from_mass_response(
                    self.result["detections"],
                    self.result["background"],
                    self.result["lod"],
                    density=density,
                    molarratio=molarratio,
                    massresponse=massresponse,
                )
            )

        self.mean.setBaseValue(np.mean(self.result["sizes"]))
        unit = self.mean.setBestUnit()
        self.median.setBaseValue(np.median(self.result["sizes"]))
        self.median.setUnit(unit)
        self.lod.setBaseValue(np.amax(self.result.get("lod_size", None)))
        self.mean.setUnit(unit)

        self.count.setText(f"{self.sample.detections.size}")
        self.number.setBaseValue(self.result.get("number_concentration", None))
        self.number.setBestUnit()
        self.conc.setBaseValue(self.result.get("concentration", None))
        unit = self.conc.setBestUnit()
        self.background.setBaseValue(self.result.get("background_concentration", None))
        self.background.setUnit(unit)

        self.updateTable()
        self.updateChart()
