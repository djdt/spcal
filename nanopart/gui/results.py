from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtCharts import QtCharts

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

from typing import Tuple


class ResultsWidget(QtWidgets.QWidget):
    signal_units = {"cnt": 1.0}
    size_units = {"nm": 1e-9, "μm": 1e-6, "m": 1.0}
    mass_units = {
        "fg": 1e-18,
        "pg": 1e-15,
        "ng": 1e-12,
        "μg": 1e-9,
        "g": 1e-3,
        "kg": 1.0,
    }

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
            [0, 0, 0, 0],
            names=["mean", "median", "lod", ""],
            pens=[
                QtGui.QPen(QtGui.QColor(255, 0, 0), 1.5, QtCore.Qt.DashLine),
                QtGui.QPen(QtGui.QColor(0, 0, 255), 1.5, QtCore.Qt.DashLine),
                QtGui.QPen(QtGui.QColor(0, 172, 0), 1.5, QtCore.Qt.DashLine),
                QtGui.QPen(QtGui.QColor(0, 172, 0), 1.5, QtCore.Qt.DashLine),
            ],
            visible_in_legend=[True, True, True, False],
        )
        self.chartview = ParticleChartView(self.chart)
        self.chartview.setRubberBand(QtCharts.QChartView.HorizontalRubberBand)
        self.chartview.setRenderHint(QtGui.QPainter.Antialiasing)

        self.table = ResultsTable()

        self.fitmethod = QtWidgets.QComboBox()
        self.fitmethod.addItems(["None", "Normal", "Lognormal"])
        self.fitmethod.setCurrentText("Lognormal")

        self.fitmethod.currentIndexChanged.connect(self.updateChart)

        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["Signal", "Mass", "Size"])
        self.mode.setCurrentText("Size")
        self.mode.currentIndexChanged.connect(self.updateTexts)
        self.mode.currentIndexChanged.connect(self.updateTable)
        self.mode.currentIndexChanged.connect(self.updateChart)

        self.outputs = QtWidgets.QGroupBox("Outputs")
        self.outputs.setLayout(QtWidgets.QHBoxLayout())

        self.count = QtWidgets.QLineEdit()
        self.count.setReadOnly(True)
        self.number = UnitsWidget(
            {"#/L": 1.0, "#/ml": 1e3},
            default_unit="#/L",
            formatter=".0f",
        )
        self.number.setReadOnly(True)
        self.conc = UnitsWidget(
            concentration_units,
            default_unit="ng/L",
        )
        self.conc.setReadOnly(True)
        self.background = UnitsWidget(
            concentration_units,
            default_unit="ng/L",
        )
        self.background.setReadOnly(True)

        self.lod = UnitsWidget(
            self.size_units,
            default_unit="nm",
        )
        self.lod.setReadOnly(True)
        self.mean = UnitsWidget(
            self.size_units,
            default_unit="nm",
        )
        self.mean.setReadOnly(True)
        self.median = UnitsWidget(
            self.size_units,
            default_unit="nm",
        )
        self.median.setReadOnly(True)

        layout_outputs_left = QtWidgets.QFormLayout()
        layout_outputs_left.addRow("Particles:", self.count)
        layout_outputs_left.addRow("Number Conc.:", self.number)
        layout_outputs_left.addRow("Concentration:", self.conc)
        layout_outputs_left.addRow("Ionic:", self.background)

        layout_outputs_right = QtWidgets.QFormLayout()
        layout_outputs_right.addRow("Mean:", self.mean)
        layout_outputs_right.addRow("Median:", self.median)
        layout_outputs_right.addRow("LOD:", self.lod)

        self.outputs.layout().addLayout(layout_outputs_left)
        self.outputs.layout().addLayout(layout_outputs_right)

        self.label_file = QtWidgets.QLabel()

        self.button_export = QtWidgets.QPushButton("Export Results")
        self.button_export.pressed.connect(self.dialogExportResults)

        self.button_export_image = QtWidgets.QPushButton("Save Image")
        self.button_export_image.pressed.connect(self.dialogExportImage)

        layout_table_options = QtWidgets.QHBoxLayout()
        layout_table_options.addStretch(1)
        layout_table_options.addWidget(QtWidgets.QLabel("Mode:"), 0)
        layout_table_options.addWidget(self.mode)

        layout_table = QtWidgets.QVBoxLayout()
        layout_table.addLayout(layout_table_options)
        layout_table.addWidget(self.table, 1)

        layout_filename = QtWidgets.QHBoxLayout()
        layout_filename.addWidget(self.button_export, 0, QtCore.Qt.AlignLeft)
        layout_filename.addWidget(self.label_file, 1)

        layout_chart_options = QtWidgets.QHBoxLayout()
        layout_chart_options.addWidget(self.button_export_image)
        layout_chart_options.addStretch(1)
        layout_chart_options.addWidget(QtWidgets.QLabel("Fit:"), 0)
        layout_chart_options.addWidget(self.fitmethod)

        layout_outputs = QtWidgets.QVBoxLayout()
        layout_outputs.addLayout(layout_filename)
        layout_outputs.addWidget(self.outputs)
        layout_outputs.addWidget(self.chartview)
        layout_outputs.addLayout(layout_chart_options)

        layout = QtWidgets.QHBoxLayout()
        layout.addLayout(layout_table, 0)
        layout.addLayout(layout_outputs, 1)
        self.setLayout(layout)

    def dialogExportResults(self) -> None:
        file, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export", "", "CSV Documents (*.csv)"
        )
        if file != "":
            export_nanoparticle_results(Path(file), self.result)

    def dialogExportImage(self) -> None:
        file, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Image", "", "PNG Images (*.png)"
        )
        if file != "":
            self.chartview.saveToFile(file)

    def asBestUnit(self, data: np.ndarray, current_unit: str = "") -> Tuple[float, str]:
        units = {
            "z": 1e-21,
            "a": 1e-18,
            "f": 1e-15,
            "p": 1e-12,
            "n": 1e-9,
            "μ": 1e-6,
            "m": 1e-3,
            "": 1.0,
            "k": 1e3,
            "M": 1e6,
        }

        data = data * units[current_unit]

        mean = np.mean(data)
        pwr = 10 ** int(np.log10(mean) - (1 if mean < 1.0 else 0))

        vals = list(units.values())
        names = list(units.keys())
        idx = np.searchsorted(list(units.values()), pwr) - 1

        return data / vals[idx], vals[idx] / units[current_unit], names[idx]

    def readyForResults(self) -> bool:
        if not self.options.isComplete():
            return False
        if not self.sample.isComplete():
            return False

        method = self.options.efficiency_method.currentText()
        if method != "Manual" and not self.reference.isComplete():
            return False
        return True

    def updateChart(self) -> None:
        mode = self.mode.currentText()
        if mode == "Mass":
            data, mult, unit = self.asBestUnit(self.result["masses"], "k")
            lod = self.result["lod_mass"] / mult
            self.chart.xaxis.setTitleText(f"Mass ({unit}g)")
        elif mode == "Signal":
            data = self.result["detections"]  # counts
            lod = self.result["lod"]
            self.chart.xaxis.setTitleText("Signal (counts)")
        elif mode == "Size":
            data, mult, unit = self.asBestUnit(self.result["sizes"])
            lod = self.result["lod_size"] / mult
            self.chart.xaxis.setTitleText(f"Size ({unit}m)")

        # Crush the LOD for chart
        if isinstance(lod, np.ndarray):
            lod_min, lod_max = lod[0], lod[1]
        else:
            lod_min, lod_max = lod, lod

        bins = np.histogram_bin_edges(data, bins=self.nbins)
        if len(bins) - 1 < 16:
            bins = np.histogram_bin_edges(data, bins=16)
        elif len(bins) - 1 > 128:
            bins = np.histogram_bin_edges(data, bins=128)

        hist, _ = np.histogram(
            data, bins=bins, range=(data.min(), np.percentile(data, 99.9))
        )
        self.chart.setData(hist, bins, xmin=0.0)

        self.chart.setVerticalLines([np.mean(data), np.median(data), lod_min, lod_max])

        self.updateChartFit(hist, bins, data.size)

    def updateChartFit(self, hist: np.ndarray, bins: np.ndarray, size: int) -> None:
        method = self.fitmethod.currentText()
        if method == "None":
            self.chart.fit.clear()
            self.chart.label_fit.setVisible(False)
            return

        # Convert to density
        binwidth = bins[1] - bins[0]
        hist = hist / binwidth / size

        if method == "Normal":
            fit, err, opts = fit_normal(bins[1:], hist)
        elif method == "Lognormal":
            fit, err, opts = fit_lognormal(bins[1:], hist)

        # Convert from density
        fit = fit * binwidth * size

        self.chart.setFit(bins[1:], fit)
        self.chart.fit.setName(method)
        self.chart.label_fit.setVisible(True)

    def updateTexts(self) -> None:
        self.label_file.setText(
            Path(self.result["file"]).name + " (" + self.result["limit_method"] + ")"
        )

        mode = self.mode.currentText()
        if mode == "Signal":
            units = self.signal_units
            mean, median, lod, std = (
                np.mean(self.result["detections"]),
                np.median(self.result["detections"]),
                np.mean(self.result.get("lod", None)),
                np.std(self.result["detections"]),
            )
        elif mode == "Mass":
            units = self.mass_units
            mean, median, lod, std = (
                np.mean(self.result["masses"]),
                np.median(self.result["masses"]),
                np.mean(self.result.get("lod_mass", None)),
                np.std(self.result["masses"]),
            )
        else:
            units = self.size_units
            mean, median, lod, std = (
                np.mean(self.result["sizes"]),
                np.median(self.result["sizes"]),
                np.mean(self.result.get("lod_size", None)),
                np.std(self.result["sizes"]),
            )

        for te in [self.mean, self.median, self.lod]:
            te.setUnits(units)

        self.mean.setBaseValue(mean)
        self.mean.setBaseError(std)
        self.median.setBaseValue(median)
        self.lod.setBaseValue(lod)

        unit = self.mean.setBestUnit()
        self.median.setUnit(unit)
        self.lod.setUnit(unit)

        self.count.setText(
            f"{self.sample.detections.size} ± {self.sample.detections_std:.1f}"
        )
        self.number.setBaseValue(self.result.get("number_concentration", None))
        self.number.setBestUnit()
        self.conc.setBaseValue(self.result.get("concentration", None))
        unit = self.conc.setBestUnit()
        self.background.setBaseValue(self.result.get("background_concentration", None))
        ionic_error = self.result.get("background_concentration", None)
        if ionic_error is not None:
            ionic_error *= self.result["background_std"] / self.result["background"]
        self.background.setBaseError(ionic_error)
        self.background.setUnit(unit)

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

        self.result = {
            "background": self.sample.background,
            "background_std": self.sample.background_std,
            "detections": self.sample.detections,
            "detections_std": self.sample.detections_std,
            "events": self.sample.numberOfEvents(),
            "file": self.sample.label_file.text(),
            "limit_method": self.sample.limits[0],
            "limit_window": int(self.options.window_size.text()),
            "lod": self.sample.limits[3],
        }

        method = self.options.efficiency_method.currentText()
        if not self.readyForResults():
            self.mode.setCurrentText("Signal")
            self.mode.setEnabled(False)
        else:
            self.mode.setEnabled(True)

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

            if self.options.diameter.hasAcceptableInput():
                self.result["sizes"] /= np.mean(self.result["sizes"])
                self.result["sizes"] *= self.options.diameter.baseValue()

        self.updateTexts()
        self.updateTable()
        self.updateChart()
