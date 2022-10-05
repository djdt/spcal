from PySide6 import QtCore, QtGui, QtWidgets

import numpy as np
from pathlib import Path

from spcal.calc import (
    results_from_mass_response,
    results_from_nebulisation_efficiency,
)
from spcal.fit import fit_normal, fit_lognormal
from spcal.io import export_nanoparticle_results
from spcal.util import cell_concentration

# from spcal.gui.charts import ParticleHistogram, ParticleChartView
from spcal.gui.inputs import SampleWidget, ReferenceWidget
from spcal.gui.options import OptionsWidget
from spcal.gui.tables import ResultsTable
from spcal.gui.units import UnitsWidget

from typing import Any, Dict, Tuple


class ResultsWidget(QtWidgets.QWidget):
    signal_units = {"counts": 1.0}
    size_units = {"nm": 1e-9, "μm": 1e-6, "m": 1.0}
    mass_units = {
        "ag": 1e-21,
        "fg": 1e-18,
        "pg": 1e-15,
        "ng": 1e-12,
        "μg": 1e-9,
        "g": 1e-3,
        "kg": 1.0,
    }
    conc_units = {
        "amol/L": 1e-18,
        "fmol/L": 1e-15,
        "pmol/L": 1e-12,
        "nmol/L": 1e-9,
        "μmol/L": 1e-6,
        "mmol/L": 1e-3,
        "mol/L": 1,
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
        self.result: Dict[str, Any] = {}

        # self.chart = ParticleHistogram()
        # self.chart.drawVerticalLines(
        #     [0, 0, 0, 0],
        #     names=["mean", "median", "lod", ""],
        #     pens=[
        #         QtGui.QPen(QtGui.QColor(255, 0, 0), 1.5, QtCore.Qt.DashLine),
        #         QtGui.QPen(QtGui.QColor(0, 0, 255), 1.5, QtCore.Qt.DashLine),
        #         QtGui.QPen(QtGui.QColor(0, 172, 0), 1.5, QtCore.Qt.DashLine),
        #         QtGui.QPen(QtGui.QColor(0, 172, 0), 1.5, QtCore.Qt.DashLine),
        #     ],
        #     visible_in_legend=[True, True, True, False],
        # )
        # self.chartview = ParticleChartView(self.chart)
        # self.chartview.setRubberBand(QtCharts.QChartView.HorizontalRubberBand)
        # self.chartview.setRenderHint(QtGui.QPainter.Antialiasing)

        self.table = ResultsTable()

        self.fitmethod = QtWidgets.QComboBox()
        self.fitmethod.addItems(["None", "Normal", "Lognormal"])
        self.fitmethod.setCurrentText("Lognormal")

        self.fitmethod.currentIndexChanged.connect(self.updateChart)

        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["Signal", "Mass (kg)", "Size (m)", "Conc. (mol/L)"])
        self.mode.setItemData(
            0,
            "Accumulated detection singal.",
            QtCore.Qt.ToolTipRole,
        )
        self.mode.setItemData(
            1,
            "Particle mass, requires calibration.",
            QtCore.Qt.ToolTipRole,
        )
        self.mode.setItemData(
            2,
            "Particle size, requires calibration.",
            QtCore.Qt.ToolTipRole,
        )
        self.mode.setItemData(
            3,
            "Intracellular concentration, requires cell diameter and analyte molarmass.",
            QtCore.Qt.ToolTipRole,
        )
        self.mode.setCurrentText("Size (m)")
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
        layout_outputs_left.addRow("No. Detections:", self.count)
        layout_outputs_left.addRow("No. Concentration:", self.number)
        layout_outputs_left.addRow("Concentration:", self.conc)
        layout_outputs_left.addRow("Ionic Background:", self.background)

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

        # layout_chart_options = QtWidgets.QHBoxLayout()
        # layout_chart_options.addWidget(self.button_export_image)
        # layout_chart_options.addStretch(1)
        # layout_chart_options.addWidget(QtWidgets.QLabel("Fit:"), 0)
        # layout_chart_options.addWidget(self.fitmethod)

        layout_outputs = QtWidgets.QVBoxLayout()
        layout_outputs.addLayout(layout_filename)
        layout_outputs.addWidget(self.outputs)
        # layout_outputs.addWidget(self.chartview)
        # layout_outputs.addLayout(layout_chart_options)

        layout = QtWidgets.QHBoxLayout()
        layout.addLayout(layout_table, 0)
        layout.addLayout(layout_outputs, 1)
        self.setLayout(layout)

    def dialogExportResults(self) -> None:
        file, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export", "", "CSV Documents (*.csv)"
        )
        if file != "":
            export_nanoparticle_results(Path(file), self.result)

    def dialogExportImage(self) -> None:
        file, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Image", "", "PNG Images (*.png)"
        )
        if file != "":
            self.chartview.saveToFile(file)

    def asBestUnit(
        self, data: np.ndarray, current_unit: str = ""
    ) -> Tuple[np.ndarray, float, str]:
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
        if method != "Manual Input" and not self.reference.isComplete():
            return False
        return True

    def updateChart(self) -> None:
        mode = self.mode.currentText()
        if mode == "Mass (kg)":
            data, mult, unit = self.asBestUnit(self.result["masses"], "k")
            lod = self.result["lod_mass"] / mult
            self.chart.xaxis.setTitleText(f"Mass ({unit}g)")
        elif mode == "Signal":
            data = self.result["detections"]  # counts
            lod = self.result["lod"]
            self.chart.xaxis.setTitleText("Signal (counts)")
        elif mode == "Size (m)":
            data, mult, unit = self.asBestUnit(self.result["sizes"])
            lod = self.result["lod_size"] / mult
            self.chart.xaxis.setTitleText(f"Size ({unit}m)")
        elif mode == "Conc. (mol/L)":
            data, mult, unit = self.asBestUnit(self.result["cell_concentrations"], "")
            lod = self.result["lod_cell_concentration"] / mult
            self.chart.xaxis.setTitleText(f"Conc. ({unit}mol/L)")
        else:
            raise ValueError(f"Unknown mode '{mode}'.")

        # Crush the LOD for chart
        if isinstance(lod, np.ndarray):
            lod_min, lod_max = lod[0], lod[1]
        else:
            lod_min, lod_max = lod, lod

        # TODO option for choosing percentile
        hist_data = data[data < np.percentile(data, 98)]

        bins = np.histogram_bin_edges(hist_data, bins=self.nbins)
        if len(bins) - 1 < 16:
            bins = np.histogram_bin_edges(hist_data, bins=16)
        elif len(bins) - 1 > 128:
            bins = np.histogram_bin_edges(hist_data, bins=128)

        hist, _ = np.histogram(hist_data, bins=bins)
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
            fit = fit_normal(bins[1:], hist)[0]
        elif method == "Lognormal":
            fit = fit_lognormal(bins[1:], hist)[0]
        else:
            raise ValueError(f"Unknown fit type '{method}'.")

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
                np.mean(self.result["lod"]),
                np.std(self.result["detections"]),
            )
        elif mode == "Mass (kg)":
            units = self.mass_units
            mean, median, lod, std = (
                np.mean(self.result["masses"]),
                np.median(self.result["masses"]),
                np.mean(self.result["lod_mass"]),
                np.std(self.result["masses"]),
            )
        elif mode == "Size (m)":
            units = self.size_units
            mean, median, lod, std = (
                np.mean(self.result["sizes"]),
                np.median(self.result["sizes"]),
                np.mean(self.result["lod_size"]),
                np.std(self.result["sizes"]),
            )
        elif mode == "Conc. (mol/L)":
            units = self.conc_units
            mean, median, lod, std = (
                np.mean(self.result["cell_concentrations"]),
                np.median(self.result["cell_concentrations"]),
                np.mean(self.result["lod_cell_concentration"]),
                np.std(self.result["cell_concentrations"]),
            )
        else:
            raise ValueError(f"Unknown mode {mode}.")

        for te in [self.mean, self.median, self.lod]:
            te.setUnits(units)

        self.mean.setBaseValue(mean)
        self.mean.setBaseError(std)
        self.median.setBaseValue(median)
        self.lod.setBaseValue(lod)

        unit = self.mean.setBestUnit()
        self.median.setUnit(unit)
        self.lod.setUnit(unit)

        perc_error = self.sample.detections_std / self.sample.detections.size
        self.count.setText(
            f"{self.sample.detections.size} ± {self.sample.detections_std:.1f}"
        )
        self.number.setBaseValue(self.result.get("number_concentration", None))
        self.number.setBaseError(
            self.result.get("number_concentration", 0.0) * perc_error
        )
        self.number.setBestUnit()
        self.conc.setBaseValue(self.result.get("concentration", None))
        self.conc.setBaseError(self.result.get("concentration", 0.0) * perc_error)
        unit = self.conc.setBestUnit()
        self.background.setBaseValue(self.result.get("background_concentration", None))
        ionic_error = self.result.get("background_concentration", None)
        if ionic_error is not None:
            ionic_error *= self.result["background_std"] / self.result["background"]
        self.background.setBaseError(ionic_error)
        self.background.setUnit(unit)

    def updateTable(self) -> None:
        mode = self.mode.currentText()
        if mode == "Mass (kg)":
            data = self.result["masses"]
        elif mode == "Signal":
            data = self.result["detections"]
        elif mode == "Size (m)":
            data = self.result["sizes"]
        elif mode == "Conc. (mol/L)":
            data = self.result["cell_concentrations"]
        else:
            raise ValueError(f"Unknown mode '{mode}'.")
        self.table.model().beginResetModel()
        self.table.model().array = data[:, None]  # type: ignore
        self.table.model().endResetModel()

    def updateResults(self) -> None:

        self.result = {
            "background": self.sample.background,
            "background_std": self.sample.background_std,
            "detections": self.sample.detections,
            "detections_std": self.sample.detections_std,
            "events": self.sample.numberOfEvents(),
            "file": self.sample.label_file.text(),
            "limit_method": f"{self.sample.limits[0]},{','.join(f'{k}={v}' for k,v in self.sample.limits[1].items())}",
            "limit_window": int(self.options.window_size.text()),
            "lod": self.sample.limit_ld,
        }

        method = self.options.efficiency_method.currentText()
        if not self.readyForResults():
            self.mode.setCurrentText("Signal")
            self.mode.setEnabled(False)
            self.result["inputs"] = {}
        else:
            self.mode.setEnabled(True)

            if method in ["Manual Input", "Reference Particle"]:
                if method == "Manual Input":
                    efficiency = float(self.options.efficiency.text())
                elif method == "Reference Particle":
                    efficiency = float(self.reference.efficiency.text())
                else:
                    raise ValueError(f"Unknown method {method}.")

                dwelltime = self.options.dwelltime.baseValue()
                density = self.sample.density.baseValue()
                massfraction = float(self.sample.massfraction.text())
                time = self.sample.timeAsSeconds()
                uptake = self.options.uptake.baseValue()
                response = self.options.response.baseValue()

                self.result.update(
                    results_from_nebulisation_efficiency(
                        self.result["detections"],
                        self.result["background"],
                        self.result["lod"],
                        density=density,  # type: ignore
                        dwelltime=dwelltime,  # type: ignore
                        efficiency=efficiency,
                        massfraction=massfraction,
                        uptake=uptake,  # type: ignore
                        response=response,  # type: ignore
                        time=time,  # type: ignore
                    )
                )
                self.result["inputs"] = {
                    "density": density,
                    "dwelltime": dwelltime,
                    "transport_efficiency": efficiency,
                    "mass_fraction": massfraction,
                    "uptake": uptake,
                    "response": response,
                    "time": time,
                }
            elif method == "Mass Response":
                density = self.sample.density.baseValue()
                massfraction = float(self.sample.massfraction.text())
                massresponse = self.reference.massresponse.baseValue()

                self.result.update(
                    results_from_mass_response(
                        self.result["detections"],
                        self.result["background"],
                        self.result["lod"],
                        density=density,  # type: ignore
                        massfraction=massfraction,
                        massresponse=massresponse,  # type: ignore
                    )
                )
                self.result["inputs"] = {
                    "density": density,
                    "mass_fraction": massfraction,
                    "mass_response": massresponse,
                }

            # Cell inputs
            concindex = self.mode.findText("Conc. (mol/L)")
            celldiameter = self.options.celldiameter.baseValue()
            molarmass = self.sample.molarmass.baseValue()
            if celldiameter is not None:  # Scale sizes to hypothesised
                scale = celldiameter / np.mean(self.result["sizes"])
                self.result["sizes"] *= scale
                self.result["lod_size"] *= scale
                self.result["inputs"].update({"cell_diameter": celldiameter})

            if (
                celldiameter is not None and molarmass is not None
            ):  # Calculate the intracellular concetrations
                self.mode.model().item(concindex).setEnabled(True)

                self.result["cell_concentrations"] = cell_concentration(
                    self.result["masses"],
                    diameter=celldiameter,
                    molarmass=molarmass,
                )
                self.result["lod_cell_concentration"] = cell_concentration(
                    self.result["lod_mass"],
                    diameter=celldiameter,
                    molarmass=molarmass,
                )
                self.result["inputs"].update({"molarmass": molarmass})
            else:
                self.mode.model().item(concindex).setEnabled(False)
                if self.mode.currentIndex() == concindex:
                    self.mode.setCurrentText("Signal")

        self.updateTexts()
        self.updateTable()
        self.updateChart()
