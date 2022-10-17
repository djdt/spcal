from PySide6 import QtCore, QtGui, QtWidgets

import numpy as np
from pathlib import Path

from spcal.calc import (
    results_from_mass_response,
    results_from_nebulisation_efficiency,
)
from spcal.fit import fit_normal, fit_lognormal
from spcal.io import export_nanoparticle_results
from spcal.util import (
    cell_concentration,
    detection_element_fractions,
    fraction_components,
)

from spcal.gui.graphs import ResultsFractionView, ResultsHistView, graph_colors
from spcal.gui.iowidgets import ResultIOStack
from spcal.gui.inputs import SampleWidget, ReferenceWidget
from spcal.gui.options import OptionsWidget
from spcal.gui.util import create_action

from typing import Dict, Optional, Tuple


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
    molar_concentration_units = {
        "amol/L": 1e-18,
        "fmol/L": 1e-15,
        "pmol/L": 1e-12,
        "nmol/L": 1e-9,
        "μmol/L": 1e-6,
        "mmol/L": 1e-3,
        "mol/L": 1.0,
    }
    concentration_units = {
        "fg/L": 1e-18,
        "pg/L": 1e-15,
        "ng/L": 1e-12,
        "μg/L": 1e-9,
        "mg/L": 1e-6,
        "g/L": 1e-3,
        "kg/L": 1.0,
    }

    def __init__(
        self,
        options: OptionsWidget,
        sample: SampleWidget,
        reference: ReferenceWidget,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)

        self.options = options
        self.sample = sample
        self.reference = reference

        self.nbins = "auto"
        self.result: Dict[str, dict] = {}

        self.graph_toolbar = QtWidgets.QToolBar()
        self.graph_toolbar.setOrientation(QtCore.Qt.Vertical)
        self.graph_toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)

        self.graph_hist = ResultsHistView()
        self.graph_frac = ResultsFractionView()

        self.graph_stack = QtWidgets.QStackedWidget()
        self.graph_stack.addWidget(self.graph_hist)
        self.graph_stack.addWidget(self.graph_frac)

        self.io = ResultIOStack()

        self.fitmethod = QtWidgets.QComboBox()
        self.fitmethod.addItems(["None", "Normal", "Lognormal"])
        self.fitmethod.setCurrentText("Lognormal")

        self.fitmethod.currentIndexChanged.connect(self.drawGraph)

        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["Signal", "Mass (kg)", "Size (m)", "Conc. (mol/L)"])
        self.mode.setItemData(0, "Accumulated detection signal.", QtCore.Qt.ToolTipRole)
        self.mode.setItemData(
            1, "Particle mass, requires calibration.", QtCore.Qt.ToolTipRole
        )
        self.mode.setItemData(
            2, "Particle size, requires calibration.", QtCore.Qt.ToolTipRole
        )
        self.mode.setItemData(
            3,
            "Intracellular concentration, requires cell diameter and analyte molarmass.",
            QtCore.Qt.ToolTipRole,
        )
        self.mode.setCurrentText("Signal")
        self.mode.currentIndexChanged.connect(lambda: self.updateOutputs(None))
        self.mode.currentIndexChanged.connect(self.drawGraph)

        self.label_file = QtWidgets.QLabel()

        self.button_export = QtWidgets.QPushButton("Export Results")
        self.button_export.pressed.connect(self.dialogExportResults)

        self.button_export_image = QtWidgets.QPushButton("Save Image")
        self.button_export_image.pressed.connect(self.dialogExportImage)

        # Actions

        self.action_graph_histogram = create_action(
            "view-object-histogram-linear",
            "Histogram",
            "Switch to the histogram view.",
            lambda: self.graph_stack.setCurrentWidget(self.graph_hist),
            checkable=True,
        )
        self.action_graph_histogram.setChecked(True)
        self.action_graph_fractions = create_action(
            "office-chart-bar-stacked",
            "Histogram",
            "Switch to the histogram view.",
            lambda: self.graph_stack.setCurrentWidget(self.graph_frac),
            checkable=True,
        )
        self.action_graph_zoomout = create_action(
            "zoom-original",
            "Zoom Out",
            "Reset the plot view.",
            self.graph_hist.zoomReset,
        )
        action_group_graph_view = QtGui.QActionGroup(self)
        action_group_graph_view.addAction(self.action_graph_histogram)
        action_group_graph_view.addAction(self.action_graph_fractions)
        self.graph_toolbar.addActions(action_group_graph_view.actions())
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding
        )
        self.graph_toolbar.addWidget(spacer)
        self.graph_toolbar.addAction(self.action_graph_zoomout)

        # Layouts

        self.io.layout_top.insertWidget(
            0, QtWidgets.QLabel("Mode:"), 0, QtCore.Qt.AlignLeft
        )
        self.io.layout_top.insertWidget(1, self.mode, 0, QtCore.Qt.AlignLeft)
        self.io.layout_top.insertStretch(2, 1)

        layout_filename = QtWidgets.QHBoxLayout()
        layout_filename.addWidget(self.button_export, 0, QtCore.Qt.AlignLeft)
        layout_filename.addWidget(self.label_file, 1)

        # layout_chart_options = QtWidgets.QHBoxLayout()
        # layout_chart_options.addWidget(self.button_export_image)
        # layout_chart_options.addStretch(1)
        # layout_chart_options.addWidget(QtWidgets.QLabel("Fit:"), 0)
        # layout_chart_options.addWidget(self.fitmethod)

        layout_graph = QtWidgets.QHBoxLayout()
        layout_graph.addWidget(self.graph_toolbar, 0)
        layout_graph.addWidget(self.graph_stack, 1)

        layout_main = QtWidgets.QVBoxLayout()
        # layout_outputs.addLayout(layout_filename)
        layout_main.addWidget(self.io, 0)
        layout_main.addLayout(layout_graph, 1)

        layout = QtWidgets.QHBoxLayout()
        layout.addLayout(layout_main, 1)
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
        # if file != "":
        #     self.chartview.saveToFile(file)

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

    def drawGraph(self) -> None:
        self.drawGraphHist()
        self.drawGraphFrac()

    def drawGraphHist(self) -> None:
        self.graph_hist.clear()
        mode = self.mode.currentText()

        if mode == "Signal":
            label, unit = "Intensity (counts)", None
        elif mode == "Mass (kg)":
            label, unit = "Mass", "g"
        elif mode == "Size (m)":
            label, unit = "Size", "m"
        elif mode == "Conc. (mol/L)":
            label, unit = "Concentration", "mol/L"
        else:
            raise ValueError("drawGraph: unknown mode.")

        self.graph_hist.xaxis.setLabel(label, unit)

        graph_data = {}
        for name in self.result:
            if mode == "Signal":
                graph_data[name] = self.result[name]["detections"]
            elif mode == "Mass (kg)" and "masses" in self.result[name]:
                graph_data[name] = self.result[name]["masses"] * 1000  # convert to gram
            elif mode == "Size (m)" and "sizes" in self.result[name]:
                graph_data[name] = self.result[name]["sizes"]
            elif mode == "Conc. (mol/L)" and "cell_concentrations" in self.result[name]:
                graph_data[name] = self.result[name]["cell_concentrations"]
            else:
                continue

        # median 'sturges' bin width
        bin_width = np.median(
            [
                np.ptp(graph_data[name]) / (np.log2(graph_data[name].size) + 1)
                for name in graph_data
            ]
        )

        for name, color in zip(graph_data, graph_colors):
            bins = np.arange(
                graph_data[name].min(), graph_data[name].max() + bin_width, bin_width
            )
            bins -= bins[0] % bin_width  # align bins
            color = QtGui.QColor(color)
            color.setAlpha(128)
            self.graph_hist.drawData(
                name, graph_data[name], bins=bins, brush=QtGui.QBrush(color)
            )
        self.graph_hist.zoomReset()

    def drawGraphFrac(self) -> None:
        # Fraction view
        self.graph_frac.clear()
        mode = self.mode.currentText()

        if mode == "Signal":
            label = "Intensity"
        elif mode == "Mass (kg)":
            label = "Mass"
        elif mode == "Size (m)":
            label = "Size"
        elif mode == "Conc. (mol/L)":
            label = "Concentration"
        else:
            raise ValueError("drawGraph: unknown mode.")

        self.graph_frac.plot.setTitle(f"{label} Composition")

        graph_data = {}
        for name in self.result:
            if mode == "Signal":
                graph_data[name] = self.result[name]["detections"]
            elif mode == "Mass (kg)" and "masses" in self.result[name]:
                graph_data[name] = self.result[name]["masses"] * 1000  # convert to gram
            elif mode == "Size (m)" and "sizes" in self.result[name]:
                graph_data[name] = self.result[name]["sizes"]
            elif mode == "Conc. (mol/L)" and "cell_concentrations" in self.result[name]:
                graph_data[name] = self.result[name]["cell_concentrations"]
            else:
                continue

        if not all(name in graph_data for name in self.sample.detections):
            return

        fractions = detection_element_fractions(
            graph_data, self.sample.labels, self.sample.regions
        )
        compositions, counts = fraction_components(fractions, combine_similar=True)

        mask = counts > fractions.size * 0.05
        compositions = compositions[mask]
        counts = counts[mask]

        if counts.size == 0:
            return

        brushes = []
        for gc in graph_colors:
            color = QtGui.QColor(gc)
            color.setAlpha(128)
            brushes.append(QtGui.QBrush(color))

        self.graph_frac.drawData(compositions, counts, brushes=brushes)

    # def updateChartFit(self, hist: np.ndarray, bins: np.ndarray, size: int) -> None:
    #     method = self.fitmethod.currentText()
    #     if method == "None":
    #         self.chart.fit.clear()
    #         self.chart.label_fit.setVisible(False)
    #         return

    #     # Convert to density
    #     binwidth = bins[1] - bins[0]
    #     hist = hist / binwidth / size

    #     if method == "Normal":
    #         fit = fit_normal(bins[1:], hist)[0]
    #     elif method == "Lognormal":
    #         fit = fit_lognormal(bins[1:], hist)[0]
    #     else:
    #         raise ValueError(f"Unknown fit type '{method}'.")

    #     # Convert from density
    #     fit = fit * binwidth * size

    #     self.chart.setFit(bins[1:], fit)
    #     self.chart.fit.setName(method)
    #     self.chart.label_fit.setVisible(True)

    def updateOutputs(self, _name: Optional[str] = None) -> None:
        mode = self.mode.currentText()
        if _name is None or _name == "Overlay":
            names = list(self.sample.detections.keys())
        else:
            names = [_name]

        for name in names:
            if mode == "Signal":
                units = self.signal_units
                values = self.result[name]["detections"]
                lod = self.result[name]["lod"]
            elif mode == "Mass (kg)" and "masses" in self.result[name]:
                units = self.mass_units
                values = self.result[name]["masses"]
                lod = self.result[name]["lod_mass"]
            elif mode == "Size (m)" and "sizes" in self.result[name]:
                units = self.size_units
                values = self.result[name]["sizes"]
                lod = self.result[name]["lod_size"]
            elif mode == "Conc. (mol/L)" and "cell_concentrations" in self.result[name]:
                units = self.molar_concentration_units
                values = self.result[name]["cell_concentrations"]
                lod = self.result[name]["lod_cell_concentration"]
            else:
                self.io[name].clearOutputs()
                continue

            self.io[name].updateOutputs(
                values,
                units,
                lod,
                count=self.result[name]["detections"].size,
                count_error=self.result[name]["detections_std"],
                conc=self.result[name].get("concentration", None),
                number_conc=self.result[name].get("number_concentration", None),
                background_conc=self.result[name].get("background_concentration", None),
                background_error=self.result[name]["background_std"]
                / self.result[name]["background"],
            )

    def updateResults(self, _name: Optional[str] = None) -> None:
        method = self.options.efficiency_method.currentText()

        if _name is None or _name == "Overlay":
            names = list(self.sample.detections.keys())
        else:
            names = [_name]

        for name in names:
            trim = self.sample.trimRegion(name)
            responses = self.sample.responses[name][trim[0] : trim[1]]

            result = {
                "background": np.mean(responses[self.sample.labels[name] == 0]),
                "background_std": np.std(responses[self.sample.labels[name] == 0]),
                "detections": self.sample.detections[name],
                "detections_std": np.sqrt(self.sample.detections[name].size),
                "events": responses.size,
                "file": self.sample.label_file.text(),
                "limit_method": f"{self.sample.limits[name][0]},{','.join(f'{k}={v}' for k,v in self.sample.limits[name][1].items())}",
                "limit_window": int(self.options.window_size.text()),
                "lod": self.sample.limits[name][2]["ld"],
            }

            if method in ["Manual Input", "Reference Particle"]:
                try:
                    if method == "Manual Input":
                        efficiency = float(self.options.efficiency.text())
                    elif method == "Reference Particle" and name in self.reference.io:
                        efficiency = float(self.reference.io[name].efficiency.text())
                    else:
                        continue
                except ValueError:
                    continue

                dwelltime = self.options.dwelltime.baseValue()
                density = self.sample.io[name].density.baseValue()
                response = self.sample.io[name].response.baseValue()
                time = result["events"] * dwelltime
                uptake = self.options.uptake.baseValue()

                try:
                    massfraction = float(self.sample.io[name].massfraction.text())
                except ValueError:
                    continue

                if (
                    dwelltime is not None
                    and density is not None
                    and response is not None
                    and uptake is not None
                ):
                    result.update(
                        results_from_nebulisation_efficiency(
                            result["detections"],
                            result["background"],
                            result["lod"],
                            density=density,
                            dwelltime=dwelltime,
                            efficiency=efficiency,
                            massfraction=massfraction,
                            uptake=uptake,
                            response=response,
                            time=time,
                        )
                    )
                    result["inputs"] = {
                        "density": density,
                        "dwelltime": dwelltime,
                        "transport_efficiency": efficiency,
                        "mass_fraction": massfraction,
                        "uptake": uptake,
                        "response": response,
                        "time": time,
                    }
            elif method == "Mass Response":
                if name not in self.reference.io:
                    continue
                try:
                    massfraction = float(self.sample.io[name].massfraction.text())
                except ValueError:
                    continue

                density = self.sample.io[name].density.baseValue()
                massresponse = self.reference.io[name].massresponse.baseValue()

                if density is not None and massresponse is not None:
                    self.result.update(
                        results_from_mass_response(
                            result["detections"],
                            result["background"],
                            result["lod"],
                            density=density,
                            massfraction=massfraction,
                            massresponse=massresponse,
                        )
                    )
                    result["inputs"] = {
                        "density": density,
                        "mass_fraction": massfraction,
                        "mass_response": massresponse,
                    }

            # Cell inputs
            celldiameter = self.options.celldiameter.baseValue()
            molarmass = self.sample.io[name].molarmass.baseValue()

            if celldiameter is not None:  # Scale sizes to hypothesised
                scale = celldiameter / np.mean(result["sizes"])
                result["sizes"] *= scale
                result["lod_size"] *= scale
                result["inputs"].update({"cell_diameter": celldiameter})

            if (
                celldiameter is not None and molarmass is not None
            ):  # Calculate the intracellular concetrations
                result["cell_concentrations"] = cell_concentration(
                    result["masses"],
                    diameter=celldiameter,
                    molarmass=molarmass,
                )
                result["lod_cell_concentration"] = cell_concentration(
                    result["lod_mass"],
                    diameter=celldiameter,
                    molarmass=molarmass,
                )
                result["inputs"].update({"molarmass": molarmass})

            self.result[name] = result
            self.updateOutputs(name)
        # end for name in names
        self.drawGraph()
