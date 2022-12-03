import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.calc import results_from_mass_response, results_from_nebulisation_efficiency
from spcal.cluster import agglomerative_cluster, prepare_data_for_clustering
from spcal.fit import fit_lognormal, fit_normal, lognormal_pdf, normal_pdf
from spcal.gui.dialogs import FilterDialog, HistogramOptionsDialog
from spcal.gui.graphs import (
    ResultsFractionView,
    ResultsHistogramView,
    ResultsScatterView,
    color_schemes,
)
from spcal.gui.inputs import ReferenceWidget, SampleWidget
from spcal.gui.iowidgets import ResultIOStack
from spcal.gui.options import OptionsWidget
from spcal.gui.units import (
    mass_concentration_units,
    mass_units,
    molar_concentration_units,
    signal_units,
    size_units,
)
from spcal.gui.util import create_action
from spcal.io import export_nanoparticle_results
from spcal.particle import cell_concentration

logger = logging.getLogger(__name__)


# Todo: options dialog for each plot type, fit method, bins width, cluster parameters, etc.


class ResultsWidget(QtWidgets.QWidget):
    def __init__(
        self,
        options: OptionsWidget,
        sample: SampleWidget,
        reference: ReferenceWidget,
        color_scheme: str = "IBM Carbon",
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)

        self.draw_mode = "Overlay"
        self.color_scheme = color_scheme

        # Graph default options
        self.graph_hist_fit: str | None = "log normal"
        self.graph_hist_bin_widths: Dict[str, float | None] = {}

        self.options = options
        self.sample = sample
        self.reference = reference

        self.nbins = "auto"
        self.filters: List[Tuple[str, str, str, str, float]] = []
        self.results: Dict[str, dict] = {}

        self.graph_toolbar = QtWidgets.QToolBar()
        self.graph_toolbar.setOrientation(QtCore.Qt.Vertical)
        self.graph_toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)

        self.graph_hist = ResultsHistogramView()
        self.graph_frac = ResultsFractionView()
        self.graph_scatter = ResultsScatterView()

        self.combo_scatter_x = QtWidgets.QComboBox()
        self.combo_scatter_x.currentIndexChanged.connect(self.drawGraphScatter)
        self.combo_scatter_y = QtWidgets.QComboBox()
        self.combo_scatter_y.currentIndexChanged.connect(self.drawGraphScatter)

        self.check_scatter_logx = QtWidgets.QCheckBox("log")
        self.check_scatter_logx.clicked.connect(self.drawGraphScatter)
        self.check_scatter_logy = QtWidgets.QCheckBox("log")
        self.check_scatter_logy.clicked.connect(self.drawGraphScatter)

        # Create simple widget with graph and two combos for x / y element selection
        scatter_layout = QtWidgets.QVBoxLayout()
        scatter_combo_layout = QtWidgets.QHBoxLayout()
        scatter_combo_layout.addWidget(QtWidgets.QLabel("y:"), 0)
        scatter_combo_layout.addWidget(self.combo_scatter_y, 1)
        scatter_combo_layout.addWidget(self.check_scatter_logx, 0)
        scatter_combo_layout.addWidget(QtWidgets.QLabel("x:"), 0)
        scatter_combo_layout.addWidget(self.combo_scatter_x, 1)
        scatter_combo_layout.addWidget(self.check_scatter_logy, 0)
        scatter_layout.addWidget(self.graph_scatter)
        scatter_layout.addLayout(scatter_combo_layout)
        self.scatter_widget = QtWidgets.QWidget()
        self.scatter_widget.setLayout(scatter_layout)

        self.graph_stack = QtWidgets.QStackedWidget()
        self.graph_stack.addWidget(self.graph_hist)
        self.graph_stack.addWidget(self.graph_frac)
        self.graph_stack.addWidget(self.scatter_widget)

        self.io = ResultIOStack()

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
        self.mode.currentIndexChanged.connect(self.updateOutputs)
        self.mode.currentIndexChanged.connect(self.updateScatterElements)
        self.mode.currentIndexChanged.connect(self.drawGraph)

        self.label_file = QtWidgets.QLabel()

        self.button_export = QtWidgets.QPushButton("Export Results")
        self.button_export.pressed.connect(self.dialogExportResults)

        # Actions
        self.action_graph_histogram = create_action(
            "view-object-histogram-linear",
            "Histogram",
            "Overlay of results histograms.",
            lambda: (
                self.setDrawMode("Overlay"),
                self.graph_stack.setCurrentWidget(self.graph_hist),
            ),
            checkable=True,
        )
        self.action_graph_histogram.setChecked(True)
        self.action_graph_histogram_stacked = create_action(
            "object-rows",
            "Stacked Histograms",
            "Single histogram per result.",
            lambda: (
                self.setDrawMode("Stacked"),
                self.graph_stack.setCurrentWidget(self.graph_hist),
            ),
            checkable=True,
        )
        self.action_graph_histogram.setChecked(True)
        self.action_graph_fractions = create_action(
            "office-chart-bar-stacked",
            "Composition",
            "Show the elemental composition of peaks.",
            lambda: self.graph_stack.setCurrentWidget(self.graph_frac),
            checkable=True,
        )
        self.action_graph_scatter = create_action(
            "office-chart-scatter",
            "Scatter",
            "Create scatter plots of elements.",
            lambda: self.graph_stack.setCurrentWidget(self.scatter_widget),
            checkable=True,
        )

        self.action_filter_detections = create_action(
            "view-filter",
            "Filter Detections",
            "Filter detections based on element compositions.",
            self.dialogFilterDetections,
        )

        self.action_graph_options = create_action(
            "configure",
            "Graph Options",
            "Adjust plotting options.",
            self.dialogGraphOptions,
        )

        self.action_graph_zoomout = create_action(
            "zoom-original",
            "Zoom Out",
            "Reset the plot view.",
            self.graphZoomReset,
        )

        action_group_graph_view = QtGui.QActionGroup(self)
        action_group_graph_view.addAction(self.action_graph_histogram)
        action_group_graph_view.addAction(self.action_graph_histogram_stacked)
        action_group_graph_view.addAction(self.action_graph_fractions)
        action_group_graph_view.addAction(self.action_graph_scatter)
        self.graph_toolbar.addActions(action_group_graph_view.actions())

        self.graph_toolbar.addSeparator()
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding
        )
        self.graph_toolbar.addWidget(spacer)

        self.graph_toolbar.addSeparator()
        self.graph_toolbar.addAction(self.action_filter_detections)
        self.graph_toolbar.addAction(self.action_graph_options)

        self.graph_toolbar.addSeparator()
        self.graph_toolbar.addAction(self.action_graph_zoomout)

        # Layouts

        self.io.layout_top.insertWidget(
            0, QtWidgets.QLabel("Mode:"), 0, QtCore.Qt.AlignLeft
        )
        self.io.layout_top.insertWidget(1, self.mode, 0, QtCore.Qt.AlignLeft)
        self.io.layout_top.insertStretch(2, 1)

        layout_filename = QtWidgets.QHBoxLayout()
        layout_filename.addWidget(self.label_file, 1, QtCore.Qt.AlignLeft)
        layout_filename.addWidget(self.button_export, 0, QtCore.Qt.AlignRight)

        # layout_chart_options = QtWidgets.QHBoxLayout()
        # layout_chart_options.addWidget(self.button_export_image)
        # layout_chart_options.addStretch(1)

        layout_graph = QtWidgets.QHBoxLayout()
        layout_graph.addWidget(self.graph_toolbar, 0)
        layout_graph.addWidget(self.graph_stack, 1)

        layout_main = QtWidgets.QVBoxLayout()
        layout_main.addWidget(self.io, 0)
        layout_main.addLayout(layout_graph, 1)
        layout_main.addLayout(layout_filename)

        layout = QtWidgets.QHBoxLayout()
        layout.addLayout(layout_main, 1)
        self.setLayout(layout)

    def setBinWidths(self, widths: Dict[str, float | None]) -> None:
        self.graph_hist_bin_widths.update(widths)
        self.drawGraphHist()

    def setColorScheme(self, scheme: str) -> None:
        self.color_scheme = scheme
        self.drawGraph()

    def setDrawMode(self, mode: str) -> None:
        self.draw_mode = mode
        self.drawGraphHist()

    def setFilters(self, filters) -> None:
        self.filters = filters
        self.updateResults()

    def setHistogramFit(self, fit: str | None) -> None:
        self.graph_hist_fit = fit or None  # for fit == ''
        self.drawGraphHist()

    def dialogGraphOptions(self) -> None:
        if self.graph_stack.currentWidget() == self.graph_hist:
            dlg = HistogramOptionsDialog(
                self.graph_hist_fit, self.graph_hist_bin_widths, self.window()
            )
            dlg.fitChanged.connect(self.setHistogramFit)
            dlg.binWidthsChanged.connect(self.setBinWidths)
        dlg.show()

    def dialogExportResults(self) -> None:
        file, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export", "", "CSV Documents (*.csv)"
        )
        if file != "":
            export_nanoparticle_results(Path(file), self.results)

    # def dialogExportImage(self) -> None:
    #     file, _ = QtWidgets.QFileDialog.getSaveFileName(
    #         self, "Export Image", "", "PNG Images (*.png)"
    #     )
    #     # if file != "":
    #     #     self.chartview.saveToFile(file)

    def dialogFilterDetections(self) -> None:
        dlg = FilterDialog(list(self.results.keys()), self.filters, parent=self)
        dlg.filtersChanged.connect(self.setFilters)
        dlg.open()

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
        if len(self.results) > 1:
            self.drawGraphFractions()
            self.drawGraphScatter()

    def drawGraphHist(self) -> None:
        self.graph_hist.clear()
        mode = self.mode.currentText()

        if mode == "Signal":
            label, unit = "Intensity (counts)", ""
            bin_width = self.graph_hist_bin_widths.get("signal", None)
        elif mode == "Mass (kg)":
            label, unit = "Mass", "g"
            bin_width = self.graph_hist_bin_widths.get("mass", None)
            if bin_width is not None:
                bin_width *= 1000  # convert to gram
        elif mode == "Size (m)":
            label, unit = "Size", "m"
            bin_width = self.graph_hist_bin_widths.get("size", None)
        elif mode == "Conc. (mol/L)":
            label, unit = "Concentration", "mol/L"
            bin_width = self.graph_hist_bin_widths.get("concentration", None)
        else:
            raise ValueError("drawGraphHist: unknown mode")

        names = list(self.results.keys())

        graph_data = {}
        for name, result in self.results.items():
            indices = self.results[name]["indicies"]
            if indices.size < 2:
                continue
            if mode == "Signal":
                graph_data[name] = result["detections"][indices]
            elif mode == "Mass (kg)" and "masses" in result:
                graph_data[name] = result["masses"][indices] * 1000  # convert to gram
            elif mode == "Size (m)" and "sizes" in result:
                graph_data[name] = result["sizes"][indices]
            elif mode == "Conc. (mol/L)" and "cell_concentrations" in result:
                graph_data[name] = result["cell_concentrations"][indices]
            else:
                continue

        # median 'sturges' bin width
        if bin_width is None:
            bin_width = np.median(
                [
                    np.ptp(graph_data[name]) / (np.log2(graph_data[name].size) + 1)
                    for name in graph_data
                ]
            )
        # Limit maximum number of bins
        for data in graph_data.values():
            min_bin_width = (data.max() - data.min()) / 1024
            if bin_width < min_bin_width:
                logger.warning("drawGraphHist: exceeded maximum bins, setting to 1024")
                bin_width = min_bin_width
                break

        scheme = color_schemes[self.color_scheme]
        for i, name in enumerate(graph_data):
            bins = np.arange(
                graph_data[name].min(), graph_data[name].max() + bin_width, bin_width
            )
            bins -= bins[0] % bin_width  # align bins
            color = QtGui.QColor(scheme[names.index(name) % len(scheme)])
            if self.draw_mode == "Overlay":
                plot_name = "Overlay"
                width = 1.0 / len(graph_data)
                if len(graph_data) == 1:
                    width /= 2.0
                offset = i * width
            elif self.draw_mode == "Stacked":
                plot_name = name
                width = 0.5
                offset = 0.0
            else:
                raise ValueError("drawGraphHist: invalid draw mode")

            plot = self.graph_hist.getHistogramPlot(plot_name, xlabel=label, xunit=unit)
            hist, centers = plot.drawData(  # type: ignore
                name,
                graph_data[name],
                bins=bins,
                bar_width=width,
                bar_offset=offset,
                brush=QtGui.QBrush(color),
            )
            if self.draw_mode != "Overlay" and self.graph_hist_fit is not None:
                hist = hist / bin_width / graph_data[name].size
                xs = np.linspace(centers[0] - bin_width, centers[-1] + bin_width, 1024)
                if self.graph_hist_fit == "normal":
                    fit = fit_normal(centers, hist)[2]
                    ys = normal_pdf(xs * fit[2], fit[0], fit[1])
                elif self.graph_hist_fit == "log normal":
                    fit = fit_lognormal(centers, hist)[2]
                    ys = lognormal_pdf(xs + fit[2], fit[0], fit[1])
                else:
                    raise ValueError(
                        f"drawGraphHist: unknown fit {self.graph_hist_fit}"
                    )

                ys = ys * bin_width * graph_data[name].size
                pen = QtGui.QPen(QtCore.Qt.red, 2.0)
                pen.setCosmetic(True)
                plot.drawFit(xs, ys, pen=pen)
        self.graph_hist.zoomReset()

    def drawGraphFractions(self) -> None:
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
            raise ValueError("drawGraphFractions: unknown mode")

        self.graph_frac.plot.setTitle(f"{label} Composition")

        # Save names order to preserve colors
        names = list(self.results.keys())

        # Get list of any un filter detection
        valid = np.zeros(self.results[names[0]]["detections"].size, dtype=bool)
        for result in self.results.values():
            valid[result["indicies"]] = True

        num_valid = np.count_nonzero(valid)
        if num_valid == 0:
            return

        graph_data = {}
        for name, result in self.results.items():
            if mode == "Signal":
                graph_data[name] = result["detections"][valid]
            elif mode == "Mass (kg)" and "masses" in result:
                graph_data[name] = result["masses"][valid] * 1000  # convert to gram
            elif mode == "Size (m)" and "sizes" in result:
                graph_data[name] = result["sizes"][valid]
            elif mode == "Conc. (mol/L)" and "cell_concentrations" in result:
                graph_data[name] = result["cell_concentrations"][valid]
            else:
                continue

        if len(graph_data) == 0:
            return

        fractions = prepare_data_for_clustering(graph_data)
        # fractions = np.empty((num_valid, len(graph_data)), dtype=np.float64)
        # for i, name in enumerate(graph_data):
        #     fractions[:, i] = graph_data[name]

        if fractions.shape[0] == 1:  # single peak
            means, counts = fractions, np.array([1])
        elif fractions.shape[1] == 1:  # single element
            means, counts = np.array([[1.0]]), np.array([np.count_nonzero(fractions)])
        else:
            means, counts = agglomerative_cluster(fractions, 0.05)

        compositions = np.empty(
            counts.size, dtype=[(name, np.float64) for name in graph_data]
        )
        for i, name in enumerate(graph_data):
            compositions[name] = means[:, i]

        mask = counts > fractions.shape[0] * 0.05
        compositions = compositions[mask]
        counts = counts[mask]

        if counts.size == 0:
            return

        scheme = color_schemes[self.color_scheme]
        brushes = []

        for name in compositions.dtype.names:
            color = QtGui.QColor(scheme[names.index(name) % len(scheme)])
            brushes.append(QtGui.QBrush(color))

        self.graph_frac.drawData(compositions, counts, brushes=brushes)

    def drawGraphScatter(self) -> None:
        self.graph_scatter.clear()

        # Set the elements
        xname = self.combo_scatter_x.currentText()
        yname = self.combo_scatter_y.currentText()
        mode = self.mode.currentText()
        if mode == "Signal":
            label, unit = "Intensity (counts)", ""
            x = self.results[xname]["detections"]
            y = self.results[yname]["detections"]
        elif mode == "Mass (kg)":
            label, unit = "Mass", "g"
            x = self.results[xname]["masses"] * 1000
            y = self.results[yname]["masses"] * 1000
        elif mode == "Size (m)":
            label, unit = "Size", "m"
            x = self.results[xname]["sizes"]
            y = self.results[yname]["sizes"]
        elif mode == "Conc. (mol/L)":
            label, unit = "Concentration", "mol/L"
            x = self.results[xname]["cell_concentrations"]
            y = self.results[yname]["cell_concentrations"]
        else:
            raise ValueError("drawGraphScatter: unknown mode")

        valid = np.intersect1d(
            self.results[xname]["indicies"],
            self.results[yname]["indicies"],
            assume_unique=True,
        )

        num_valid = np.count_nonzero(valid)
        if num_valid == 0:
            return

        self.graph_scatter.xaxis.setLabel(text=label, units=unit)
        self.graph_scatter.yaxis.setLabel(text=label, units=unit)

        self.graph_scatter.drawData(
            x[valid],
            y[valid],
            logx=self.check_scatter_logx.isChecked(),
            logy=self.check_scatter_logy.isChecked(),
        )
        if num_valid > 2:
            self.graph_scatter.drawFit(
                x[valid],
                y[valid],
                1,
                logx=self.check_scatter_logx.isChecked(),
                logy=self.check_scatter_logy.isChecked(),
            )

    def graphZoomReset(self) -> None:
        self.graph_frac.zoomReset()
        self.graph_hist.zoomReset()

    def updateScatterElements(self) -> None:
        mode = self.mode.currentText()
        if mode == "Signal":
            key = "detections"
        elif mode == "Mass (kg)":
            key = "masses"
        elif mode == "Size (m)":
            key = "sizes"
        elif mode == "Conc. (mol/L)":
            key = "cell_concentrations"
        else:
            raise ValueError("updateScatterElements: unknown mode")

        elements = [name for name in self.results if key in self.results[name]]

        for i, combo in enumerate([self.combo_scatter_x, self.combo_scatter_y]):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(elements)
            if current in elements:
                combo.setCurrentText(current)
            elif len(elements) > 1:
                combo.setCurrentIndex(i)
            combo.blockSignals(False)

    def updateOutputs(self) -> None:
        mode = self.mode.currentText()

        self.io.repopulate(list(self.results.keys()))

        for name, result in self.results.items():
            if mode == "Signal":
                units = signal_units
                values = result["detections"]
                lod = result["lod"]
            elif mode == "Mass (kg)" and "masses" in result:
                units = mass_units
                values = result["masses"]
                lod = result["lod_mass"]
            elif mode == "Size (m)" and "sizes" in result:
                units = size_units
                values = result["sizes"]
                lod = result["lod_size"]
            elif mode == "Conc. (mol/L)" and "cell_concentrations" in result:
                units = molar_concentration_units
                values = result["cell_concentrations"]
                lod = result["lod_cell_concentration"]
            else:
                self.io[name].clearOutputs()
                continue

            indicies = result["indicies"]

            self.io[name].updateOutputs(
                values[indicies],
                units,
                lod,
                count=indicies.size,
                count_percent=indicies.size / values.size * 100.0,
                count_error=np.sqrt(indicies.size),
                conc=result.get("concentration", None),
                number_conc=result.get("number_concentration", None),
                background_conc=result.get("background_concentration", None),
                background_error=result["background_std"] / result["background"],
            )

    def filterResults(self) -> None:
        condition = np.ones(self.sample.detections.size, dtype=bool)
        for filt in self.filters:
            boolean, name, unit, operation, value = filt

            ops = {
                ">": np.greater,
                "<": np.less,
                ">=": np.greater_equal,
                "<=": np.less_equal,
                "==": np.equal,
            }
            bool_ops = {"And": np.logical_and, "Or": np.logical_or}

            indicies = self.results[name]["indicies"]
            if unit == "Intensity":
                data = self.results[name]["detections"]
            elif unit == "Mass" and "masses" in self.results[name]:
                data = self.results[name]["masses"]
            elif unit == "Size" and "sizes" in self.results[name]:
                data = self.results[name]["sizes"]
            else:
                continue

            valid = ops[operation](data, value)
            condition = bool_ops[boolean](condition, valid)

        valid_indicies = np.flatnonzero(condition)
        for name in self.results:
            indicies = self.results[name]["indicies"]
            self.results[name]["indicies"] = indicies[np.in1d(indicies, valid_indicies)]

    def updateResults(self) -> None:
        method = self.options.efficiency_method.currentText()

        self.results.clear()

        self.label_file.setText(f"Results for: {self.sample.label_file.text()}")

        dwelltime = self.options.dwelltime.baseValue()
        uptake = self.options.uptake.baseValue()

        names = list(self.sample.detections.dtype.names)

        for name in names:
            trim = self.sample.trimRegion(name)
            responses = self.sample.responses[name][trim[0] : trim[1]]

            indicies = np.flatnonzero(self.sample.detections[name])

            result = {
                "background": np.mean(responses[self.sample.labels == 0]),
                "background_std": np.std(responses[self.sample.labels == 0]),
                "detections": self.sample.detections[name],
                "indicies": indicies,
                "total_detections": self.sample.detections.size,
                "events": responses.size,
                "file": self.sample.label_file.text(),
                "limit_method": f"{self.sample.limits[name][0]},{','.join(f'{k}={v}' for k,v in self.sample.limits[name][1].items())}",
                "lod": self.sample.limits[name][2]["ld"],
                "inputs": {"dwelltime": dwelltime},
            }
            if self.options.check_use_window.isChecked():
                result["limit_window"] = int(self.options.window_size.text())

            if method in ["Manual Input", "Reference Particle"]:
                try:
                    if method == "Manual Input":
                        efficiency = float(self.options.efficiency.text())
                    elif method == "Reference Particle":
                        efficiency = self.reference.getEfficiency(name)
                    else:
                        efficiency = None
                except ValueError:
                    efficiency = None

                density = self.sample.io[name].density.baseValue()
                response = self.sample.io[name].response.baseValue()
                time = result["events"] * dwelltime

                try:
                    mass_fraction = float(self.sample.io[name].massfraction.text())
                except ValueError:
                    mass_fraction = None

                if (
                    dwelltime is not None
                    and density is not None
                    and efficiency is not None
                    and mass_fraction is not None
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
                            mass_fraction=mass_fraction,
                            uptake=uptake,
                            response=response,
                            time=time,
                        )
                    )
                    result["inputs"].update(
                        {
                            "density": density,
                            "transport_efficiency": efficiency,
                            "mass_fraction": mass_fraction,
                            "uptake": uptake,
                            "response": response,
                            "time": time,
                        }
                    )
            elif method == "Mass Response" and name in self.reference.io:
                try:
                    mass_fraction = float(self.sample.io[name].massfraction.text())
                except ValueError:
                    mass_fraction = None

                density = self.sample.io[name].density.baseValue()
                mass_response = self.reference.io[name].massresponse.baseValue()

                if (
                    density is not None
                    and mass_fraction is not None
                    and mass_response is not None
                ):
                    result.update(
                        results_from_mass_response(
                            result["detections"],
                            result["background"],
                            result["lod"],
                            density=density,
                            mass_fraction=mass_fraction,
                            mass_response=mass_response,
                        )
                    )
                    result["inputs"].update(
                        {
                            "density": density,
                            "mass_fraction": mass_fraction,
                            "mass_response": mass_response,
                        }
                    )
            # end if method

            # Cell inputs
            cell_diameter = self.options.celldiameter.baseValue()
            molar_mass = self.sample.io[name].molarmass.baseValue()

            if (
                cell_diameter is not None and "sizes" in result
            ):  # Scale sizes to hypothesised
                scale = cell_diameter / np.mean(result["sizes"])
                result["sizes"] *= scale
                result["lod_size"] *= scale
                result["inputs"].update({"cell_diameter": cell_diameter})

            if (
                cell_diameter is not None and molar_mass is not None
            ):  # Calculate the intracellular concetrations
                result["cell_concentrations"] = cell_concentration(
                    result["masses"],
                    diameter=cell_diameter,
                    molar_mass=molar_mass,
                )
                result["lod_cell_concentration"] = cell_concentration(
                    result["lod_mass"],
                    diameter=cell_diameter,
                    molar_mass=molar_mass,
                )
                result["inputs"].update({"molar_mass": molar_mass})

            self.results[name] = result

        self.filterResults()
        # end for name in names
        self.updateOutputs()
        self.updateScatterElements()
        self.updateEnabledItems()

        self.drawGraph()

    def updateEnabledItems(self) -> None:
        # Only enable modes that have data
        for key, index in zip(["masses", "sizes", "cell_concentrations"], [1, 2, 3]):
            enabled = any(key in result for result in self.results.values())
            if not enabled and self.mode.currentIndex() == index:
                self.mode.setCurrentIndex(0)
            self.mode.model().item(index).setEnabled(enabled)

        # Only enable fraction view and stack if more than one element
        single_result = len(self.results) == 1
        self.action_graph_fractions.setEnabled(not single_result)
        self.action_graph_histogram_stacked.setEnabled(not single_result)
        self.action_graph_scatter.setEnabled(not single_result)
        if single_result:  # Switch to histogram
            self.action_graph_histogram.trigger()
