import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.cluster import agglomerative_cluster, prepare_data_for_clustering
from spcal.fit import fit_lognormal, fit_normal, lognormal_pdf, normal_pdf
from spcal.gui.dialogs.export import ExportDialog
from spcal.gui.dialogs.filter import FilterDialog
from spcal.gui.dialogs.graphoptions import (
    CompositionsOptionsDialog,
    HistogramOptionsDialog,
)
from spcal.gui.graphs import color_schemes
from spcal.gui.graphs.base import SinglePlotGraphicsView
# from spcal.gui.graphs.plots import HistogramPlotItem
from spcal.gui.graphs.views import CompositionView, HistogramView, ScatterView
from spcal.gui.inputs import ReferenceWidget, SampleWidget
from spcal.gui.iowidgets import ResultIOStack
from spcal.gui.options import OptionsWidget
from spcal.gui.util import create_action
from spcal.result import SPCalResult
from spcal.siunits import (
    mass_units,
    molar_concentration_units,
    signal_units,
    size_units,
)

logger = logging.getLogger(__name__)


class ResultsWidget(QtWidgets.QWidget):
    mode_labels = {
        "Signal": ("Intensity (counts)", "", 1.0),
        "Mass (kg)": ("Mass", "g", 1e3),
        "Size (m)": ("Size", "m", 1.0),
        "Conc. (mol/L)": ("Concentration", "mol/L", 1.0),
    }
    mode_keys = {
        "Signal": "signal",
        "Mass (kg)": "mass",
        "Size (m)": "size",
        "Conc. (mol/L)": "cell_concentration",
    }

    def __init__(
        self,
        options: OptionsWidget,
        sample: SampleWidget,
        reference: ReferenceWidget,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)

        self.options = options
        self.sample = sample
        self.reference = reference

        self.filters: List[Tuple[str, str, str, str, float]] = []
        # Graph default options
        self.graph_options: Dict[str, Any] = {
            "histogram": {
                "mode": "overlay",
                "fit": "log normal",
                "bin widths": {
                    "signal": None,
                    "mass": None,
                    "size": None,
                    "cell_concentration": None,
                },
            },
            "composition": {"distance": 0.03, "minimum size": "5%"},
        }
        self.results: Dict[str, SPCalResult] = {}

        self.graph_toolbar = QtWidgets.QToolBar()
        self.graph_toolbar.setOrientation(QtCore.Qt.Vertical)
        self.graph_toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)

        self.graph_hist = HistogramView()
        self.graph_composition = CompositionView()
        self.graph_scatter = ScatterView()
        # self.graph_pca = PCAView()

        self.combo_scatter_x = QtWidgets.QComboBox()
        self.combo_scatter_x.currentIndexChanged.connect(self.drawGraphScatter)
        self.combo_scatter_y = QtWidgets.QComboBox()
        self.combo_scatter_y.currentIndexChanged.connect(self.drawGraphScatter)

        self.check_scatter_logx = QtWidgets.QCheckBox("log")
        self.check_scatter_logx.clicked.connect(self.drawGraphScatter)
        self.check_scatter_logy = QtWidgets.QCheckBox("log")
        self.check_scatter_logy.clicked.connect(self.drawGraphScatter)

        self.scatter_fit_degree = QtWidgets.QSpinBox()
        self.scatter_fit_degree.setRange(1, 9)
        self.scatter_fit_degree.setValue(1)
        self.scatter_fit_degree.valueChanged.connect(self.drawGraphScatter)

        # Create simple widget with graph and two combos for x / y element selection
        scatter_layout = QtWidgets.QVBoxLayout()
        scatter_combo_layout = QtWidgets.QHBoxLayout()
        scatter_combo_layout.addWidget(QtWidgets.QLabel("y:"), 0)
        scatter_combo_layout.addWidget(self.combo_scatter_y, 1)
        scatter_combo_layout.addWidget(self.check_scatter_logx, 0)
        scatter_combo_layout.addWidget(QtWidgets.QLabel("x:"), 0)
        scatter_combo_layout.addWidget(self.combo_scatter_x, 1)
        scatter_combo_layout.addWidget(self.check_scatter_logy, 0)
        scatter_combo_layout.addWidget(QtWidgets.QLabel("degree:"), 0)
        scatter_combo_layout.addWidget(self.scatter_fit_degree, 1)
        scatter_layout.addWidget(self.graph_scatter)
        scatter_layout.addLayout(scatter_combo_layout)
        self.scatter_widget = QtWidgets.QWidget()
        self.scatter_widget.setLayout(scatter_layout)

        self.graph_stack = QtWidgets.QStackedWidget()
        self.graph_stack.addWidget(self.graph_hist)
        self.graph_stack.addWidget(self.graph_composition)
        self.graph_stack.addWidget(self.scatter_widget)

        self.io = ResultIOStack()
        self.io.nameChanged.connect(self.updateGraphsForName)

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
        self.mode.currentIndexChanged.connect(self.redraw)

        self.label_file = QtWidgets.QLabel()

        self.button_export = QtWidgets.QPushButton("Export Results")
        self.button_export.pressed.connect(self.dialogExportResults)

        # Actions
        self.action_graph_histogram = create_action(
            "view-object-histogram-linear",
            "Histogram",
            "Overlay of results histograms.",
            lambda: (
                self.setHistDrawMode("overlay"),
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
                self.setHistDrawMode("single"),
                self.graph_stack.setCurrentWidget(self.graph_hist),
            ),
            checkable=True,
        )
        self.action_graph_histogram.setChecked(True)
        self.action_graph_compositions = create_action(
            "office-chart-pie",
            "Composition",
            "Show the elemental composition of peaks.",
            lambda: self.graph_stack.setCurrentWidget(self.graph_composition),
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
        action_group_graph_view.addAction(self.action_graph_compositions)
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

    def colorForName(self, name: str) -> QtGui.QColor:
        scheme = color_schemes[QtCore.QSettings().value("colorscheme", "IBM Carbon")]
        return QtGui.QColor(scheme[self.sample.names.index(name) % len(scheme)])

    def setFilters(self, filters) -> None:
        self.filters = filters
        self.updateResults()

    def setCompDistance(self, distance: float) -> None:
        self.graph_options["composition"]["distance"] = distance
        self.drawGraphCompositions()

    def setCompSize(self, size: float | str) -> None:
        self.graph_options["composition"]["minimum size"] = size
        self.drawGraphCompositions()

    def setHistDrawMode(self, mode: str) -> None:
        self.graph_options["histogram"]["mode"] = mode
        self.drawGraphHist()

    def setHistBinWidths(self, widths: Dict[str, float | None]) -> None:
        self.graph_options["histogram"]["bin widths"].update(widths)
        self.drawGraphHist()

    def setHistFit(self, fit: str | None) -> None:
        self.graph_options["histogram"]["fit"] = fit or None  # for fit == ''
        self.drawGraphHist()

    # Dialogs
    def dialogGraphOptions(self) -> None:
        if self.graph_stack.currentWidget() == self.graph_hist:
            dlg = HistogramOptionsDialog(
                self.graph_options["histogram"]["fit"],
                self.graph_options["histogram"]["bin widths"],
                parent=self,
            )
            dlg.fitChanged.connect(self.setHistFit)
            dlg.binWidthsChanged.connect(self.setHistBinWidths)
        elif self.graph_stack.currentWidget() == self.graph_composition:
            dlg = CompositionsOptionsDialog(
                self.graph_options["composition"]["distance"],
                self.graph_options["composition"]["minimum size"],
                parent=self,
            )
            dlg.distanceChanged.connect(self.setCompDistance)
            dlg.minimumSizeChanged.connect(self.setCompSize)
        else:  # Todo: scatter
            return
        dlg.show()

    def dialogExportResults(self) -> None:
        path = Path(self.sample.label_file.text())
        dlg = ExportDialog(
            path.with_name(path.stem + "_results.csv"),
            self.results,
            units=self.bestUnitsForResults(),
            parent=self,
        )
        dlg.open()

    # def dialogexport_single_particle_results(self) -> None:
    #     file, _ = QtWidgets.QFileDialog.getSaveFileName(
    #         self, "Export Image", "", "PNG Images (*.png)"
    #     )
    #     # if file != "":
    #     #     self.chartview.saveToFile(file)

    def dialogFilterDetections(self) -> None:
        dlg = FilterDialog(list(self.results.keys()), self.filters, parent=self)
        dlg.filtersChanged.connect(self.setFilters)
        dlg.open()

    # Plotting
    def redraw(self) -> None:
        self.drawGraphHist()
        if len(self.results) > 1:
            self.drawGraphCompositions()
            self.drawGraphScatter()

    def drawGraphHist(self) -> None:
        self.graph_hist.plot.clear()
        mode = self.mode.currentText()

        label, unit, modifier = self.mode_labels[mode]
        key = self.mode_keys[mode]
        bin_width = self.graph_options["histogram"]["bin widths"][key]

        graph_data = {}
        lods = {}
        for name, result in self.results.items():
            if (
                self.graph_options["histogram"]["mode"] == "single"
                and name != self.io.combo_name.currentText()
            ):
                continue
            indices = result.indicies
            if indices.size < 2 or key not in result.detections:
                continue
            graph_data[name] = result.detections[key][indices]
            graph_data[name] = np.clip(  # Remove outliers
                graph_data[name], 0.0, np.percentile(graph_data[name], 95)
            )
            lods[name] = result.convertTo(result.limits.detection_threshold, key)

        if len(graph_data) == 0:
            return

        # median FD bin width
        if bin_width is None:
            bin_width = np.median(
                [
                    2.0
                    * np.subtract(*np.percentile(graph_data[name], [75, 25]))
                    / np.cbrt(graph_data[name].size)
                    for name in graph_data
                ]
            )
        # Limit maximum / minimum number of bins
        data_range = np.ptp(np.concatenate(list(graph_data.values())))
        min_bins, max_bins = 16, 1024
        if bin_width < data_range / max_bins:
            logger.warning(
                f"drawGraphHist: exceeded maximum bins, setting to {max_bins}"
            )
            bin_width = data_range / max_bins
        elif bin_width > data_range / min_bins:
            logger.warning(
                f"drawGraphHist: less than minimum bins, setting to {min_bins}"
            )
            bin_width = data_range / min_bins
        bin_width *= modifier  # convert to base unit (kg -> g)

        for i, name in enumerate(graph_data):
            color = self.colorForName(name)
            bins = np.arange(
                graph_data[name].min() * modifier,
                graph_data[name].max() * modifier + bin_width,
                bin_width,
            )
            bins -= bins[0] % bin_width  # align bins
            if self.graph_options["histogram"]["mode"] == "overlay":
                width = 1.0 / len(graph_data)
                offset = i * width
            elif self.graph_options["histogram"]["mode"] == "single":
                width = 1.0
                offset = 0.0
            else:
                raise ValueError("drawGraphHist: invalid draw mode")

            self.graph_hist.xaxis.setLabel(text=label, units=unit)
            hist, centers = self.graph_hist.drawData(  # type: ignore
                name,
                graph_data[name] * modifier,
                bins=bins,
                bar_width=width,
                bar_offset=offset,
                brush=QtGui.QBrush(color),
            )

            visible = not self.graph_options["histogram"]["mode"] == "overlay"
            if self.graph_options["histogram"]["fit"] is not None:
                hist = hist / bin_width / graph_data[name].size
                xs = np.linspace(centers[0] - bin_width, centers[-1] + bin_width, 1024)
                if self.graph_options["histogram"]["fit"] == "normal":
                    fit = fit_normal(centers, hist)[2]
                    ys = normal_pdf(xs * fit[2], fit[0], fit[1])
                elif self.graph_options["histogram"]["fit"] == "log normal":
                    fit = fit_lognormal(centers, hist)[2]
                    ys = lognormal_pdf(xs + fit[2], fit[0], fit[1])
                else:
                    raise ValueError(
                        "drawGraphHist: unknown fit "
                        f"{self.graph_options['histogram']['fit']}"
                    )

                ys = ys * bin_width * graph_data[name].size
                pen = QtGui.QPen(color, 1.0)
                pen.setCosmetic(True)
                self.graph_hist.drawFit(xs, ys, pen=pen, name=name, visible=visible)

            # Draw all the limits
            pen = QtGui.QPen(color, 2.0, QtCore.Qt.PenStyle.DotLine)
            pen.setCosmetic(True)

            if lods[name] is not None:
                self.graph_hist.drawLimit(
                    np.mean(lods[name]),
                    "LOD",
                    pos=0.95,
                    pen=pen,
                    name=name,
                    visible=visible,
                )
            pen.setStyle(QtCore.Qt.PenStyle.DashLine)
            self.graph_hist.drawLimit(
                np.mean(graph_data[name]),
                "mean",
                pos=0.95,
                pen=pen,
                name=name,
                visible=visible,
            )
        self.graph_hist.setDataLimits(xMax=1.0, yMax=1.1)
        self.graph_hist.zoomReset()

    def drawGraphCompositions(self) -> None:
        # composition view
        self.graph_composition.clear()
        mode = self.mode.currentText()

        label, _, _ = self.mode_labels[mode]
        key = self.mode_keys[mode]

        # self.graph_composition.plot.setTitle(f"{label} Composition")

        # Save names order to preserve colors
        names = list(self.results.keys())

        # Get list of any un filter detection
        valid = np.zeros(self.results[names[0]].detections["signal"].size, dtype=bool)
        for result in self.results.values():
            valid[result.indicies] = True

        num_valid = np.count_nonzero(valid)
        if num_valid == 0:
            return

        graph_data = {}
        for name, result in self.results.items():
            if key not in result.detections:
                continue
            graph_data[name] = result.detections[key][valid]
            # no need to use modifier, normalised

        if len(graph_data) == 0:
            return

        fractions = prepare_data_for_clustering(graph_data)

        if fractions.shape[0] == 1:  # single peak
            means, counts = fractions, np.array([1])
        elif fractions.shape[1] == 1:  # single element
            means, counts = np.array([[1.0]]), np.array([np.count_nonzero(fractions)])
        else:
            means, stds, counts = agglomerative_cluster(
                fractions, self.graph_options["composition"]["distance"]
            )

        compositions = np.empty(
            counts.size, dtype=[(name, np.float64) for name in graph_data]
        )
        for i, name in enumerate(graph_data):
            compositions[name] = means[:, i]

        size = self.graph_options["composition"]["minimum size"]
        # Get minimum size as number
        if isinstance(size, str) and size.endswith("%"):
            size = fractions.shape[0] * float(size.rstrip("%")) / 100.0
        elif isinstance(size, str | float):
            size = float(size)
        else:
            raise ValueError("drawGraphFractions: size is neither float nor a % str")

        mask = counts > size
        compositions = compositions[mask]
        counts = counts[mask]

        if counts.size == 0:
            return

        brushes = []

        assert compositions.dtype.names is not None
        for name in compositions.dtype.names:
            color = self.colorForName(name)
            brushes.append(QtGui.QBrush(color))

        self.graph_composition.drawData(compositions, counts, brushes=brushes)
        # self.graph_composition.drawTitle(compositions, counts, brushes=brushes)

    def drawGraphScatter(self) -> None:
        self.graph_scatter.clear()

        # Set the elements
        xname = self.combo_scatter_x.currentText()
        yname = self.combo_scatter_y.currentText()
        mode = self.mode.currentText()
        label, unit, modifier = self.mode_labels[mode]
        key = self.mode_keys[mode]

        x = self.results[xname].detections[key] * modifier
        y = self.results[yname].detections[key] * modifier

        valid = np.intersect1d(
            self.results[xname].indicies,
            self.results[yname].indicies,
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
                self.scatter_fit_degree.value(),
                logx=self.check_scatter_logx.isChecked(),
                logy=self.check_scatter_logy.isChecked(),
            )

    def graphZoomReset(self) -> None:
        self.graph_composition.zoomReset()
        self.graph_hist.zoomReset()

    def readyForResults(self) -> bool:
        if not self.options.isComplete():
            return False
        if not self.sample.isComplete():
            return False

        method = self.options.efficiency_method.currentText()
        if method != "Manual Input" and not self.reference.isComplete():
            return False
        return True

    def updateGraphsForName(self, name: str) -> None:
        if self.graph_options["histogram"]["mode"] == "single":
            self.drawGraphHist()

    def updateScatterElements(self) -> None:
        mode = self.mode.currentText()
        key = self.mode_keys[mode]

        elements = [
            name for name in self.results if key in self.results[name].detections
        ]

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
            lod = self.sample.limits[name].detection_threshold
            if mode == "Signal":
                units = signal_units
                values = result.detections["signal"]
            elif mode == "Mass (kg)" and "mass" in result.detections:
                units = mass_units
                values = result.detections["mass"]
                lod = result.asMass(lod)  # type: ignore
            elif mode == "Size (m)" and "size" in result.detections:
                units = size_units
                values = result.detections["size"]
                lod = result.asSize(lod)  # type: ignore
            elif mode == "Conc. (mol/L)" and "cell_concentration" in result.detections:
                units = molar_concentration_units
                values = result.detections["cell_concentration"]
                lod = result.asCellConcentration(lod)  # type: ignore
            else:
                self.io[name].clearOutputs()
                continue

            indicies = result.indicies

            self.io[name].updateOutputs(
                values[indicies],
                units,
                lod,  # type: ignore
                count=result.number,
                count_percent=indicies.size / values.size * 100.0,
                count_error=result.number_error,
                conc=result.mass_concentration,
                number_conc=result.number_concentration,
                background_conc=result.ionic_background,
                background_error=result.background / result.background_error,
            )

    def filterResults(self) -> None:
        condition = np.ones(self.sample.detections.size, dtype=bool)
        for filt in self.filters:
            boolean, name, unit, operation, value = filt

            ops: Dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
                ">": np.greater,
                "<": np.less,
                ">=": np.greater_equal,
                "<=": np.less_equal,
                "==": np.equal,
            }
            bool_ops: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
                "And": np.logical_and,
                "Or": np.logical_or,
            }

            indicies = self.results[name].indicies
            if unit == "Intensity":
                data = self.results[name].detections["signal"]
            elif unit == "Mass" and "mass" in self.results[name].detections:
                data = self.results[name].detections["mass"]
            elif unit == "Size" and "size" in self.results[name].detections:
                data = self.results[name].detections["size"]
            else:
                continue

            valid = ops[operation](data, value)
            condition = bool_ops[boolean](condition, valid)

        valid_indicies = np.flatnonzero(condition)
        for name in self.results:
            indicies = self.results[name].indicies
            self.results[name].indicies = indicies[np.in1d(indicies, valid_indicies)]

    def updateResults(self) -> None:
        method = self.options.efficiency_method.currentText()

        self.results.clear()

        self.label_file.setText(f"Results for: {self.sample.label_file.text()}")

        dwelltime = self.options.dwelltime.baseValue()
        uptake = self.options.uptake.baseValue()

        assert dwelltime is not None
        assert self.sample.detections.dtype.names is not None
        for name in self.sample.detections.dtype.names:
            result = self.sample.asResult(name)
            if result.number == 0:
                continue

            inputs = {
                "dwelltime": dwelltime,
                "uptake": uptake,
                "cell_diameter": self.options.celldiameter.baseValue(),
                "molar_mass": self.sample.io[name].molarmass.baseValue(),
                "density": self.sample.io[name].density.baseValue(),
                "response": self.sample.io[name].response.baseValue(),
                "time": result.events * dwelltime,
            }

            try:
                if method == "Manual Input":
                    inputs["efficiency"] = self.options.efficiency.value()
                elif method == "Reference Particle":
                    inputs["efficiency"] = self.reference.getEfficiency(name)
                elif method == "Mass Response":
                    inputs["mass_response"] = self.reference.io[
                        name
                    ].massresponse.baseValue()
            except ValueError:
                pass
            inputs["mass_fraction"] = self.sample.io[name].massfraction.value()

            # No None inputs
            result.inputs.update({k: v for k, v in inputs.items() if v is not None})

            try:
                if method in ["Manual Input", "Reference Particle"]:
                    result.fromNebulisationEfficiency()
                elif method == "Mass Response":
                    result.fromMassResponse()
            except ValueError:
                pass

            self.results[name] = result

        self.filterResults()
        # end for name in names
        self.updateOutputs()
        self.updateScatterElements()
        self.updateEnabledItems()

        self.redraw()

    def updateEnabledItems(self) -> None:
        # Only enable modes that have data
        for key, index in zip(["mass", "size", "cell_concentration"], [1, 2, 3]):
            enabled = any(key in result.detections for result in self.results.values())
            if not enabled and self.mode.currentIndex() == index:
                self.mode.setCurrentIndex(0)
            self.mode.model().item(index).setEnabled(enabled)

        # Only enable composition view and stack if more than one element
        single_result = len(self.results) == 1
        self.action_graph_compositions.setEnabled(not single_result)
        self.action_graph_histogram_stacked.setEnabled(not single_result)
        self.action_graph_scatter.setEnabled(not single_result)
        if single_result:  # Switch to histogram
            self.action_graph_histogram.trigger()

    def bestUnitsForResults(self) -> Dict[str, Tuple[str, float]]:
        best_units = {
            # "signal": ("counts", 1.0),
            "mass": ("kg", 1.0),
            "size": ("m", 1.0),
            "cell_concentration": ("mol/L", 1.0),
        }
        for key, units in zip(
            best_units, [mass_units, size_units, molar_concentration_units]
        ):
            unit_keys = list(units.keys())
            unit_values = list(units.values())
            for result in self.results.values():
                if key not in result.detections:
                    continue
                mean = np.mean(result.detections[key])
                idx = max(np.searchsorted(list(unit_values), mean) - 1, 0)
                if unit_values[idx] < best_units[key][1]:
                    best_units[key] = unit_keys[idx], unit_values[idx]

        best_units["signal"] = ("counts", 1.0)
        return best_units
