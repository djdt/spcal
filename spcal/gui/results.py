import logging
from pathlib import Path
from typing import Any

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.cluster import agglomerative_cluster, prepare_data_for_clustering
from spcal.gui.dialogs.export import ExportDialog
from spcal.gui.dialogs.filter import FilterDialog
from spcal.gui.dialogs.graphoptions import (
    CompositionsOptionsDialog,
    HistogramOptionsDialog,
    ScatterOptionsDialog,
)
from spcal.gui.graphs.base import SinglePlotGraphicsView
from spcal.gui.graphs.composition import CompositionView
from spcal.gui.graphs.histogram import HistogramView
from spcal.gui.graphs.pca import PCAView
from spcal.gui.graphs.scatter import ScatterView
from spcal.gui.inputs import ReferenceWidget, SampleWidget
from spcal.gui.iowidgets import ResultIOStack
from spcal.gui.options import OptionsWidget
from spcal.gui.util import create_action
from spcal.result import ClusterFilter, Filter, SPCalResult
from spcal.siunits import (
    mass_units,
    molar_concentration_units,
    signal_units,
    size_units,
    volume_units,
)

logger = logging.getLogger(__name__)


class ResultsWidget(QtWidgets.QWidget):
    mode_keys = {
        "Signal": "signal",
        "Mass": "mass",
        "Size": "size",
        "Volume": "volume",
        "Concentration": "cell_concentration",
    }
    mode_units = {
        "Signal": signal_units,
        "Mass": mass_units,
        "Size": size_units,
        "Volume": volume_units,
        "Concentration": molar_concentration_units,
    }
    mode_labels = {  # these differ from SPCalResult.base_units
        "Signal": ("Intensity (counts)", "", 1.0),
        "Mass": ("Mass", "g", 1e3),
        "Size": ("Size", "m", 1.0),
        "Volume": ("Volume", "m³", 1.0),
        "Concentration": ("Concentration", "mol/L", 1.0),
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

        self.filters: list[list[Filter]] = []
        self.cluster_filters: list[ClusterFilter] = []
        # Graph default options
        self.graph_options: dict[str, Any] = {
            "histogram": {
                "draw filtered": False,
                "mode": "overlay",
                "fit": "log normal",
                "bin widths": {
                    "signal": None,
                    "mass": None,
                    "size": None,
                    "volume": None,
                    "cell_concentration": None,
                },
                "percentile": 95.0,
            },
            "composition": {"distance": 0.03, "minimum size": "5%", "mode": "pie"},
            "scatter": {"draw filtered": False, "weighting": "none"},
            # "pca": {"draw filtered": False},  # while it is possible to do this, it doesn't make sense
        }

        self.results: dict[str, SPCalResult] = {}
        # for load on demand, see self.clusters property
        self._clusters: dict[str, np.ndarray] | None = None

        self.update_required = True
        self.redraw_required = {
            "histogram": True,
            "composition": False,
            "scatter": False,
            "pca": False,
        }

        self.graph_toolbar = QtWidgets.QToolBar()
        self.graph_toolbar.setOrientation(QtCore.Qt.Vertical)
        self.graph_toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)

        self.graph_hist = HistogramView()
        self.graph_composition = CompositionView()
        self.graph_scatter = ScatterView()
        self.graph_pca = PCAView()

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

        self.combo_pca_colour = QtWidgets.QComboBox()
        self.combo_pca_colour.currentIndexChanged.connect(self.drawGraphPCA)

        pca_layout = QtWidgets.QVBoxLayout()
        pca_toolbar = QtWidgets.QHBoxLayout()
        pca_toolbar.addWidget(QtWidgets.QLabel("Colour:"), 0)
        pca_toolbar.addWidget(self.combo_pca_colour, 1)
        pca_layout.addWidget(self.graph_pca, 1)
        pca_layout.addLayout(pca_toolbar, 0)

        self.pca_widget = QtWidgets.QWidget()
        self.pca_widget.setLayout(pca_layout)

        self.graph_stack = QtWidgets.QStackedWidget()
        self.graph_stack.addWidget(self.graph_hist)
        self.graph_stack.addWidget(self.graph_composition)
        self.graph_stack.addWidget(self.scatter_widget)
        self.graph_stack.addWidget(self.pca_widget)

        self.io = ResultIOStack()
        self.io.nameChanged.connect(self.updateGraphsForName)

        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(list(self.mode_keys.keys()))
        self.mode.setItemData(0, "Accumulated detection signal.", QtCore.Qt.ToolTipRole)
        self.mode.setItemData(
            1, "Particle mass, requires calibration.", QtCore.Qt.ToolTipRole
        )
        self.mode.setItemData(
            2, "Particle size, requires calibration.", QtCore.Qt.ToolTipRole
        )
        self.mode.setItemData(
            3,
            "Particle volume, requires calibration.",
            QtCore.Qt.ToolTipRole,
        )
        self.mode.setItemData(
            4,
            "Intracellular concentration, requires cell diameter and molarmass.",
            QtCore.Qt.ToolTipRole,
        )
        self.mode.setCurrentText("Signal")
        self.mode.currentIndexChanged.connect(self.updateOutputs)
        self.mode.currentIndexChanged.connect(self.updateScatterElements)
        self.mode.currentIndexChanged.connect(self.updatePCAElements)
        self.mode.currentIndexChanged.connect(self.redraw)

        self.label_file = QtWidgets.QLabel()

        self.button_export = QtWidgets.QPushButton("Export Results")
        self.button_export.pressed.connect(self.dialogExportResults)

        # Actions
        self.action_graph_histogram = create_action(
            "view-object-histogram-logarithmic",
            "Histogram",
            "Overlay of results histograms.",
            lambda: (
                self.setHistDrawMode("overlay"),
                self.drawIfRequired("histogram"),
                self.graph_stack.setCurrentWidget(self.graph_hist),
            ),
            checkable=True,
        )
        self.action_graph_histogram.setChecked(True)
        self.action_graph_histogram_single = create_action(
            "view-object-histogram-linear",
            "Stacked Histograms",
            "Single histogram per result.",
            lambda: (
                self.setHistDrawMode("single"),
                self.drawIfRequired("histogram"),
                self.graph_stack.setCurrentWidget(self.graph_hist),
            ),
            checkable=True,
        )
        self.action_graph_histogram.setChecked(True)
        self.action_graph_compositions = create_action(
            "office-chart-pie",
            "Composition",
            "Show the elemental composition of peaks.",
            lambda: (
                self.drawIfRequired("composition"),
                self.graph_stack.setCurrentWidget(self.graph_composition),
            ),
            checkable=True,
        )
        self.action_graph_scatter = create_action(
            "office-chart-scatter",
            "Scatter",
            "Create scatter plots of elements.",
            lambda: (
                self.drawIfRequired("scatter"),
                self.graph_stack.setCurrentWidget(self.scatter_widget),
            ),
            checkable=True,
        )
        self.action_graph_pca = create_action(
            "skg-chart-bubble",
            "PCA",
            "Create PCA plot of detections.",
            lambda: (
                self.drawIfRequired("pca"),
                self.graph_stack.setCurrentWidget(self.pca_widget),
            ),
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
        action_group_graph_view.addAction(self.action_graph_histogram_single)
        action_group_graph_view.addAction(self.action_graph_compositions)
        action_group_graph_view.addAction(self.action_graph_scatter)
        action_group_graph_view.addAction(self.action_graph_pca)
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

    @property
    def clusters(self) -> dict[str, np.ndarray]:
        if self._clusters is None:
            self._clusters = {}
            for mode, key in self.mode_keys.items():
                data = self.resultsForKey(key)
                idx = self.filterIndicies(clusters=False)
                # data = data[Filter.filter_results(self.filters, self.results)]
                #     key, include_zeros=True, filter=True, filter_clusters=False
                # )
                if len(data) == 0 or idx.size == 0:
                    continue
                for k, v in data.items():
                    data[k] = v[idx]

                X = prepare_data_for_clustering(data)
                T = agglomerative_cluster(
                    X, self.graph_options["composition"]["distance"]
                )
                self._clusters[key] = T
        return self._clusters

    def filterIndicies(self, clusters: bool = True) -> np.ndarray:
        try:
            idx = np.arange(next(iter(self.results.values())).detections.size)
        except StopIteration:
            return np.array([], dtype=int)

        idx = np.intersect1d(
            idx, Filter.filter_results(self.filters, self.results), assume_unique=True
        )
        if clusters and len(self.cluster_filters) > 0:
            idx = np.intersect1d(
                idx,
                ClusterFilter.filter_clusters(self.cluster_filters, self.clusters),
                assume_unique=True,
            )
        return idx

    def resultsForKey(self, key: str) -> dict[str, np.ndarray]:
        """Function to get results without filtering."""
        return {
            k: v.calibrated(key, use_indicies=False)
            for k, v in self.results.items()
            if v.canCalibrate(key)
        }

    def colorForName(self, name: str) -> QtGui.QColor:
        return self.sample.colorForName(name)

    def updateNames(self, names: dict[str, str]) -> None:
        for old, new in names.items():
            if old == new:  # pragma: no cover
                continue
            if old in self.results:
                self.results[new] = self.results.pop(old)
            if self._clusters is not None and old in self._clusters:
                self._clusters[new] = self._clusters.pop(old)
            if old in self.io:
                index = self.io.combo_name.findText(old)
                self.io.combo_name.setItemText(index, new)

            for group in self.filters:
                for filter in group:
                    if filter.name == old:
                        filter.name = new

        self.updateScatterElements()
        self.updatePCAElements()
        self.redraw()

    def setFilters(
        self, filters: list[list[Filter]], cluster_filters: list[ClusterFilter]
    ) -> None:
        self.filters = filters
        self.cluster_filters = cluster_filters
        self.updateResults()

    def setCompDistance(self, distance: float) -> None:
        self.graph_options["composition"]["distance"] = distance
        self._clusters = None
        self.drawGraphCompositions()

    def setCompMode(self, mode: str) -> None:
        self.graph_options["composition"]["mode"] = mode
        self.drawGraphCompositions()

    def setCompSize(self, size: float | str) -> None:
        self.graph_options["composition"]["minimum size"] = size
        self.drawGraphCompositions()

    def setGraphFont(self, font: QtGui.QFont) -> None:
        self.graph_composition.setFont(font)
        self.graph_hist.setFont(font)
        self.graph_pca.setFont(font)
        self.graph_scatter.setFont(font)
        self.graph_stack.setFont(font)

    def setHistDrawMode(self, mode: str) -> None:
        self.graph_options["histogram"]["mode"] = mode
        self.drawGraphHist()

    def setHistBinWidths(self, widths: dict[str, float | None]) -> None:
        self.graph_options["histogram"]["bin widths"].update(widths)
        self.drawGraphHist()

    def setHistFit(self, fit: str | None) -> None:
        self.graph_options["histogram"]["fit"] = fit or None  # for fit == ''
        self.drawGraphHist()

    def setHistPercentile(self, p: float) -> None:
        self.graph_options["histogram"]["percentile"] = p
        self.drawGraphHist()

    def setHistDrawFiltered(self, show: bool) -> None:
        self.graph_options["histogram"]["draw filtered"] = show
        self.drawGraphHist()

    def setScatterWeighting(self, weighting: str) -> None:
        self.graph_options["scatter"]["weighting"] = weighting
        self.drawGraphScatter()

    def setScatterDrawFiltered(self, show: bool) -> None:
        self.graph_options["scatter"]["draw filtered"] = show
        self.drawGraphScatter()

    # Dialogs
    def dialogGraphOptions(
        self,
    ) -> HistogramOptionsDialog | CompositionsOptionsDialog | None:
        if self.graph_stack.currentWidget() == self.graph_hist:
            dlg = HistogramOptionsDialog(
                self.graph_options["histogram"]["fit"],
                self.graph_options["histogram"]["bin widths"],
                self.graph_options["histogram"]["percentile"],
                self.graph_options["histogram"]["draw filtered"],
                parent=self,
            )
            dlg.fitChanged.connect(self.setHistFit)
            dlg.binWidthsChanged.connect(self.setHistBinWidths)
            dlg.percentileChanged.connect(self.setHistPercentile)
            dlg.drawFilteredChanged.connect(self.setHistDrawFiltered)
        elif self.graph_stack.currentWidget() == self.graph_composition:
            dlg = CompositionsOptionsDialog(
                self.graph_options["composition"]["distance"],
                self.graph_options["composition"]["minimum size"],
                self.graph_options["composition"]["mode"],
                parent=self,
            )
            dlg.distanceChanged.connect(self.setCompDistance)
            dlg.minimumSizeChanged.connect(self.setCompSize)
            dlg.modeChanged.connect(self.setCompMode)
        elif self.graph_stack.currentWidget() == self.scatter_widget:
            dlg = ScatterOptionsDialog(
                self.graph_options["scatter"]["weighting"],
                self.graph_options["scatter"]["draw filtered"],
                parent=self,
            )
            dlg.weightingChanged.connect(self.setScatterWeighting)
            dlg.drawFilteredChanged.connect(self.setScatterDrawFiltered)
        else:  # Todo: scatter
            return None
        dlg.show()
        return dlg

    def dialogExportResults(self) -> None:
        path = Path(self.sample.label_file.text())

        regions = self.sample.regions
        times = self.options.dwelltime.baseValue() * (
            regions[:, 0] + (regions[:, 1] - regions[:, 0]) / 2.0
        )

        dlg = ExportDialog(
            path.with_name(path.stem + "_results.csv"),
            self.results,
            self.clusters,
            times,
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
        # if len(self.clusters) > 0:
        #     max_idx = np.amax([idx.max() for idx in self.clusters.values()])
        # else:
        #     max_idx = -1
        dlg = FilterDialog(
            list(self.results.keys()),
            self.filters,
            self.cluster_filters,
            number_clusters=99,
            parent=self,
        )
        dlg.filtersChanged.connect(self.setFilters)
        dlg.open()

    # Plotting
    def drawIfRequired(self, graph: str | None = None) -> None:
        if graph is None:
            w = self.graph_stack.widget(self.graph_stack.currentIndex())
            if w == self.graph_hist:
                graph = "histogram"
            elif w == self.graph_composition:
                graph = "composition"
            elif w == self.scatter_widget:
                graph = "scatter"
            elif w == self.pca_widget:
                graph = "pca"
            else:
                raise ValueError(f"unkown graph widget '{graph}'")

        if self.redraw_required[graph]:
            if graph == "histogram":
                self.drawGraphHist()
            elif graph == "composition":
                self.drawGraphCompositions()
            elif graph == "scatter":
                self.drawGraphScatter()
            elif graph == "pca":
                self.drawGraphPCA()
            else:
                ValueError(f"unkown graph type '{graph}'")
            self.redraw_required[graph] = False

    def redraw(self) -> None:
        for k in self.redraw_required.keys():
            self.redraw_required[k] = True
        self.drawIfRequired()

    def drawGraphHist(self) -> None:
        self.graph_hist.clear()
        mode = self.mode.currentText()

        label, unit, modifier = self.mode_labels[mode]
        key = self.mode_keys[mode]
        bin_width = self.graph_options["histogram"]["bin widths"][key]

        names = (
            [self.io.combo_name.currentText()]
            if self.graph_options["histogram"]["mode"] == "single"
            else self.results.keys()
        )
        graph_data = {
            k: np.maximum(0.0, v)
            for k, v in self.resultsForKey(key).items()
            if k in names
        }
        idx = self.filterIndicies()
        graph_idx = {
            k: np.intersect1d(idx, np.nonzero(v), assume_unique=True)
            for k, v in graph_data.items()
        }
        for k, v in graph_idx.items():
            if v.size == 0:
                graph_data.pop(k)

        if len(graph_data) == 0:
            return

        # median FD bin width
        if bin_width is None:
            bin_width = np.median(
                [
                    2.0
                    * np.subtract(
                        *np.percentile(graph_data[name][graph_idx[name]], [75, 25])
                    )
                    / np.cbrt(graph_idx[name].size)
                    for name in graph_data
                ]
            )
        # Limit maximum / minimum number of bins
        data_range = 0.0
        for name, data in graph_data.items():
            ptp = np.percentile(
                data[graph_idx[name]], self.graph_options["histogram"]["percentile"]
            ) - np.amin(data[graph_idx[name]])
            if ptp > data_range:
                data_range = ptp

        if data_range == 0.0:  # prevent drawing if no range, i.e. one point
            return

        min_bins, max_bins = 10, 1000
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

        for i, (name, data) in enumerate(graph_data.items()):
            color = self.colorForName(name)
            max_bin = np.percentile(
                data[graph_idx[name]], self.graph_options["histogram"]["percentile"]
            )
            bins = np.arange(
                data.min() * modifier, max_bin * modifier + bin_width, bin_width
            )
            if bins.size == 1:
                bins = np.array([bins[0], max_bin])
            bins -= bins[0] % bin_width  # align bins
            if self.graph_options["histogram"]["mode"] == "overlay":
                width = 1.0 / len(graph_data)
                offset = i * width
            elif self.graph_options["histogram"]["mode"] == "single":
                width = 1.0
                offset = 0.0
            else:
                raise ValueError("drawGraphHist: invalid draw mode")

            lod = self.results[name].convertTo(
                self.results[name].limits.detection_limit, key
            )

            non_zero = np.flatnonzero(data)

            limits = {"mean": np.mean(data[graph_idx[name]]) * modifier}
            if isinstance(lod, np.ndarray):
                limits["LOD (mean)"] = np.nanmean(lod) * modifier
            else:
                limits["LOD"] = lod * modifier

            # Auto SI prefix does not work with squared (or cubed) units
            self.graph_hist.xaxis.enableAutoSIPrefix(mode not in ["Signal", "Volume"])

            self.graph_hist.xaxis.setLabel(text=label, units=unit)
            self.graph_hist.draw(
                data[graph_idx[name]] * modifier,
                filtered_data=data[np.setdiff1d(non_zero, idx, assume_unique=True)]
                * modifier,
                bins=bins,
                bar_width=width,
                bar_offset=offset,
                brush=QtGui.QBrush(color),
                name=name,
                draw_fit=self.graph_options["histogram"]["fit"],
                fit_visible=self.graph_options["histogram"]["mode"] == "single",
                draw_limits=limits,
                limits_visible=self.graph_options["histogram"]["mode"] == "single",
                draw_filtered=self.graph_options["histogram"]["draw filtered"],
            )

        self.graph_hist.setDataLimits(xMax=1.0, yMax=1.1)
        self.graph_hist.zoomReset()

    def drawGraphCompositions(self) -> None:
        # composition view
        self.graph_composition.clear()
        mode = self.mode.currentText()
        key = self.mode_keys[mode]

        label, _, _ = self.mode_labels[mode]
        self.graph_composition.plot.setTitle(f"{label} Composition")

        graph_data = self.resultsForKey(
            key
        )  # , include_zeros=True, filter_clusters=False)
        idx = self.filterIndicies(clusters=False)
        if len(graph_data) == 0 or idx.size == 0:
            return
        for k, v in graph_data.items():
            graph_data[k] = v[idx]

        brushes = [QtGui.QBrush(self.colorForName(name)) for name in graph_data.keys()]
        self.graph_composition.draw(
            graph_data,
            self.clusters[self.mode_keys[mode]],
            min_size=self.graph_options["composition"]["minimum size"],
            mode=self.graph_options["composition"]["mode"],
            brushes=brushes,
        )

    def drawGraphPCA(self) -> None:
        self.graph_pca.clear()

        mode = self.mode.currentText()
        key = self.mode_keys[mode]

        label, unit, modifier = self.mode_labels[mode]
        graph_data = self.resultsForKey(key)  # , include_zeros=True)
        idx = self.filterIndicies()
        if len(graph_data) < 2 or idx.size == 0:
            return
        for k, v in graph_data.items():
            graph_data[k] = v[idx]

        X = np.stack(list(graph_data.values()), axis=1)
        brush = QtGui.QBrush(QtCore.Qt.black)

        self.graph_pca.draw(X, brush=brush, feature_names=list(graph_data.keys()))

        colorby = self.combo_pca_colour.currentText()
        if colorby != "None":
            if colorby == "Total":
                idx = np.sum(X, axis=1)
            else:
                idx = X[:, list(graph_data.keys()).index(colorby)]
            idx = np.digitize(idx, np.linspace(idx.min(), idx.max(), 31))
            self.graph_pca.colorScatter(idx)

        self.graph_pca.setDataLimits(xMin=-0.1, xMax=1.1, yMin=-0.1, yMax=1.1)
        self.graph_pca.zoomReset()

    def drawGraphScatter(self) -> None:
        self.graph_scatter.clear()

        # Set the elements
        mode = self.mode.currentText()
        label, unit, modifier = self.mode_labels[mode]
        key = self.mode_keys[mode]

        rx = self.results[self.combo_scatter_x.currentText()]
        ry = self.results[self.combo_scatter_y.currentText()]

        x = rx.calibrated(key, use_indicies=False) * modifier
        y = ry.calibrated(key, use_indicies=False) * modifier

        valid = np.intersect1d(rx.indicies, ry.indicies, assume_unique=True)
        filtered = np.setdiff1d(
            np.intersect1d(np.flatnonzero(x), np.flatnonzero(y), assume_unique=True),
            valid,
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
        if self.graph_options["scatter"]["draw filtered"]:
            self.graph_scatter.drawData(
                x[filtered],
                y[filtered],
                logx=self.check_scatter_logx.isChecked(),
                logy=self.check_scatter_logy.isChecked(),
                pen=QtGui.QPen(QtCore.Qt.gray),
            )
        if num_valid > 2:
            self.graph_scatter.drawFit(
                x[valid],
                y[valid],
                self.scatter_fit_degree.value(),
                logx=self.check_scatter_logx.isChecked(),
                logy=self.check_scatter_logy.isChecked(),
                weighting=self.graph_options["scatter"]["weighting"],
            )

        self.graph_scatter.zoomReset()

    def graphZoomReset(self) -> None:
        widget = self.graph_stack.currentWidget()
        if hasattr(widget, "zoomReset"):
            widget.zoomReset()
        else:
            child = widget.findChild(SinglePlotGraphicsView)
            if child is not None:
                child.zoomReset()

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
            name for name in self.results if self.results[name].canCalibrate(key)
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

    def updatePCAElements(self) -> None:
        mode = self.mode.currentText()
        key = self.mode_keys[mode]

        elements = ["None", "Total"] + [
            name for name in self.results if self.results[name].canCalibrate(key)
        ]
        current = self.combo_pca_colour.currentText()
        self.combo_pca_colour.blockSignals(True)
        self.combo_pca_colour.clear()
        self.combo_pca_colour.addItems(elements)
        if current in elements:
            self.combo_pca_colour.setCurrentText(current)
        else:
            self.combo_pca_colour.setCurrentIndex(0)
        self.combo_pca_colour.blockSignals(False)

    def updateOutputs(self) -> None:
        mode = self.mode.currentText()
        key = self.mode_keys[mode]
        units = self.mode_units[mode]

        previous_name = self.io.combo_name.currentText()
        self.io.repopulate(list(self.results.keys()))
        if previous_name in self.io:
            self.io.combo_name.setCurrentText(previous_name)

        for name, result in self.results.items():
            if not result.canCalibrate(key):
                self.io[name].clearOutputs()
                continue
            lod = self.sample.limits[name].detection_limit
            lod = result.convertTo(lod, key)

            # re-calculate results, they could be filtered
            # indicies = result.indicies

            self.io[name].updateOutputs(
                result.calibrated(key),
                units,
                lod,  # type: ignore
                count=result.number,
                count_percent=result.number / result.detections.size * 100.0,
                count_error=result.number_error,
                conc=result.mass_concentration,
                number_conc=result.number_concentration,
                background_conc=result.ionic_background,
                background_error=result.background / result.background_error,
            )

    def filterResults(self) -> None:
        """Filters the current results.

        Filters are slected in the ``spcal.gui.dialogs.FilterDialog``. Filters are
        stored as a list of groups where each filters are combined by && (logical and).
        Each group is || (logical or'd) together.

        If no filters are selected then this does nothing.
        """
        if len(self.filters) == 0:
            return

        filter_indicies = Filter.filter_results(self.filters, self.results)

        for name in self.results:
            indicies = self.results[name].indicies
            self.results[name]._indicies = np.intersect1d(
                indicies, filter_indicies, assume_unique=True
            )

    def filterClusters(self) -> None:
        if len(self.cluster_filters) == 0:
            return

        filter_indicies = ClusterFilter.filter_clusters(
            self.cluster_filters, self.clusters
        )

        valid = SPCalResult.all_valid_indicies(list(self.results.values()))
        for name in self.results:
            indicies = self.results[name].indicies
            self.results[name]._indicies = np.intersect1d(
                indicies, valid[filter_indicies], assume_unique=True
            )

        # for key in self.clusters.keys():
        #     self._clusters[key] = self._clusters[key][filter_indicies]

    def updateResults(self) -> None:
        method = self.options.efficiency_method.currentText()

        self.results.clear()
        self._clusters = None

        self.label_file.setText(f"Results for: {self.sample.label_file.text()}")

        dwelltime = self.options.dwelltime.baseValue()
        uptake = self.options.uptake.baseValue()

        assert dwelltime is not None
        names = [
            name
            for name in self.sample.detection_names
            if name in self.sample.enabled_names
        ]
        for name in names:
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
                elif method == "Mass Response" and name in self.reference.io:
                    inputs["mass_response"] = self.reference.io[
                        name
                    ].massresponse.baseValue()
            except ValueError:
                pass
            inputs["mass_fraction"] = self.sample.io[name].massfraction.value()

            # No None inputs
            result.inputs.update({k: v for k, v in inputs.items() if v is not None})
            self.results[name] = result
        # end for name in names

        self.filterResults()
        self.filterClusters()  # will call self.clusters to load clusters if needed
        self.updateOutputs()
        self.updateScatterElements()
        self.updatePCAElements()
        self.updateEnabledItems()

        self.redraw()
        self.update_required = False

    def updateEnabledItems(self) -> None:
        # Only enable modes that have data
        for key, index in zip(
            ["mass", "size", "volume", "cell_concentration"], [1, 2, 3, 4]
        ):
            enabled = any(result.canCalibrate(key) for result in self.results.values())
            if not enabled and self.mode.currentIndex() == index:
                self.mode.setCurrentIndex(0)
            self.mode.model().item(index).setEnabled(enabled)

        # Only enable composition view and stack if more than one element
        nresults = sum(result.indicies.size > 0 for result in self.results.values())
        self.action_graph_compositions.setEnabled(nresults > 1)
        self.action_graph_histogram.setEnabled(nresults > 1)
        self.action_graph_scatter.setEnabled(nresults > 1)
        self.action_graph_pca.setEnabled(nresults > 1)
        if nresults == 1:  # Switch to histogram
            self.action_graph_histogram_single.trigger()

    def bestUnitsForResults(self) -> dict[str, tuple[str, float]]:
        best_units: dict[str, tuple[str, float]] = {}
        for key, units in zip(
            SPCalResult.base_units,
            [
                signal_units,
                mass_units,
                size_units,
                volume_units,
                molar_concentration_units,
            ],
        ):
            unit_keys = list(units.keys())
            unit_values = list(units.values())
            for k, result in self.results.items():
                if not result.canCalibrate(key):
                    continue
                mean = np.mean(result.calibrated(key))
                idx = max(np.searchsorted(list(unit_values), mean) - 1, 0)
                if (
                    unit_values[idx]
                    <= best_units.get(key, SPCalResult.base_units[key])[1]
                ):
                    best_units[key] = unit_keys[idx], unit_values[idx]

        return best_units

    def requestUpdate(self) -> None:
        self.update_required = True

    def isUpdateRequired(self) -> bool:
        return self.update_required

    def resetInputs(self) -> None:
        self.filters.clear()
        self.cluster_filters.clear()
        self.results.clear()
        self._clusters = None
