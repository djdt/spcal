import logging
import sys
from pathlib import Path
from types import TracebackType

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.datafile import SPCalDataFile
from spcal.gui.batch import BatchProcessDialog
from spcal.gui.dialogs.calculator import CalculatorDialog
from spcal.gui.dialogs.filter import FilterDialog
from spcal.gui.dialogs.graphoptions import (
    CompositionsOptionsDialog,
    HistogramOptionsDialog,
)
from spcal.gui.dialogs.io import ImportDialogBase
from spcal.gui.dialogs.response import ResponseDialog
from spcal.gui.dialogs.tools import MassFractionCalculatorDialog, ParticleDatabaseDialog
from spcal.gui.docks.datafile import SPCalDataFilesDock
from spcal.gui.docks.instrumentoptions import SPCalInstrumentOptionsDock
from spcal.gui.docks.isotopeoptions import SPCalIsotopeOptionsDock
from spcal.gui.docks.limitoptions import SPCalLimitOptionsDock
from spcal.gui.docks.outputs import SPCalOutputsDock
from spcal.gui.docks.toolbar import SPCalToolBar
from spcal.gui.graphs import color_schemes, symbols
from spcal.gui.graphs.base import SinglePlotGraphicsView
from spcal.gui.graphs.composition import CompositionView
from spcal.gui.graphs.histogram import HistogramView
from spcal.gui.graphs.particle import ParticleView
from spcal.gui.io import get_import_dialog_for_path, get_open_spcal_path
from spcal.gui.log import LoggingDialog
from spcal.gui.util import create_action
from spcal.io.session import restoreSession, saveSession
from spcal.isotope import SPCalIsotope
from spcal.processing import (
    SPCalIsotopeOptions,
    SPCalProcessingMethod,
    SPCalProcessingResult,
)

logger = logging.getLogger(__name__)


class SPCalGraph(QtWidgets.QStackedWidget):
    requestRedraw = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Signal Graph")

        settings = QtCore.QSettings()
        font = QtGui.QFont(
            str(settings.value("GraphFont/Family", "SansSerif")),
            pointSize=int(settings.value("GraphFont/PointSize", 10)),  # type: ignore
        )

        self.particle = ParticleView(font=font)
        self.histogram = HistogramView(font=font)
        self.composition = CompositionView(font=font)

        self.addWidget(self.particle)  # type: ignore , works
        self.addWidget(self.histogram)  # type: ignore , works
        self.addWidget(self.composition)  # type: ignore , works

        self.action_view_composition = create_action(
            "office-chart-pie",
            "Composition View",
            "Cluster and view results as pie or bar charts.",
            None,
            checkable=True,
        )
        self.action_view_particle = create_action(
            "office-chart-line",
            "Particle View",
            "View raw signal and detected particle peaks.",
            None,
            checkable=True,
        )
        self.action_view_histogram = create_action(
            "view-object-histogram-linear",
            "Results View",
            "View signal and calibrated results as histograms.",
            None,
            checkable=True,
        )
        self.action_view_composition.triggered.connect(
            lambda: self.setCurrentWidget(self.composition)
        )
        self.action_view_particle.triggered.connect(
            lambda: self.setCurrentWidget(self.particle)
        )
        self.action_view_histogram.triggered.connect(
            lambda: self.setCurrentWidget(self.histogram)
        )

        self.action_view_options = create_action(
            "configure",
            "Graph Options",
            "Set options specific to the current graph.",
            self.dialogGraphOptions,
        )

    def clear(self):
        for i in range(self.count()):
            widget = self.widget(i)
            if isinstance(widget, SinglePlotGraphicsView):
                widget.clear()

    def currentView(self) -> str:
        view = self.currentWidget()
        if view == self.particle:
            return "particle"
        elif view == self.histogram:
            return "histogram"
        elif view == self.composition:
            return "composition"
        else:
            raise ValueError("current view is invalid")

    def setCompositionOptions(self, min_size: str | float, mode: str):
        self.composition.min_size = min_size
        self.composition.mode = mode
        self.requestRedraw.emit()

    def setHistogramOptions(
        self, widths: dict[str, float | None], percentile: float, draw_filtered: bool
    ):
        self.histogram.bin_widths = widths
        self.histogram.max_percentile = percentile
        self.histogram.draw_filtered = draw_filtered
        self.requestRedraw.emit()

    def dialogGraphOptions(self):
        view = self.currentView()
        if view == "histogram":
            dlg = HistogramOptionsDialog(
                bin_widths=self.histogram.bin_widths,
                percentile=self.histogram.max_percentile,
                draw_filtered=self.histogram.draw_filtered,
                parent=self,
            )
            dlg.optionsChanged.connect(self.setHistogramOptions)
        elif view == "composition":
            dlg = CompositionsOptionsDialog(
                minimum_size=self.composition.min_size,
                mode=self.composition.mode,
                parent=self,
            )
            dlg.optionsChanged.connect(self.setCompositionOptions)
        else:
            return

        dlg.open()
        return dlg

    def drawResultsParticle(
        self,
        results: dict[SPCalIsotope, SPCalProcessingResult],
        isotopes: list[SPCalIsotope],
        key: str,
    ):
        scheme = color_schemes[
            str(QtCore.QSettings().value("colorscheme", "IBM Carbon"))
        ]
        keys = list(results.keys())

        start, end = np.inf, 0.0
        for i, isotope in enumerate(isotopes):
            color = QtGui.QColor(scheme[keys.index(isotope) % len(scheme)])
            symbol = symbols[keys.index(isotope) % len(symbols)]

            pen = QtGui.QPen(color, 1.0 * self.devicePixelRatio())
            pen.setCosmetic(True)
            brush = QtGui.QBrush(color)
            self.particle.drawResult(
                results[isotope],
                pen=pen,
                brush=brush,
                scatter_size=5.0 * np.sqrt(self.devicePixelRatio()),
                scatter_symbol=symbol,
            )
            start = min(start, results[isotope].times[0])
            end = max(end, results[isotope].times[-1])
        self.particle.drawRegion(start, end)

    def drawResultsComposition(
        self,
        results: dict[SPCalIsotope, SPCalProcessingResult],
        key: str,
        clusters: np.ndarray,
    ):
        scheme = color_schemes[
            str(QtCore.QSettings().value("colorscheme", "IBM Carbon"))
        ]
        keys = list(results.keys())

        pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 1.0 * self.devicePixelRatio())
        pen.setCosmetic(True)

        brushes = []
        for isotope, result in results.items():
            color = QtGui.QColor(scheme[keys.index(isotope) % len(scheme)])
            color.setAlphaF(0.66)
            brushes.append(QtGui.QBrush(color))
        self.composition.drawResults(results, clusters, key, pen, brushes)

    def drawResultsHistogram(
        self,
        results: dict[SPCalIsotope, SPCalProcessingResult],
        isotopes: list[SPCalIsotope],
        key: str,
    ):
        scheme = color_schemes[
            str(QtCore.QSettings().value("colorscheme", "IBM Carbon"))
        ]
        keys = list(results.keys())

        pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 1.0 * self.devicePixelRatio())
        pen.setCosmetic(True)

        drawable = []
        brushes = []
        for isotope in isotopes:
            if results[isotope].canCalibrate(key):
                drawable.append(results[isotope])
                color = QtGui.QColor(scheme[keys.index(isotope) % len(scheme)])
                color.setAlphaF(0.66)
                brushes.append(QtGui.QBrush(color))
        self.histogram.drawResults(drawable, key, pen=pen, brushes=brushes)


class SPCalMainWindow(QtWidgets.QMainWindow):
    resultsChanged = QtCore.Signal(SPCalDataFile)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("SPCal")
        self.resize(1600, 900)

        self.setDockNestingEnabled(True)

        self.log = LoggingDialog()
        self.log.setWindowTitle("SPCal Log")

        self.graph = SPCalGraph()

        self.toolbar = SPCalToolBar(
            [
                self.graph.action_view_particle,
                self.graph.action_view_histogram,
                self.graph.action_view_composition,
                self.graph.action_view_options,
            ]
        )

        self.instrument_options = SPCalInstrumentOptionsDock()
        self.limit_options = SPCalLimitOptionsDock()
        self.isotope_options = SPCalIsotopeOptionsDock()

        self.files = SPCalDataFilesDock()
        self.outputs = SPCalOutputsDock()

        self.processing_methods = {
            "default": SPCalProcessingMethod(
                self.instrument_options.asInstrumentOptions(),
                self.limit_options.asLimitOptions(),
                {},
                [],
                accumulation_method=self.limit_options.limit_accumulation,
                points_required=self.limit_options.points_required,
                prominence_required=self.limit_options.prominence_required,
                calibration_mode="efficiency",
            )
        }

        self.processing_results: dict[
            SPCalDataFile, dict[SPCalIsotope, SPCalProcessingResult]
        ] = {}
        self.processing_clusters: dict[SPCalDataFile, dict[str, np.ndarray]] = {}

        self.graph.particle.requestPeakProperties.connect(self.dialogPeakProperties)
        self.graph.requestRedraw.connect(self.redraw)

        self.toolbar.isotopeChanged.connect(self.redraw)
        self.toolbar.viewChanged.connect(self.redraw)
        self.toolbar.action_all_isotopes.triggered.connect(self.redraw)
        self.toolbar.action_filter.triggered.connect(self.dialogFilterDetections)

        self.instrument_options.optionsChanged.connect(self.onInstrumentOptionsChanged)
        self.limit_options.optionsChanged.connect(self.onLimitOptionsChanged)
        self.isotope_options.optionChanged.connect(self.onIsotopeOptionChanged)
        self.resultsChanged.connect(self.onResultsChanged)

        self.files.dataFileAdded.connect(self.updateRecentFiles)
        self.files.dataFileChanged.connect(self.updateForDataFile)
        self.files.dataFileRemoved.connect(self.removeFileFromResults)

        self.addToolBar(self.toolbar)

        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.files)
        self.addDockWidget(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.instrument_options
        )
        self.addDockWidget(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.limit_options
        )
        self.addDockWidget(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.isotope_options
        )

        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.outputs)

        self.setCentralWidget(self.graph)

        self.createMenuBar()
        self.updateRecentFiles()

    def clusters(self, data_file: SPCalDataFile, key: str) -> np.ndarray:
        # todo: filter first based on key and save cluster peak data
        if data_file not in self.processing_clusters:
            self.processing_clusters[data_file] = {}
        if key not in self.processing_clusters[data_file]:
            self.processing_clusters[data_file][key] = self.processing_methods[
                "default"
            ].processClusters(self.processing_results[data_file], key)
        return self.processing_clusters[data_file][key]

    def updateForDataFile(self, data_file: SPCalDataFile | None):
        if data_file is None:
            self.isotope_options.setIsotopes([])
            self.toolbar.setIsotopes([])
            self.outputs.setIsotopes([])
            self.outputs.setResults({})
            self.graph.clear()
            return

        self.instrument_options.event_time.setBaseValue(data_file.event_time)

        self.isotope_options.setIsotopes(data_file.selected_isotopes)

        self.toolbar.setIsotopes(data_file.selected_isotopes)
        self.outputs.setIsotopes(data_file.selected_isotopes)

        method = self.processing_methods["default"]
        for isotope in data_file.selected_isotopes:
            if isotope in method.isotope_options:
                self.isotope_options.setIsotopeOption(
                    isotope, self.processing_methods["default"].isotope_options[isotope]
                )
            else:
                method.isotope_options[isotope] = SPCalIsotopeOptions(None, None, None)
        self.reprocess(data_file)

    def removeFileFromResults(self, data_file: SPCalDataFile):
        self.processing_results.pop(data_file)

    def redraw(self):
        data_file = self.files.currentDataFile()
        if data_file is None:
            return

        key = self.outputs.combo_key.currentText()
        isotopes = self.toolbar.selectedIsotopes()

        view = self.graph.currentView()

        self.graph.clear()
        if view == "particle":
            self.graph.drawResultsParticle(
                self.processing_results[data_file], isotopes, key
            )
        elif view == "histogram":
            self.graph.drawResultsHistogram(
                self.processing_results[data_file], isotopes, key
            )
        elif view == "composition":
            clusters = self.clusters(data_file, key)
            self.graph.drawResultsComposition(
                self.processing_results[data_file], key, clusters
            )

    def onClusterDistanceChanged(self, distance: float):
        method = self.processing_methods["default"]
        if not np.isclose(method.cluster_distance, distance):
            method.cluster_distance = distance
            self.reprocess(self.files.currentDataFile())

    def onInstrumentOptionsChanged(self):
        method = self.processing_methods["default"]
        method.instrument_options = self.instrument_options.asInstrumentOptions()
        method.calibration_mode = (
            self.instrument_options.calibration_mode.currentText().lower()
        )
        # todo: update not reprocess
        self.reprocess(self.files.currentDataFile())

    def onLimitOptionsChanged(self):
        method = self.processing_methods["default"]
        method.limit_options = self.limit_options.asLimitOptions()
        method.accumulation_method = self.limit_options.limit_accumulation
        method.points_required = self.limit_options.points_required
        method.prominence_required = self.limit_options.prominence_required
        self.reprocess(self.files.currentDataFile())

    def onIsotopeOptionChanged(self, isotope: SPCalIsotope):
        option = self.isotope_options.optionForIsotope(isotope)
        self.processing_methods["default"].isotope_options[isotope] = option
        # todo: update not reprocess
        self.reprocess(self.files.currentDataFile())

    def onResultsChanged(self, data_file: SPCalDataFile):
        if data_file == self.files.currentDataFile():
            self.outputs.setResults(self.processing_results[data_file])
            self.redraw()

    def reprocess(self, data_file: SPCalDataFile | None):
        if not isinstance(data_file, SPCalDataFile):
            files = self.files.dataFiles()
        else:
            files = [data_file]

        for file in files:
            self.processing_results[file] = self.processing_methods[
                "default"
            ].processDataFile(file)
            # refresh clusters
            if file in self.processing_clusters:
                self.processing_clusters[file].clear()
            self.resultsChanged.emit(file)

    def dialogPeakProperties(self) -> QtWidgets.QDialog | None:
        from spcal.gui.dialogs.peakproperties import PeakPropertiesDialog

        data_file = self.files.currentDataFile()

        dlg = PeakPropertiesDialog(
            self.processing_results[data_file],
            self.toolbar.combo_isotope.currentIsotope(),
        )
        dlg.exec()

    def createMenuBar(self):
        # File
        self.action_open_sample = create_action(
            "document-open",
            "&Open Sample File",
            "Import SP data from a CSV file.",
            lambda: self.dialogLoadFile(),
        )
        self.action_open_recent = QtGui.QActionGroup(self)
        self.action_open_recent.triggered.connect(
            lambda a: self.dialogLoadFile(a.text().replace("&", ""))
        )
        # self.action_open_reference = create_action(
        #     "document-open",
        #     "Open &Reference File",
        #     "Import reference SP data from a CSV file.",
        #     lambda: self.reference.dialogLoadFile(),
        # )

        self.action_open_batch = create_action(
            "document-multiple",
            "&Batch Processing",
            "Process multiple files using the current sample and reference settings.",
            self.dialogBatchProcess,
        )
        self.action_open_batch.setEnabled(False)

        self.action_save_session = create_action(
            "cloud-upload",
            "Save Session",
            "Save the current options, inputs and data to a session file.",
            self.dialogSaveSession,
        )

        self.action_load_session = create_action(
            "cloud-download",
            "Restore Session",
            "Restore the saved session.",
            self.dialogLoadSession,
        )

        # Todo move to right click menu
        self.action_export = create_action(
            "document-save-as",
            "E&xport Data",
            "Save single particle signal data to various formats.",
            self.dialogExportData,
        )
        self.action_export.setEnabled(False)

        self.action_close = create_action(
            "window-close", "Quit", "Exit SPCal.", self.close
        )

        # Edit
        self.action_clear = create_action(
            "edit-reset",
            "Clear Data and Inputs",
            "Clears loaded datas and resets the option, sample and reference inputs.",
            self.resetInputs,
        )
        self.action_calculator = create_action(
            "insert-math-expression",
            "Signal Calculator",
            "Perform arbitrary calculations on signal data.",
            self.dialogCalculator,
        )

        # Tools
        self.action_mass_fraction_calculator = create_action(
            "folder-calculate",
            "Mass Fraction Calculator",
            "Calculate the mass fraction of elements in a compound.",
            self.dialogMassFractionCalculator,
        )
        self.action_particle_database = create_action(
            "folder-database",
            "Density Database",
            "Search for compound densities.",
            self.dialogParticleDatabase,
        )
        self.action_ionic_response_tool = create_action(
            "document-open",
            "Ionic Response Calculator",
            "Read ionic responses from a standard file and apply to "
            "sample and reference.",
            self.dialogIonicResponse,
        )

        # View
        current_scheme = QtCore.QSettings().value("colorscheme", "IBM Carbon")
        self.action_color_scheme = QtGui.QActionGroup(self)
        for scheme in color_schemes.keys():
            action = self.action_color_scheme.addAction(scheme)
            action.setCheckable(True)
            if scheme == current_scheme:
                action.setChecked(True)
        self.action_color_scheme.triggered.connect(self.setColorScheme)

        self.action_display_sigfigs = create_action(
            "format-precision-more",
            "Output Sig. Figures",
            "Set the number of significant figures shown for outputs.",
            self.setSignificantFigures,
        )

        self.action_font = create_action(
            "font",
            "Graph Font",
            "Choose the font to use for text in graphs.",
            self.setGraphFont,
        )

        # Help
        self.action_log = create_action(
            "dialog-information",
            "Show &Log",
            "Show the error and information log.",
            self.log.open,
        )
        self.action_documentation = create_action(
            "documentation",
            "Online Documentation",
            "Opens a link to documenation on usage.",
            self.linkToDocumenation,
        )
        self.action_about = create_action(
            "help-about", "About", "About SPCal.", self.about
        )

        menufile = self.menuBar().addMenu("&File")
        menufile.addAction(self.action_open_sample)

        self.menu_recent = menufile.addMenu("Open Recent")
        self.menu_recent.setIcon(QtGui.QIcon.fromTheme("document-open-recent"))
        self.menu_recent.setEnabled(False)

        # menufile.addSeparator()
        # menufile.addAction(self.action_open_reference)
        menufile.addSeparator()
        menufile.addAction(self.action_open_batch)
        menufile.addSeparator()
        # menufile.addAction(self.action_export)
        menufile.addAction(self.action_save_session)
        menufile.addAction(self.action_load_session)
        menufile.addSeparator()
        menufile.addAction(self.action_close)

        menuedit = self.menuBar().addMenu("&Edit")
        menuedit.addAction(self.action_clear)
        menuedit.addSeparator()
        menuedit.addAction(self.action_calculator)
        menuedit.addAction(self.action_ionic_response_tool)
        menuedit.addSeparator()
        menuedit.addAction(self.action_mass_fraction_calculator)
        menuedit.addAction(self.action_particle_database)

        menuview = self.menuBar().addMenu("&View")

        menu_cs = menuview.addMenu("Color Scheme")
        menu_cs.setIcon(QtGui.QIcon.fromTheme("color-management"))
        menu_cs.setStatusTip("Change the colorscheme of the input and result graphs.")
        menu_cs.setToolTip("Change the colorscheme of the input and result graphs.")
        menu_cs.addActions(self.action_color_scheme.actions())

        menuview.addAction(self.action_font)
        menuview.addAction(self.action_display_sigfigs)

        menuhelp = self.menuBar().addMenu("&Help")
        menuhelp.addAction(self.action_log)
        menuhelp.addAction(self.action_documentation)
        menuhelp.addAction(self.action_about)

    def about(self) -> QtWidgets.QDialog:
        dlg = QtWidgets.QMessageBox(
            QtWidgets.QMessageBox.Icon.Information,
            "About SPCal",
            (
                "sp/scICP-MS processing.\n"
                f"Version {QtWidgets.QApplication.applicationVersion()}\n"
                "https://github.com/djdt/spcal"
            ),
            parent=self,
        )
        if self.windowIcon() is not None:
            dlg.setIconPixmap(self.windowIcon().pixmap(64, 64))
        dlg.open()
        return dlg

    def dialogLoadFile(self, path: Path | None = None) -> ImportDialogBase | None:
        if path is None:
            path = get_open_spcal_path(self)
            if path is None:
                return None
        else:
            path = Path(path)

        dlg = get_import_dialog_for_path(
            self,
            path,
            self.files.currentDataFile(),
            screening_method=self.processing_methods["default"],
        )
        dlg.dataImported.connect(self.files.addDataFile)
        # the importer can take up a lot of memory so delete it
        dlg.finished.connect(dlg.deleteLater)
        dlg.exec()
        return dlg

    def dialogBatchProcess(self) -> BatchProcessDialog:
        raise NotImplementedError

    #     self.results.updateResults()  # Force an update
    #     dlg = BatchProcessDialog(
    #         [],
    #         self.sample,
    #         self.reference,
    #         self.options,
    #         self.results,
    #         parent=self,
    #     )
    #     dlg.open()
    #     return dlg
    #
    def dialogCalculator(self) -> CalculatorDialog:
        raise NotImplementedError

    #     dlg = CalculatorDialog(self.sample.names, self.sample.current_expr, parent=self)
    #     dlg.expressionAdded.connect(self.sample.addExpression)
    #     dlg.expressionRemoved.connect(self.sample.removeExpression)
    #     dlg.expressionAdded.connect(self.reference.addExpression)
    #     dlg.expressionRemoved.connect(self.reference.removeExpression)
    #     dlg.open()
    #     return dlg

    def dialogExportData(self) -> None:
        raise NotImplementedError

    #     path, filter = QtWidgets.QFileDialog.getSaveFileName(
    #         self,
    #         "Save Data As",
    #         "",
    #         "Numpy Archive (*.npz);;CSV Document(*.csv);;All files(*)",
    #     )
    #     if path == "":
    #         return
    #
    #     filter_suffix = filter[filter.rfind(".") : -1]
    #
    #     path = Path(path)
    #     if filter_suffix != "":  # append suffix if missing
    #         path = path.with_suffix(filter_suffix)
    #
    #     names = self.sample.responses.dtype.names
    #     if path.suffix.lower() == ".csv":
    #         trim = self.sample.trimRegion(names[0])
    #         header = " ".join(name for name in names)
    #         np.savetxt(
    #             path,
    #             self.sample.responses[trim[0] : trim[1]],
    #             delimiter="\t",
    #             comments="",
    #             header=header,
    #             fmt="%.16g",
    #         )
    #     elif path.suffix.lower() == ".npz":
    #         np.savez_compressed(
    #             path,
    #             **{name: self.sample.trimmedResponse(name) for name in names},
    #         )
    #     else:
    #         raise ValueError("dialogExportData: file suffix must be '.npz' or '.csv'.")

    def dialogFilterDetections(self) -> None:
        data_file = self.files.currentDataFile()
        if data_file is None:
            return
        dlg = FilterDialog(
            list(self.processing_results[data_file].keys()),
            self.processing_methods["default"].filters,
            [],
            number_clusters=99,
            parent=self,
        )
        dlg.filtersChanged.connect(self.processing_methods["default"].setFilters)
        dlg.filtersChanged.connect(self.reprocess)
        dlg.open()

    def dialogMassFractionCalculator(self) -> MassFractionCalculatorDialog:
        dlg = MassFractionCalculatorDialog(parent=self)
        dlg.open()
        return dlg

    def dialogParticleDatabase(self) -> ParticleDatabaseDialog:
        dlg = ParticleDatabaseDialog(parent=self)
        dlg.open()
        return dlg

    def dialogIonicResponse(self) -> ResponseDialog:
        raise NotImplementedError
        dlg = ResponseDialog(parent=self)
        dlg.responsesSelected.connect(self.sample.io.setResponses)
        dlg.responsesSelected.connect(self.reference.io.setResponses)
        dlg.open()
        return dlg

    def dialogSaveSession(self) -> None:
        raise NotImplementedError

        def onFileSelected(file: str) -> None:
            path = Path(file).with_suffix(".spcal")
            saveSession(path, self.options, self.sample, self.reference, self.results)
            QtCore.QSettings().setValue("LastSession", str(path))

        dir = Path(QtCore.QSettings().value("LastSession", ""))
        if dir.is_file():
            dir = dir.parent

        # Dont use getSaveFileName as it doesn't have setDefaultSuffix
        dlg = QtWidgets.QFileDialog(
            self, "Save Session", str(dir), "SPCal Sessions(*.spcal);;All Files(*)"
        )
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dlg.setDefaultSuffix(".spcal")

        dlg.fileSelected.connect(onFileSelected)
        dlg.exec()

    def dialogLoadSession(self) -> None:
        raise NotImplementedError
        settings = QtCore.QSettings()
        dir = Path(settings.value("LastSession", ""))
        if dir.is_file():
            dir = dir.parent
        file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Session", str(dir), "SPCal Sessions(*.spcal);;All Files(*)"
        )
        if file == "":
            return
        restoreSession(
            Path(file), self.options, self.sample, self.reference, self.results
        )

    def linkToDocumenation(self) -> None:
        QtGui.QDesktopServices.openUrl("https://spcal.readthedocs.io")

    # def onInputsChanged(self) -> None:
    #     # Reference tab is neb method requires
    #     self.tabs.setTabEnabled(
    #         self.tabs.indexOf(self.reference),
    #         self.options.efficiency_method.currentText()
    #         in ["Reference Particle", "Mass Response"],
    #     )
    #     self.tabs.setTabEnabled(
    #         self.tabs.indexOf(self.results),
    #         self.readyForResults(),
    #     )
    #     self.action_open_batch.setEnabled(self.readyForResults())
    #     self.action_export.setEnabled(self.readyForResults())

    # def onTabChanged(self, index: int) -> None:
    #     if index == self.tabs.indexOf(self.results):
    #         if self.results.isUpdateRequired():
    #             self.results.updateResults()

    # def readyForResults(self) -> bool:
    #     return self.sample.isComplete() and (
    #         self.options.efficiency_method.currentText() in ["Manual Input"]
    #         or self.reference.isComplete()
    #     )
    #
    def resetInputs(self) -> None:
        self.instrument_options.resetInputs()
        self.limit_options.resetInputs()
        self.isotope_options.resetInputs()
        # self.tabs.setCurrentIndex(0)
        #
        # self.results.resetInputs()
        # self.sample.resetInputs()
        # self.reference.resetInputs()
        # self.options.resetInputs()

    def setColorScheme(self, action: QtGui.QAction) -> None:
        scheme = action.text().replace("&", "")
        QtCore.QSettings().setValue("colorscheme", scheme)

        self.redraw()

    def setGraphFont(self) -> None:
        settings = QtCore.QSettings()
        current = QtGui.QFont(
            settings.value("GraphFont/Family", "SansSerif"),
            pointSize=int(settings.value("GraphFont/PointSize", 10)),
        )

        ok, font = QtWidgets.QFontDialog.getFont(current, parent=self)
        if ok:
            settings.setValue("GraphFont/Family", font.family())
            settings.setValue("GraphFont/PointSize", font.pointSize())
            self.sample.setGraphFont(font)
            self.reference.setGraphFont(font)
            self.results.setGraphFont(font)

    def setSignificantFigures(self) -> None:
        settings = QtCore.QSettings()
        current = int(settings.value("SigFigs", 4))

        value, ok = QtWidgets.QInputDialog.getInt(
            self, "Set Output Sig. Fig.", "Significant Figures", current, 1, 11, 1
        )
        if ok:
            settings.setValue("SigFigs", value)
            self.instrument_options.setSignificantFigures(value)
            self.limit_options.setSignificantFigures(value)
            self.isotope_options.setSignificantFigures(value)
            self.sample.io.setSignificantFigures(value)
            self.reference.io.setSignificantFigures(value)
            self.results.io.setSignificantFigures(value)

    def syncSampleAndReference(self) -> None:
        # Sync response
        for name, io in zip(self.sample.io.names(), self.sample.io.widgets()):
            if name in self.reference.io:
                ref_io = self.reference.io[name]
                sample_value = io.response.baseValue()
                ref_value = ref_io.response.baseValue()
                if sample_value is not None and ref_value is None:
                    ref_io.response.setBaseValue(sample_value)
                elif sample_value is None and ref_value is not None:
                    io.response.setBaseValue(ref_value)

                io.syncOutput(ref_io, "response")

    def updateRecentFiles(self, new_path: SPCalDataFile | Path | None = None) -> None:
        MAX_RECENT_FILES = 10
        settings = QtCore.QSettings()

        if isinstance(new_path, SPCalDataFile):
            new_path = new_path.path

        paths = []
        num = settings.beginReadArray("RecentFiles")
        for i in range(num):
            settings.setArrayIndex(i)
            path = Path(settings.value("Path"))
            if path != new_path:
                paths.append(path)
        settings.endArray()

        if new_path is not None:
            paths.insert(0, new_path)
            paths = paths[:MAX_RECENT_FILES]

            settings.remove("RecentFiles")
            settings.beginWriteArray("RecentFiles", len(paths))
            for i, path in enumerate(paths):
                settings.setArrayIndex(i)
                settings.setValue("Path", str(path))
            settings.endArray()

        # Clear old actions
        self.menu_recent.clear()
        for action in self.action_open_recent.actions():
            self.action_open_recent.removeAction(action)

        # Add new
        self.menu_recent.setEnabled(len(paths) > 0)
        for path in paths:
            action = QtGui.QAction(str(path), self)
            self.action_open_recent.addAction(action)
            self.menu_recent.addAction(action)

    def exceptHook(
        self, etype: type, value: BaseException, tb: TracebackType | None = None
    ):  # pragma: no cover
        """Redirect errors to the log."""
        if etype is KeyboardInterrupt:
            logger.info("Keyboard interrupt, exiting.")
            sys.exit(1)
        logger.exception("Uncaught exception", exc_info=(etype, value, tb))
        QtWidgets.QMessageBox.critical(self, "Uncaught Exception", str(value))
