import json
import logging
import sys
from pathlib import Path
from types import TracebackType

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.datafile import SPCalDataFile

from spcal.gui.batch.wizard import SPCalBatchProcessingWizard
from spcal.gui.dialogs.calculator import CalculatorDialog
from spcal.gui.dialogs.export import ExportDialog
from spcal.gui.dialogs.filter import FilterDialog
from spcal.gui.dialogs.io import ImportDialogBase
from spcal.gui.dialogs.response import ResponseDialog
from spcal.gui.dialogs.tools import (
    MassFractionCalculatorDialog,
    ParticleDatabaseDialog,
    TransportEfficiencyDialog,
)
from spcal.gui.docks.central import SPCalCentralWidget
from spcal.gui.docks.datafile import SPCalDataFilesDock
from spcal.gui.docks.instrumentoptions import SPCalInstrumentOptionsDock
from spcal.gui.docks.isotopeoptions import SPCalIsotopeOptionsDock
from spcal.gui.docks.limitoptions import SPCalLimitOptionsDock
from spcal.gui.docks.outputs import SPCalOutputsDock
from spcal.gui.docks.toolbar import SPCalOptionsToolBar, SPCalViewToolBar
from spcal.gui.graphs import color_schemes
from spcal.gui.io import (
    NU_FILE_FILTER,
    TEXT_FILE_FILTER,
    TOFWERK_FILE_FILTER,
    get_import_dialog_for_path,
    get_open_spcal_path,
    is_spcal_path,
)
from spcal.gui.log import LoggingDialog
from spcal.gui.util import create_action
from spcal.io.session import (
    save_session_json,
    decode_json_method,
    decode_json_datafile,
)
from spcal.isotope import SPCalIsotope, SPCalIsotopeBase, SPCalIsotopeExpression
from spcal.processing import CALIBRATION_KEYS
from spcal.processing.options import SPCalIsotopeOptions
from spcal.processing.method import SPCalProcessingMethod
from spcal.processing.result import SPCalProcessingResult
from spcal.processing.filter import SPCalIndexFilter, SPCalResultFilter

logger = logging.getLogger(__name__)


class SPCalMainWindow(QtWidgets.QMainWindow):
    resultsChanged = QtCore.Signal(SPCalDataFile)
    currentMethodChanged = QtCore.Signal(SPCalProcessingMethod)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("SPCal")
        self.resize(1600, 900)

        self.setDockNestingEnabled(True)

        self.setAcceptDrops(True)

        self.log = LoggingDialog()
        self.log.setWindowTitle("SPCal Log")

        settings = QtCore.QSettings()
        method = SPCalProcessingMethod(
            accumulation_method=str(
                settings.value("Threshold/AccumulationMethod", "signal mean")
            ),
            points_required=int(settings.value("Threshold/PointsRequired", 1)),  # type: ignore
            prominence_required=float(
                settings.value("Threshold/ProminenceRequired", 0.2)  # type: ignore
            ),
        )
        self.processing_methods = {"default": method}

        self.graph = SPCalCentralWidget()

        self.toolbar = SPCalOptionsToolBar()
        self.toolbar_view = SPCalViewToolBar()

        self.instrument_options = SPCalInstrumentOptionsDock(
            method.instrument_options, method.calibration_mode, method.cluster_distance
        )

        self.limit_options = SPCalLimitOptionsDock(
            method.limit_options,
            method.accumulation_method,
            method.points_required,
            method.prominence_required,
        )
        self.isotope_options = SPCalIsotopeOptionsDock()

        self.files = SPCalDataFilesDock()
        self.files.screening_method = self.processing_methods["default"]
        self.outputs = SPCalOutputsDock()

        self.processing_results: dict[
            SPCalDataFile, dict[SPCalIsotopeBase, SPCalProcessingResult]
        ] = {}
        self.processing_clusters: dict[SPCalDataFile, dict[str, np.ndarray]] = {}

        self.graph.particle.requestPeakProperties.connect(self.dialogPeakProperties)
        self.graph.particle.exclusionRegionsChanged.connect(self.setExclusionRegions)
        self.graph.requestRedraw.connect(self.redraw)

        self.instrument_options.efficiencyDialogRequested.connect(
            self.dialogTransportEfficiencyCalculator
        )
        self.instrument_options.optionsChanged.connect(self.onInstrumentOptionsChanged)

        self.limit_options.optionsChanged.connect(self.onLimitOptionsChanged)

        self.isotope_options.optionChanged.connect(self.onIsotopeOptionChanged)
        self.isotope_options.requestCurrentIsotope.connect(
            self.toolbar.combo_isotope.setCurrentIsotope
        )

        self.resultsChanged.connect(self.onResultsChanged)

        self.currentMethodChanged.connect(self.files.setScreeningMethod)

        self.files.dataFileAdded.connect(self.onDataFileAdded)
        self.files.dataFileRemoved.connect(self.removeFileFromResults)
        self.files.dataFilesChanged.connect(self.updateForDataFiles)

        self.outputs.requestCurrentIsotope.connect(
            self.toolbar.combo_isotope.setCurrentIsotope
        )
        self.outputs.requestRemoveIsotopes.connect(self.removeIsotopes)
        self.outputs.requestAddExpression.connect(self.addExpression)
        self.outputs.requestRemoveExpressions.connect(self.removeExpressions)

        self.toolbar.isotopeChanged.connect(self.redraw)
        self.toolbar.scatterOptionsChanged.connect(self.redraw)
        self.toolbar.keyChanged.connect(self.onKeyChanged)
        self.toolbar.requestFilterDialog.connect(self.dialogFilterDetections)

        self.toolbar_view.viewChanged.connect(self.toolbar.onViewChanged)
        self.toolbar_view.viewChanged.connect(self.graph.setView)
        self.toolbar_view.requestViewOptionsDialog.connect(
            self.graph.dialogGraphOptions
        )

        self.setCentralWidget(self.graph)
        self.defaultLayout()

        self.createMenuBar()
        self.updateRecentFiles()

    # Access

    def clusters(self, data_file: SPCalDataFile, key: str) -> np.ndarray:
        if data_file not in self.processing_clusters:
            self.processing_clusters[data_file] = {}
        if key not in self.processing_clusters[data_file]:
            self.processing_clusters[data_file][key] = self.processing_methods[
                "default"
            ].processClusters(self.processing_results[data_file], key)
        return self.processing_clusters[data_file][key]

    @QtCore.Slot()
    def currentMethod(self) -> SPCalProcessingMethod:
        return self.processing_methods["default"]

    @QtCore.Slot()
    def setCurrentMethod(self, method: SPCalProcessingMethod):
        self.processing_methods["default"] = method

        self.instrument_options.optionsChanged.disconnect(
            self.onInstrumentOptionsChanged
        )
        self.instrument_options.setInstrumentOptions(
            method.instrument_options, method.calibration_mode
        )
        self.instrument_options.optionsChanged.connect(self.onInstrumentOptionsChanged)

        self.limit_options.optionsChanged.disconnect(self.onLimitOptionsChanged)
        self.limit_options.setLimitOptions(
            method.limit_options,
            method.accumulation_method,
            method.points_required,
            method.prominence_required,
        )
        self.limit_options.optionsChanged.connect(self.onLimitOptionsChanged)

        self.currentMethodChanged.emit(method)

    # Menu

    def createMenuBar(self):
        # File
        self.action_open = create_action(
            "document-open",
            "&Open Sample File",
            "Import SP data from a CSV file.",
            lambda: self.dialogLoadFile(),
        )
        self.action_open.setShortcut(QtGui.QKeySequence.StandardKey.Open)

        self.action_open_recent = QtGui.QActionGroup(self)
        self.action_open_recent.triggered.connect(self.openRecentFile)

        self.action_open_batch = create_action(
            "document-multiple",
            "&Batch Processing",
            "Process multiple files using the current sample and reference settings.",
            self.dialogBatchProcess,
        )

        self.action_save_session = create_action(
            "cloud-download",
            "Save Session",
            "Save the current options, inputs and data to a session file.",
            self.dialogSaveSession,
        )

        self.action_load_session = create_action(
            "cloud-upload",
            "Restore Session",
            "Restore the saved session.",
            self.dialogLoadSession,
        )

        # Todo move to right click menu
        self.action_export = create_action(
            "document-save-as",
            "E&xport Results",
            "Save single particle results as a CSV.",
            self.dialogExportResults,
        )
        self.action_export.setEnabled(False)
        self.action_export.setShortcut(QtGui.QKeySequence.StandardKey.Save)

        self.action_close = create_action(
            "window-close", "Quit", "Exit SPCal.", self.close
        )
        self.action_close.setShortcut(QtGui.QKeySequence.StandardKey.Quit)

        # Edit
        self.action_clear = create_action(
            "edit-reset",
            "Clear Data and Inputs",
            "Clears loaded datas and resets the option, sample and reference inputs.",
            self.reset,
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
            "gtk-select-font",
            "Graph Font",
            "Choose the font to use for text in graphs.",
            self.setGraphFont,
        )

        self.action_default_layout = create_action(
            "view-group",
            "Restore Default Layout",
            "Return dock widgets to default positions and sizes",
            self.defaultLayout,
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
        menufile.addAction(self.action_open)

        self.menu_recent = menufile.addMenu("Open Recent")
        self.menu_recent.setIcon(QtGui.QIcon.fromTheme("document-open-recent"))
        self.menu_recent.setEnabled(False)

        menufile.addSeparator()
        menufile.addAction(self.action_export)
        menufile.addAction(self.action_open_batch)
        menufile.addSeparator()
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
        menuview.addSeparator()

        menu_docks = menuview.addMenu("Show/hide dock widgets")
        for dock in self.findChildren(QtWidgets.QDockWidget):
            menu_docks.addAction(dock.toggleViewAction())

        menuview.addAction(self.action_default_layout)

        menuhelp = self.menuBar().addMenu("&Help")
        menuhelp.addAction(self.action_log)
        menuhelp.addAction(self.action_documentation)
        menuhelp.addAction(self.action_about)

    def colorForIsotope(
        self, isotope: SPCalIsotopeBase, data_file: SPCalDataFile
    ) -> QtGui.QColor:
        scheme = color_schemes[
            str(QtCore.QSettings().value("colorscheme", "IBM Carbon"))
        ]
        if isinstance(isotope, SPCalIsotope):
            idx = data_file.selected_isotopes.index(isotope)
        elif isinstance(isotope, SPCalIsotopeExpression):
            method = self.currentMethod()
            idx = method.expressions.index(isotope) + len(data_file.selected_isotopes)
        else:
            raise ValueError(f"unknown isotope type '{type(isotope)}'")

        data_files = self.files.selectedDataFiles()
        for i in range(0, data_files.index(data_file)):
            idx += len(data_files[i].selected_isotopes)
        return scheme[idx % len(scheme)]

    # Method modification

    def addExpression(self, expr: SPCalIsotopeExpression):
        method = self.currentMethod()
        if expr not in method.expressions:
            method.expressions.append(expr)
            self.currentMethodChanged.emit(method)
            self.updateForDataFiles(
                self.files.currentDataFile(), self.files.selectedDataFiles()
            )

    def removeExpressions(self, expressions: list[SPCalIsotopeExpression]):
        method = self.currentMethod()
        for expr in expressions:
            if expr in method.expressions:
                method.expressions.remove(expr)
            self.removeIsotopeFromResults(expr)
        self.currentMethodChanged.emit(method)
        self.updateForDataFiles(
            self.files.currentDataFile(), self.files.selectedDataFiles()
        )

    def setExclusionRegions(
        self, regions: list[tuple[float, float]], data_file: SPCalDataFile | None = None
    ):
        if data_file is None:
            data_file = self.files.currentDataFile()
        if data_file is None:
            raise ValueError("cannot set exclusion regions, invalid data file")

        method = self.currentMethod()
        method.exclusion_regions = regions

        self.currentMethodChanged.emit(method)
        self.reprocess([data_file])

    def setResponses(self, responses: dict[SPCalIsotopeBase, float]):
        method = self.currentMethod()
        for isotope, response in responses.items():
            if isotope in method.isotope_options:
                method.isotope_options[isotope].response = response
            else:
                method.isotope_options[isotope] = SPCalIsotopeOptions(
                    None, response, None
                )
            if isotope in self.isotope_options.isotopeOptions():
                self.isotope_options.setIsotopeOption(
                    isotope, method.isotope_options[isotope]
                )

    # def setClusterDistance(self, distance: float):
    #     method = self.currentMethod()
    #     if not np.isclose(method.cluster_distance, distance):
    #         method.cluster_distance = distance
    #
    #         self.currentMethodChanged.emit(method)
    #         self.reprocess(self.files.currentDataFile())

    # Slots for signals

    def openRecentFile(self, action: QtGui.QAction):
        path = Path(action.text().replace("&", ""))
        if not path.exists():
            QtWidgets.QMessageBox.warning(
                self, "File Not Found", f"The file '{path}' does not exist."
            )
            self.updateRecentFiles(remove_path=path)
        else:
            self.dialogLoadFile(path)

    def onDataFileAdded(self, data_file: SPCalDataFile):
        self.reprocess([data_file])
        self.updateRecentFiles(data_file)
        logger.info(
            f"DataFile '{data_file.path.stem}' imported with {data_file.num_events} events."
        )

    def updateForDataFiles(
        self, current: SPCalDataFile | None, selected: list[SPCalDataFile]
    ):
        if current is None:
            self.isotope_options.setIsotopes([])
            self.toolbar.setIsotopes([])
            self.outputs.setResults([])
            self.graph.clear()
            return

        # Set the options to current
        method: SPCalProcessingMethod = self.currentMethod()
        isotopes = current.selected_isotopes + method.validExpressions(current)

        # Add any missing isotopes to method and isotope options
        isotopes = sorted(
            isotopes,
            key=lambda iso: iso.isotope if isinstance(iso, SPCalIsotope) else 9999,
        )

        method_changed = False
        self.isotope_options.blockSignals(True)
        self.isotope_options.setIsotopes(isotopes)
        for isotope in current.selected_isotopes:
            if isotope not in method.isotope_options:
                method.isotope_options[isotope] = SPCalIsotopeOptions(None, None, None)
                method_changed = True
            self.isotope_options.setIsotopeOption(
                isotope, method.isotope_options[isotope]
            )
        self.isotope_options.blockSignals(False)
        if method_changed:
            self.currentMethodChanged.emit(method)

        # Reprocess if new isotopes exist
        if current not in self.processing_results or any(
            isotope not in self.processing_results[current] for isotope in isotopes
        ):
            self.reprocess([current])
        else:
            self.onResultsChanged(current)

        all_isotopes = set(isotopes)
        for file in selected:
            all_isotopes = all_isotopes.union(file.selected_isotopes)

        self.toolbar.setIsotopes(
            sorted(
                all_isotopes,
                key=lambda iso: iso.isotope if isinstance(iso, SPCalIsotope) else 9999,
            )
        )

        self.redraw()

    def onInstrumentOptionsChanged(self):
        method = self.currentMethod()
        method.instrument_options = self.instrument_options.instrumentOptions()
        method.calibration_mode = self.instrument_options.calibrationMode().lower()
        method.cluster_distance = self.instrument_options.clusterDistance()

        self.currentMethodChanged.emit(method)
        self.reprocess()

    def onIsotopeOptionChanged(self, isotope: SPCalIsotopeBase):
        data_file = self.files.currentDataFile()
        if data_file is None:
            return
        method = self.currentMethod()

        option = self.isotope_options.optionForIsotope(isotope)
        method.isotope_options[isotope] = option

        self.currentMethodChanged.emit(method)
        self.reprocess(isotopes=[isotope])

    def onKeyChanged(self, key: str):
        self.outputs.updateOutputsForKey(key)
        if self.graph.currentView() != "particle":
            self.redraw()

    def onLimitOptionsChanged(self):
        method = self.currentMethod()
        method.limit_options = self.limit_options.limitOptions()
        method.accumulation_method = self.limit_options.accumulationMethod()
        method.points_required = self.limit_options.pointsRequired()
        method.prominence_required = self.limit_options.prominenceRequired()

        self.currentMethodChanged.emit(method)
        self.reprocess()

    def onResultsChanged(self, data_file: SPCalDataFile):
        isotopes = data_file.selected_isotopes + self.currentMethod().expressions
        removed = [
            iso for iso in self.processing_results[data_file] if iso not in isotopes
        ]
        for isotope in removed:
            self.processing_results[data_file].pop(isotope)

        if data_file == self.files.currentDataFile():
            results = sorted(
                self.processing_results[data_file].values(),
                key=lambda result: (
                    int(result.isotope.isotope)
                    if isinstance(result.isotope, SPCalIsotope)
                    else 9999
                ),
            )
            self.outputs.setResults(results)
            self.redraw()

        self.action_export.setEnabled(len(self.processing_results) > 0)

    # data modification

    def removeFileFromResults(self, data_file: SPCalDataFile):
        self.processing_results.pop(data_file)

    def removeIsotopes(self, isotopes: list[SPCalIsotope]):
        for file in self.files.dataFiles():
            for isotope in isotopes:
                if isotope in file.selected_isotopes:
                    file.selected_isotopes.remove(isotope)
        self.reprocess()

    def removeIsotopeFromResults(self, isotope: SPCalIsotopeBase):
        for data_file, results in self.processing_results.items():
            if isotope in results:
                results.pop(isotope)

    def reprocess(
        self,
        data_files: list[SPCalDataFile] | None = None,
        isotopes: list[SPCalIsotopeBase] | None = None,
    ):
        if data_files is None:
            data_files = self.files.dataFiles()

        method = self.currentMethod()

        for file in data_files:
            if isotopes is None:
                isotopes = file.selected_isotopes  # type: ignore

            existing = self.processing_results.get(file, {})
            existing.update(method.processDataFile(file, isotopes))
            self.processing_results[file] = existing
            method.filterResults(self.processing_results[file])
            # refresh clusters
            if file in self.processing_clusters:
                self.processing_clusters[file].clear()

            if sum(len(filters) for filters in method.index_filters) > 0:
                keys = np.unique(
                    [f.key for filters in method.index_filters for f in filters]
                )
                clusters = {key: self.clusters(file, key) for key in keys}
                method.filterIndicies(self.processing_results[file], clusters)

            self.resultsChanged.emit(file)

    # drawing

    def redraw(self):
        key = self.toolbar.combo_key.currentText()
        view = self.graph.currentView()
        isotopes = self.toolbar.selectedIsotopes()

        self.graph.clear()

        if view == "composition":
            data_file = self.files.currentDataFile()
            if data_file is None:
                return
            clusters = self.clusters(data_file, key)
            colors = [
                self.colorForIsotope(isotope, data_file)
                for isotope in self.processing_results[data_file].keys()
            ]
            self.graph.drawResultsComposition(
                list(self.processing_results[data_file].values()), colors, key, clusters
            )
        elif view == "scatter":
            data_file = self.files.currentDataFile()
            if data_file is None:
                return

            self.graph.drawResultsScatterExpr(
                list(self.processing_results[data_file].values()),
                self.toolbar.scatter_x.text(),
                self.toolbar.scatter_y.text(),
                self.toolbar.scatter_key_x.currentText(),
                self.toolbar.scatter_key_y.currentText(),
            )

        elif view == "spectra":
            data_file = self.files.currentDataFile()
            isotope = self.toolbar.combo_isotope.currentIsotope()
            if data_file is None or self.processing_results[data_file][isotope] is None:
                return

            self.graph.drawResultsSpectra(
                data_file,
                self.processing_results[data_file][isotope],
                None,
            )
        else:
            files = self.files.selectedDataFiles()
            if len(files) == 0:
                current = self.files.currentDataFile()
                files = [current] if current is not None else []

            drawable = []
            names, colors = [], []
            for data_file in files:
                for isotope, result in self.processing_results[data_file].items():
                    if isotope in isotopes:
                        drawable.append(result)
                        colors.append(self.colorForIsotope(isotope, data_file))
                        name = (
                            f"{data_file.path.stem} - {isotope}"
                            if len(files) > 1
                            else str(isotope)
                        )
                        names.append(name)

            if view == "particle":
                self.graph.drawResultsParticle(drawable, colors, names, key)
                for start, end in self.currentMethod().exclusion_regions:
                    self.graph.particle.addExclusionRegion(start, end)
            elif view == "histogram":
                self.graph.drawResultsHistogram(drawable, colors, names, key)

    # Dialogs

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

    def dialogBatchProcess(self):  # BatchProcessDialog:
        df = self.files.currentDataFile()
        dlg = SPCalBatchProcessingWizard(
            df,
            self.currentMethod(),
            df.selected_isotopes if df is not None else [],
            parent=self,
        )
        dlg.exec()

    def dialogCalculator(self) -> CalculatorDialog:
        method = self.currentMethod()

        def set_expressions(expressions: list[SPCalIsotopeExpression]):
            method.expressions = expressions
            self.currentMethodChanged.emit(method)
            self.reprocess()
            self.updateForDataFiles(
                self.files.currentDataFile(), self.files.selectedDataFiles()
            )

        files = self.files.dataFiles()

        all_isotopes = set(files[0].isotopes)
        for file in files[1:]:
            all_isotopes.union(file.isotopes)
        all_isotopes = sorted(all_isotopes, key=lambda iso: iso.isotope)

        dlg = CalculatorDialog(all_isotopes, method.expressions, parent=self)
        dlg.expressionsChanged.connect(set_expressions)
        dlg.open()
        return dlg

    def dialogExportResults(self):
        # ensure clusters are generated
        for key in CALIBRATION_KEYS:
            for file in self.files.dataFiles():
                self.clusters(file, key)

        dlg = ExportDialog(
            self.files.dataFiles(),
            self.processing_results,
            self.processing_clusters,
            parent=self,
        )
        dlg.open()

    def dialogFilterDetections(self):
        data_file = self.files.currentDataFile()
        if data_file is None:
            return

        def set_filters(
            filters: list[list[SPCalResultFilter]],
            cluster_filters: list[list[SPCalIndexFilter]],
        ):
            method = self.currentMethod()
            method.setFilters(filters, cluster_filters)
            self.currentMethodChanged.emit(method)
            self.reprocess()

        method = self.currentMethod()
        dlg = FilterDialog(
            list(self.processing_results[data_file].keys()),
            method.result_filters,
            method.index_filters,
            number_clusters=99,
            parent=self,
        )
        dlg.filtersChanged.connect(set_filters)
        dlg.open()

    def dialogIonicResponse(self) -> ResponseDialog:
        dlg = ResponseDialog(parent=self)
        dlg.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        dlg.responsesSelected.connect(self.setResponses)
        dlg.open()
        return dlg

    def dialogLoadFile(self, path: Path | None = None) -> ImportDialogBase | None:
        if path is None:
            path = get_open_spcal_path(self)
            if path is None:
                return None
        else:
            path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Path '{path}' does not exist.")

        dlg = get_import_dialog_for_path(
            self,
            path,
            self.files.currentDataFile(),
            screening_method=self.currentMethod(),
        )
        dlg.dataImported.connect(self.files.addDataFile)
        # the importer can take up a lot of memory so delete it
        # dlg.finished.connect(dlg.deleteLater)
        dlg.exec()
        return dlg

    def dialogMassFractionCalculator(self) -> MassFractionCalculatorDialog:
        dlg = MassFractionCalculatorDialog(parent=self)
        dlg.open()
        return dlg

    def dialogParticleDatabase(self) -> ParticleDatabaseDialog:
        dlg = ParticleDatabaseDialog(parent=self)
        dlg.open()
        return dlg

    def dialogPeakProperties(self) -> QtWidgets.QDialog | None:
        from spcal.gui.dialogs.peakproperties import PeakPropertiesDialog

        data_file = self.files.currentDataFile()
        if data_file is None:
            return

        dlg = PeakPropertiesDialog(
            self.processing_results[data_file],
            self.toolbar.combo_isotope.currentIsotope(),
        )
        dlg.exec()

    def dialogTransportEfficiencyCalculator(self) -> TransportEfficiencyDialog | None:
        isotope = self.toolbar.combo_isotope.currentIsotope()
        data_file = self.files.currentDataFile()
        if data_file is None:
            return None

        dlg = TransportEfficiencyDialog(
            data_file, isotope, self.processing_results[data_file][isotope], parent=self
        )
        dlg.efficiencySelected.connect(
            self.instrument_options.options_widget.efficiency.setValue
        )
        dlg.isotopeOptionsChanged.connect(self.isotope_options.setIsotopeOption)
        dlg.isotopeOptionsChanged.connect(self.onIsotopeOptionChanged)
        dlg.open()
        return dlg

    def dialogSaveSession(self):
        def onFileSelected(file: str):
            path = Path(file)
            if path.suffixes != [".spcal", ".json"]:
                path = Path(path.stem).with_suffix(".spcal.json")
            save_session_json(path, self.currentMethod(), self.files.dataFiles())

        df = self.files.currentDataFile()
        if df is None:
            path = Path()
        else:
            path = df.path.parent

        # Dont use getSaveFileName as it doesn't have setDefaultSuffix
        dlg = QtWidgets.QFileDialog(
            self,
            "Save Session",
            str(path),
            "SPCal Sessions(*.spcal.json);;All Files(*)",
        )
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dlg.setDefaultSuffix(".spcal.json")

        dlg.fileSelected.connect(onFileSelected)
        dlg.exec()

    def dialogLoadSession(self):
        df = self.files.currentDataFile()
        if df is None:
            path = Path()
        else:
            path = df.path.parent

        file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Session",
            str(path),
            "SPCal Sessions(*.spcal.json);;All Files(*)",
        )
        if file == "":
            return
        with Path(file).open() as fp:
            session = json.load(fp)

        self.setCurrentMethod(decode_json_method(session["method"]))

        check_update_paths = True
        for datafile in session["datafiles"]:
            path = Path(datafile["path"])
            if not path.exists():
                if not check_update_paths:
                    continue
                button = QtWidgets.QMessageBox.warning(
                    self,
                    "Datafile Not Found",
                    f"Cannot find datafile at '{path}'. Select a new path?",
                    buttons=QtWidgets.QMessageBox.StandardButton.Yes
                    | QtWidgets.QMessageBox.StandardButton.No
                    | QtWidgets.QMessageBox.StandardButton.NoToAll,
                )
                if button == QtWidgets.QMessageBox.StandardButton.No:
                    continue
                elif button == QtWidgets.QMessageBox.StandardButton.NoToAll:
                    check_update_paths = False
                    continue
                # button == Ok
                if datafile["format"] == "text":
                    filter = TEXT_FILE_FILTER
                elif datafile["format"] == "nu":
                    filter = NU_FILE_FILTER
                elif datafile["format"] == "tofwerk":
                    filter = TOFWERK_FILE_FILTER
                else:
                    raise ValueError(f"unknown datafile format '{datafile['format']}")
                file, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self, "Select New Path", str(path.parent), filter
                )
                if file == "":
                    continue
                path = Path(file)
            self.files.addDataFile(decode_json_datafile(datafile, path))

    def linkToDocumenation(self):
        QtGui.QDesktopServices.openUrl("https://spcal.readthedocs.io")

    # UI

    def defaultLayout(self):
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, self.toolbar)
        self.addToolBar(QtCore.Qt.ToolBarArea.RightToolBarArea, self.toolbar_view)

        self.addDockWidget(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.instrument_options
        )
        self.addDockWidget(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.limit_options
        )
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.files)
        self.addDockWidget(
            QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.isotope_options
        )

        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.outputs)

        for dock in self.findChildren(QtWidgets.QDockWidget):
            dock.show()

        size = self.size()
        left_width = max(
            self.instrument_options.sizeHint().width(),
            self.limit_options.sizeHint().width(),
        )
        self.resizeDocks(
            [self.instrument_options, self.limit_options],
            [left_width, left_width],
            QtCore.Qt.Orientation.Horizontal,
        )
        self.resizeDocks(
            [self.instrument_options, self.limit_options, self.files],
            [size.height() // 3, size.height() // 3, size.height() // 3],
            QtCore.Qt.Orientation.Vertical,
        )
        isotope_width = self.isotope_options.sizeHint().width() + 48
        self.resizeDocks(
            [self.files, self.isotope_options, self.outputs],
            [left_width, isotope_width, size.width() - left_width - isotope_width],
            QtCore.Qt.Orientation.Horizontal,
        )

    def reset(self):
        self.files.reset()
        self.graph.clear()
        self.outputs.reset()
        self.instrument_options.reset()
        self.limit_options.reset()
        self.isotope_options.reset()
        self.toolbar.reset()

    def setColorScheme(self, action: QtGui.QAction):
        scheme = action.text().replace("&", "")
        QtCore.QSettings().setValue("colorscheme", scheme)

        self.redraw()

    def setGraphFont(self):
        settings = QtCore.QSettings()
        current = QtGui.QFont(
            settings.value("GraphFont/Family", "SansSerif"),  # type: ignore
            pointSize=int(settings.value("GraphFont/PointSize", 10)),  # type: ignore
        )

        ok, font = QtWidgets.QFontDialog.getFont(current, parent=self)
        if ok:
            settings.setValue("GraphFont/Family", font.family())
            settings.setValue("GraphFont/PointSize", font.pointSize())
            self.graph.setGraphFont(font)

    def setSignificantFigures(self):
        settings = QtCore.QSettings()
        current = int(settings.value("SigFigs", 4))  # type: ignore

        value, ok = QtWidgets.QInputDialog.getInt(
            self, "Set Output Sig. Fig.", "Significant Figures", current, 1, 11, 1
        )
        if ok:
            settings.setValue("SigFigs", value)
            self.instrument_options.setSignificantFigures(value)
            self.limit_options.setSignificantFigures(value)
            self.isotope_options.setSignificantFigures(value)
            self.outputs.setSignificantFigures(value)

    def updateRecentFiles(
        self,
        add_path: SPCalDataFile | Path | None = None,
        remove_path: Path | None = None,
    ):
        MAX_RECENT_FILES = 10
        settings = QtCore.QSettings()

        if isinstance(add_path, SPCalDataFile):
            add_path = add_path.path

        paths = []
        num = settings.beginReadArray("RecentFiles")
        for i in range(num):
            settings.setArrayIndex(i)
            path = Path(settings.value("Path"))
            if path != add_path:
                paths.append(path)
        settings.endArray()

        if add_path is not None:
            paths.insert(0, add_path)

        if remove_path is not None and remove_path in paths:
            paths.remove(remove_path)

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

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        for url in event.mimeData().urls():
            if is_spcal_path(url.toLocalFile()):
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent):
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if is_spcal_path(path):
                self.dialogLoadFile(path)
                event.accept()

    def exceptHook(
        self, etype: type, value: BaseException, tb: TracebackType | None = None
    ):  # pragma: no cover
        """Redirect errors to the log."""
        if etype is KeyboardInterrupt:
            logger.info("Keyboard interrupt, exiting.")
            sys.exit(1)
        logger.exception("Uncaught exception", exc_info=(etype, value, tb))
        QtWidgets.QMessageBox.critical(self, "Uncaught Exception", str(value))
