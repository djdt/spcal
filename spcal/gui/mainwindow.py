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
from spcal.gui.dialogs.color import ColorDialog
from spcal.gui.dialogs.export import ExportDialog
from spcal.gui.dialogs.filter import FilterDialog
from spcal.gui.dialogs.io.base import ImportDialogBase
from spcal.gui.dialogs.manuallimits import ManualLimitDialog
from spcal.gui.dialogs.missingpaths import MissingPathsDialog
from spcal.gui.dialogs.response import ResponseDialog
from spcal.gui.dialogs.peakproperties import PeakPropertiesDialog
from spcal.gui.dialogs.processingoptions import ProcessingOptionsDialog

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
from spcal.gui.graphs.colors import COLOR_SCHEMES, scheme_icon
from spcal.gui.io import (
    get_import_dialog_for_path,
    get_open_spcal_path,
    is_spcal_path,
    SessionImportWorker,
    is_spcal_session_path,
)
from spcal.gui.log import LoggingDialog
from spcal.gui.util import create_action
from spcal.io.session import save_session_json, decode_json_method
from spcal.isotope import SPCalIsotope, SPCalIsotopeBase, SPCalIsotopeExpression
from spcal.processing import CALIBRATION_KEYS
from spcal.processing.options import SPCalIsotopeOptions, SPCalProcessingOptions
from spcal.processing.method import SPCalProcessingMethod
from spcal.processing.result import SPCalProcessingResult
from spcal.processing.filter import SPCalIndexFilter, SPCalResultFilter

logger = logging.getLogger(__name__)


class SPCalMainWindow(QtWidgets.QMainWindow):
    currentMethodChanged = QtCore.Signal(SPCalProcessingMethod)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("SPCal")
        self.resize(1600, 900)

        self.setDockNestingEnabled(True)

        self.setAcceptDrops(True)

        self.log = LoggingDialog()
        self.log.setWindowTitle("SPCal Log")

        self.session_thread = QtCore.QThread()
        self.session_worker: SessionImportWorker | None = None

        method = self.defaultMethod()

        self.processing_methods = {"default": method}

        self.graph = SPCalCentralWidget()

        self.toolbar = SPCalOptionsToolBar()
        self.toolbar_view = SPCalViewToolBar()

        self.instrument_options = SPCalInstrumentOptionsDock(method.instrument_options)

        self.limit_options = SPCalLimitOptionsDock(method.limit_options)
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
        self.graph.particle.globalExclusionRegionsChanged.connect(
            self.setGlobalExclusionRegions
        )
        self.graph.requestRedraw.connect(self.redraw)

        self.instrument_options.efficiencyDialogRequested.connect(
            self.dialogTransportEfficiencyCalculator
        )
        self.instrument_options.optionsChanged.connect(self.onInstrumentOptionsChanged)

        self.limit_options.optionsChanged.connect(self.onLimitOptionsChanged)
        self.limit_options.options_widget.requestManualLimitsDialog.connect(
            self.dialogManualLimits
        )

        self.isotope_options.optionChanged.connect(self.onIsotopeOptionChanged)
        self.isotope_options.requestCurrentIsotope.connect(
            self.toolbar.combo_isotope.setCurrentIsotope
        )

        self.currentMethodChanged.connect(self.files.setScreeningMethod)

        self.files.dataFileAdded.connect(self.onDataFileAdded)
        self.files.dataFileAdded.connect(self.updateRecentFiles)
        self.files.dataFileRemoved.connect(self.removeFileFromResults)
        self.files.dataFilesChanged.connect(self.updateForDataFiles)
        self.files.activeDataFilesChanged.connect(self.updateForDataFiles)

        self.outputs.activeResultsChanged.connect(self.redraw)

        self.outputs.requestRemoveIsotopes.connect(self.removeIsotopes)
        self.outputs.requestAddExpression.connect(self.addExpression)
        self.outputs.requestRemoveExpressions.connect(self.removeExpressions)

        self.toolbar.scatterOptionsChanged.connect(self.redraw)
        self.toolbar.keyChanged.connect(self.onKeyChanged)
        self.toolbar.requestFilterDialog.connect(self.dialogFilterDetections)

        self.toolbar_view.viewChanged.connect(self.toolbar.onViewChanged)
        self.toolbar_view.viewChanged.connect(self.graph.setView)
        self.toolbar_view.requestViewOptionsDialog.connect(
            self.graph.dialogGraphOptions
        )

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

        self.setCentralWidget(self.graph)
        self.restoreLayout()

        self.createMenuBar()
        self.updateRecentFiles()

    def defaultMethod(self) -> SPCalProcessingMethod:
        settings = QtCore.QSettings()
        method = SPCalProcessingMethod()
        if settings.contains("DefaultMethod/Instrument/Uptake"):
            method.instrument_options.uptake = float(
                settings.value("DefaultMethod/Instrument/Uptake")
            )
        if settings.contains("DefaultMethod/Instrument/Efficiency"):
            method.instrument_options.efficiency = float(
                settings.value("DefaultMethod/Instrument/Efficiency")
            )

        if settings.contains("DefaultMethod/Limits/Gaussian/Alpha"):
            method.limit_options.gaussian_kws["alpha"] = float(
                settings.value(
                    "DefaultMethod/Limits/Gaussian/Alpha",
                )
            )
        if settings.contains("DefaultMethod/Limits/Poisson/Alpha"):
            method.limit_options.poisson_kws["alpha"] = float(
                settings.value(
                    "DefaultMethod/Limits/Poisson/Alpha",
                )
            )
        if settings.contains("DefaultMethod/Limits/CompoundPoisson/Alpha"):
            method.limit_options.compound_poisson_kws["alpha"] = float(
                settings.value(
                    "DefaultMethod/Limits/CompoundPoisson/Alpha",
                )
            )
        if settings.contains("DefaultMethod/Limits/CompoundPoisson/Sigma"):
            method.limit_options.compound_poisson_kws["sigma"] = float(
                settings.value(
                    "DefaultMethod/Limits/CompoundPoisson/Sigma",
                )
            )
        if settings.contains("DefaultMethod/Limits/MaxIterations"):
            method.limit_options.max_iterations = int(
                settings.value("DefaultMethod/Limits/MaxIterations")
            )
        if settings.contains("DefaultMethod/Limits/WindowSize"):
            method.limit_options.window_size = int(
                settings.value("DefaultMethod/Limits/WindowSize")
            )

        if settings.contains("DefaultMethod/Processing/CalibrationMode"):
            method.processing_options.calibration_mode = settings.value(
                "DefaultMethod/Processing/CalibrationMode",
            )
        if settings.contains("DefaultMethod/Processing/AccumulationMethod"):
            method.processing_options.accumulation_method = settings.value(
                "DefaultMethod/Processing/AccumulationMethod",
            )
        if settings.contains("DefaultMethod/Processing/PointsRequired"):
            method.processing_options.points_required = int(
                settings.value(
                    "DefaultMethod/Processing/PointsRequired",
                )
            )
        if settings.contains("DefaultMethod/Processing/ProminenceRequired"):
            method.processing_options.prominence_required = float(
                settings.value(
                    "DefaultMethod/Processing/ProminenceRequired",
                )
            )
        if settings.contains("DefaultMethod/Processing/ClusterDistance"):
            method.processing_options.cluster_distance = float(
                settings.value(
                    "DefaultMethod/Processing/ClusterDistance",
                )
            )

        return method

    def saveDefaultMethod(self):
        settings = QtCore.QSettings()
        method = self.currentMethod()

        if method.instrument_options.uptake is not None:
            settings.setValue(
                "DefaultMethod/Instrument/Uptake", method.instrument_options.uptake
            )
        else:
            settings.remove("DefaultMethod/Instrument/Uptake")
        if method.instrument_options.efficiency is not None:
            settings.setValue(
                "DefaultMethod/Instrument/Efficiency",
                method.instrument_options.efficiency,
            )
        else:
            settings.remove("DefaultMethod/Instrument/Efficiency")

        settings.setValue(
            "DefaultMethod/Limits/Gaussian/Alpha",
            method.limit_options.gaussian_kws["alpha"],
        )
        settings.setValue(
            "DefaultMethod/Limits/Poisson/Alpha",
            method.limit_options.poisson_kws["alpha"],
        )
        settings.setValue(
            "DefaultMethod/Limits/CompoundPoisson/Alpha",
            method.limit_options.compound_poisson_kws["alpha"],
        )
        settings.setValue(
            "DefaultMethod/Limits/CompoundPoisson/Sigma",
            method.limit_options.compound_poisson_kws["sigma"],
        )
        settings.setValue(
            "DefaultMethod/Limits/MaxIterations", method.limit_options.max_iterations
        )
        settings.setValue(
            "DefaultMethod/Limits/WindowSize", method.limit_options.window_size
        )

        settings.setValue(
            "DefaultMethod/Processing/CalibrationMode",
            method.processing_options.calibration_mode,
        )
        settings.setValue(
            "DefaultMethod/Processing/AccumulationMethod",
            method.processing_options.accumulation_method,
        )
        settings.setValue(
            "DefaultMethod/Processing/PointsRequired",
            method.processing_options.points_required,
        )
        settings.setValue(
            "DefaultMethod/Processing/ProminenceRequired",
            method.processing_options.prominence_required,
        )
        settings.setValue(
            "DefaultMethod/Processing/ClusterDistance",
            method.processing_options.cluster_distance,
        )

    # Access

    def clusters(self, data_file: SPCalDataFile, key: str) -> np.ndarray:
        if data_file not in self.processing_clusters:
            self.processing_clusters[data_file] = {}
        if key not in self.processing_clusters[data_file]:
            self.processing_clusters[data_file][key] = self.processing_methods[
                "default"
            ].processClusters(self.processing_results[data_file], key)
        return self.processing_clusters[data_file][key]

    def currentMethod(self) -> SPCalProcessingMethod:
        return self.processing_methods["default"]

    def setCurrentMethod(self, method: SPCalProcessingMethod):
        self.processing_methods["default"] = method

        self.instrument_options.optionsChanged.disconnect(
            self.onInstrumentOptionsChanged
        )
        self.instrument_options.setInstrumentOptions(method.instrument_options)
        self.instrument_options.optionsChanged.connect(self.onInstrumentOptionsChanged)

        self.isotope_options.optionChanged.disconnect(self.onIsotopeOptionChanged)
        current_isotopes = self.isotope_options.isotopes()
        for isotope, option in method.isotope_options.items():
            if isotope in current_isotopes:
                self.isotope_options.setIsotopeOption(isotope, option)
        self.isotope_options.optionChanged.connect(self.onIsotopeOptionChanged)

        self.limit_options.optionsChanged.disconnect(self.onLimitOptionsChanged)
        self.limit_options.setLimitOptions(
            method.limit_options,
        )
        self.limit_options.optionsChanged.connect(self.onLimitOptionsChanged)

        self.currentMethodChanged.emit(method)
        self.reprocess()

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
            self.dialogSessionSave,
        )

        self.action_load_session = create_action(
            "cloud-upload",
            "Restore Session",
            "Restore the saved session.",
            self.dialogSessionLoad,
        )

        # TODO: move to right click menu
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
        self.action_processing_options = create_action(
            "folder-build",
            "Processing Options",
            "Set processing options for the current method.",
            self.dialogProcessingOptions,
        )
        self.action_save_default_method = create_action(
            "document-save",
            "Set Default Method",
            "Set the current method as the default, restored when starting SPCal.",
            self.saveDefaultMethod,
        )

        # View
        current_scheme = QtCore.QSettings().value("ColorScheme", "IBM Carbon")
        self.action_color_scheme = QtGui.QActionGroup(self)
        icon_size = self.style().pixelMetric(
            QtWidgets.QStyle.PixelMetric.PM_SmallIconSize
        )
        for scheme in COLOR_SCHEMES.keys():
            action = self.action_color_scheme.addAction(scheme)
            action.setIcon(scheme_icon(scheme, icon_size, icon_size))
            action.setCheckable(True)
            if scheme == current_scheme:
                action.setChecked(True)

        self.action_custom_colors = create_action(
            "color-picker",
            "Custom...",
            "Set custom colors for the scheme.",
            self.dialogCustomColors,
        )
        self.action_color_scheme.addAction(self.action_custom_colors)

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
        self.action_github = create_action(
            "folder-git",
            "SPCal GitHub",
            "Opens a link to the GitHub.",
            self.linkToGitHub,
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
        menuedit.addSeparator()
        menuedit.addAction(self.action_processing_options)
        menuedit.addSeparator()
        menuedit.addAction(self.action_save_default_method)

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
        menuhelp.addAction(self.action_github)
        menuhelp.addAction(self.action_about)

    # Method modification

    def addExpression(self, expr: SPCalIsotopeExpression):
        method = self.currentMethod()
        if expr not in method.expressions:
            method.expressions.append(expr)
            self.currentMethodChanged.emit(method)
            self.updateForDataFiles(self.files.activeDataFiles())

    def removeExpressions(self, expressions: list[SPCalIsotopeExpression]):
        method = self.currentMethod()
        for expr in expressions:
            if expr in method.expressions:
                method.expressions.remove(expr)
            for _, results in self.processing_results.items():
                if expr in results:
                    results.pop(expr)

        self.currentMethodChanged.emit(method)
        self.updateForDataFiles(self.files.activeDataFiles())

    def setGlobalExclusionRegions(
        self, regions: list[tuple[float, float]], data_file: SPCalDataFile | None = None
    ):
        method = self.currentMethod()
        if method.exclusion_regions != regions:
            method.exclusion_regions = regions
            self.currentMethodChanged.emit(method)
            self.reprocess()

    def setExclusionRegions(
        self, regions: list[tuple[float, float]], data_file: SPCalDataFile | None = None
    ):
        if data_file is None:
            data_file = self.files.currentDataFile()
        if data_file is None:
            raise ValueError("cannot set exclusion regions, invalid data file")

        if data_file.exclusion_regions != regions:
            data_file.exclusion_regions = regions
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
        logger.info(
            f"DataFile '{data_file.path.stem}' imported with {data_file.num_events} events."
        )

    def updateForDataFiles(
        self, data_files: list[SPCalDataFile] | SPCalDataFile | None = None
    ):
        if data_files is None:
            self.isotope_options.clear()
            self.outputs.clear()
            self.graph.clear()
            return
        elif isinstance(data_files, SPCalDataFile):
            data_files = [data_files]

        # Set the options to current
        method: SPCalProcessingMethod = self.currentMethod()

        reprocess_files = []

        for file in data_files:
            exprs = [
                expr
                for expr in method.expressions
                if expr.validForIsotopes(file.isotopes)
            ]
            isotopes = sorted(file.selected_isotopes + exprs)

            method_changed = False
            self.isotope_options.blockSignals(True)
            self.isotope_options.setIsotopes(isotopes)
            for isotope in file.selected_isotopes:
                if isotope not in method.isotope_options:
                    method.isotope_options[isotope] = SPCalIsotopeOptions(
                        None, None, None
                    )
                    method_changed = True
                self.isotope_options.setIsotopeOption(
                    isotope, method.isotope_options[isotope]
                )
            self.isotope_options.blockSignals(False)
            if method_changed:
                self.currentMethodChanged.emit(method)

            # Reprocess if new isotopes exist
            if file not in self.processing_results or any(
                isotope not in self.processing_results[file] for isotope in isotopes
            ):
                reprocess_files.append(file)
            else:
                remove_isotopes = [
                    iso for iso in self.processing_results[file] if iso not in isotopes
                ]
                for iso in remove_isotopes:
                    self.processing_results[file].pop(iso)

        self.reprocess(reprocess_files)
        self.redraw()

    def onInstrumentOptionsChanged(self):
        method = self.currentMethod()
        method.instrument_options = self.instrument_options.instrumentOptions()
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
        self.currentMethodChanged.emit(method)
        self.reprocess()

    def updateOutputs(self):
        previous = self.outputs.activeResults()
        results = {
            data_file: self.processing_results[data_file]
            for data_file in self.files.activeDataFiles()
        }
        self.outputs.setResults(results)

        if any(
            data_file in previous for data_file in results
        ):  # some datafiles are shared, keep selection
            self.outputs.setActiveResults(previous)
        else:
            previous_isotopes = {
                iso for result in previous.values() for iso in result.keys()
            }
            isotopes = {iso for result in results.values() for iso in result.keys()}
            if any(isotope in previous_isotopes for isotope in isotopes):
                self.outputs.view.setSelectedIsotopes(list(previous_isotopes))
            else:
                self.outputs.view.setSelectedRows([0])
                self.outputs.view.setCurrentRow(0)

        # if any(result.isotope in previous for result in results):
        #     self.outputs.setActiveIsotopes(previous)
        # elif len(results) > 0:
        #     self.outputs.view.setCurrentIsotope([results[0].isotope])
        self.action_export.setEnabled(len(self.processing_results) > 0)

    # data modification

    def removeFileFromResults(self, data_file: SPCalDataFile):
        self.processing_results.pop(data_file)

    def removeIsotopes(self, isotopes: list[SPCalIsotope]):
        for file in self.files.dataFiles():
            for isotope in isotopes:
                if (
                    file in self.processing_results
                    and isotope in self.processing_results[file]
                ):
                    self.processing_results[file].pop(isotope)
                if isotope in file.selected_isotopes:
                    file.selected_isotopes.remove(isotope)
        self.reprocess()

    def reprocess(
        self,
        data_files: list[SPCalDataFile] | None = None,
        isotopes: list[SPCalIsotopeBase] | None = None,
    ):
        """Reprocess loaded files."""
        if data_files is None:
            data_files = self.files.dataFiles()

        method = self.currentMethod()

        for file in data_files:
            if isotopes is not None:
                existing = self.processing_results.get(file, {})
                valid_isotopes = [
                    iso for iso in isotopes if iso in file.selected_isotopes
                ]
                existing.update(method.processDataFile(file, valid_isotopes))
                self.processing_results[file] = existing
            else:
                self.processing_results[file] = method.processDataFile(
                    file, file.selected_isotopes
                )

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

        self.updateOutputs()

    # drawing
    def customColors(self) -> list[QtGui.QColor]:
        settings = QtCore.QSettings()
        colors = []
        for i in range(settings.beginReadArray("CustomColors")):
            settings.setArrayIndex(i)
            color = settings.value("Color", QtGui.QColor(0, 0, 0))
            assert isinstance(color, QtGui.QColor)
            colors.append(color)

        if len(colors) == 0:
            colors.append(QtGui.QColor(0, 0, 0))
        return colors

    def colorForIsotope(
        self, isotope: SPCalIsotopeBase, data_file: SPCalDataFile
    ) -> QtGui.QColor:
        scheme_name = str(QtCore.QSettings().value("ColorScheme", "IBM Carbon"))
        if scheme_name == "Custom":
            scheme = self.customColors()
        else:
            scheme = COLOR_SCHEMES[scheme_name]
        if isinstance(isotope, SPCalIsotope):
            idx = data_file.selected_isotopes.index(isotope)
        elif isinstance(isotope, SPCalIsotopeExpression):
            method = self.currentMethod()
            idx = method.expressions.index(isotope) + len(data_file.selected_isotopes)
        else:
            raise ValueError(f"unknown isotope type '{type(isotope)}'")

        data_files = self.files.activeDataFiles()
        for i in range(0, data_files.index(data_file)):
            idx += len(data_files[i].selected_isotopes)
        return scheme[idx % len(scheme)]

    def redraw(self):
        key = self.toolbar.currentKey()
        view = self.graph.currentView()

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
            if (
                data_file is None
                or self.processing_results[data_file].get(isotope, None) is None
            ):
                return

            self.graph.drawResultsSpectra(
                data_file,
                self.processing_results[data_file][isotope],
                None,
            )
        else:
            results = self.outputs.activeResults()
            # files = self.files.activeDataFiles()
            #
            drawable = []
            names, colors = [], []
            for data_file in results:
                for isotope, result in results[data_file].items():
                    drawable.append(result)
                    colors.append(self.colorForIsotope(isotope, data_file))
                    name = str(isotope)
                    if len(results) > 2:
                        name = data_file.path.stem + " - " + name
                    names.append(name)
            # for data_file in files:
            #     for isotope in isotopes:
            #         if isotope in self.processing_results[data_file]:
            #             drawable.append(self.processing_results[data_file][isotope])
            #             colors.append(self.colorForIsotope(isotope, data_file))

            if view == "particle":
                self.graph.drawResultsParticle(drawable, colors, names, key)
                if len(results) == 1:
                    for start, end in next(iter(results)).exclusion_regions:
                        self.graph.particle.addExclusionRegion(start, end)
                for start, end in self.currentMethod().exclusion_regions:
                    self.graph.particle.addGlobalExclusionRegion(start, end)
                self.graph.particle.action_exclusion_region.setEnabled(
                    len(results) == 1
                )
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
            self.updateForDataFiles(self.files.activeDataFiles())

        files = self.files.dataFiles()

        all_isotopes = set(files[0].isotopes)
        for file in files[1:]:
            all_isotopes.union(file.isotopes)
        all_isotopes = sorted(all_isotopes)

        dlg = CalculatorDialog(all_isotopes, method.expressions, parent=self)
        dlg.expressionsChanged.connect(set_expressions)
        dlg.open()
        return dlg

    def dialogCustomColors(self):
        def set_custom_colors(colors: list[QtGui.QColor]):
            settings = QtCore.QSettings()
            settings.remove("CustomColors")
            settings.beginWriteArray("CustomColors", len(colors))
            for i, color in enumerate(colors):
                settings.setArrayIndex(i)
                settings.setValue("Color", color)
            settings.endArray()
            settings.setValue("ColorScheme", "Custom")
            self.redraw()

        dlg = ColorDialog(colors=self.customColors(), parent=self)
        dlg.colorsSelected.connect(set_custom_colors)
        dlg.open()

    def dialogExportResults(self):
        df = self.files.currentDataFile()
        if df is None:
            return
        # ensure clusters are generated
        for key in CALIBRATION_KEYS:
            self.clusters(df, key)

        dlg = ExportDialog(
            df,
            self.processing_results[df],
            self.processing_clusters[df],
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

    def dialogManualLimits(self) -> ManualLimitDialog:
        method = self.currentMethod()
        dlg = ManualLimitDialog(
            method.limit_options.manual_limits,
            list(method.isotope_options.keys()),
            parent=self,
        )
        dlg.manualLimitsChanged.connect(self.limit_options.setManualLimits)
        dlg.open()
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
        data_file = self.files.currentDataFile()
        if data_file is None:
            return

        dlg = PeakPropertiesDialog(
            self.processing_results[data_file],
            self.toolbar.combo_isotope.currentIsotope(),
        )
        dlg.exec()

    def dialogProcessingOptions(self):
        def set_processing_options(options: SPCalProcessingOptions):
            method = self.currentMethod()
            method.processing_options = options
            self.currentMethodChanged.emit(method)
            self.reprocess()

        method = self.currentMethod()

        dlg = ProcessingOptionsDialog(method.processing_options, parent=self)
        dlg.optionsChanged.connect(set_processing_options)
        dlg.open()
        return dlg

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
        dlg.uptakeChanged.connect(
            self.instrument_options.options_widget.uptake.setBaseValue
        )
        dlg.isotopeOptionsChanged.connect(self.isotope_options.setIsotopeOption)
        dlg.isotopeOptionsChanged.connect(self.onIsotopeOptionChanged)
        dlg.open()
        return dlg

    def dialogSessionLoad(self):
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

        self.restoreSession(Path(file))

    def _stopSessionThread(self):
        if self.session_thread.isRunning():
            self.session_thread.requestInterruption()
            self.session_thread.quit()
            self.session_thread.wait(1000)
        self.files.dataFileAdded.connect(self.updateRecentFiles)

    def dialogSessionSave(self):
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

    def restoreSession(self, path: Path):
        with path.open() as fp:
            session = json.load(fp)

        datafiles = [
            (datafile, Path(datafile["path"])) for datafile in session["datafiles"]
        ]
        if len(datafiles) == 0:
            return

        missing = sum(not p.exists() for _, p in datafiles)

        if missing > 0:
            button = QtWidgets.QMessageBox.warning(
                self,
                "Datafile Not Found",
                f"Cannot find {missing} datafiles. Select new paths?",
                buttons=QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No
                | QtWidgets.QMessageBox.StandardButton.Cancel,
            )
            if button == QtWidgets.QMessageBox.StandardButton.Yes:
                new_paths = MissingPathsDialog.getMissingPaths(
                    self, [p for _, p in datafiles]
                )
                datafiles = [(df[0], new) for df, new in zip(datafiles, new_paths)]
            elif button == QtWidgets.QMessageBox.StandardButton.Cancel:
                return

        self.setCurrentMethod(decode_json_method(session["method"]))
        self.files.dataFileAdded.disconnect(self.updateRecentFiles)

        dlg = QtWidgets.QProgressDialog(
            "Importing Datafiles...", "Cancel", 0, len(datafiles), self
        )

        self.worker = SessionImportWorker(
            [(df, path) for df, path in datafiles if path.exists()]
        )

        dlg.canceled.connect(self._stopSessionThread)

        self.worker.moveToThread(self.session_thread)
        self.worker.progress.connect(dlg.setValue)
        self.worker.datafileImported.connect(self.files.addDataFile)
        self.worker.finished.connect(dlg.reset)
        self.worker.finished.connect(self._stopSessionThread)  # cleanup

        self.session_thread.started.connect(self.worker.read)
        self.session_thread.finished.connect(self.worker.deleteLater)
        self.session_thread.start()

        dlg.show()

    def linkToDocumenation(self):
        QtGui.QDesktopServices.openUrl("https://spcal.readthedocs.io")

    def linkToGitHub(self):
        QtGui.QDesktopServices.openUrl("https://github.com/djdt/spcal")

    # UI
    # save / restore layout
    def restoreLayout(self):
        settings = QtCore.QSettings()
        if settings.contains("Layout/Window/Geometry"):
            self.restoreGeometry(settings.value("Layout/Window/Geometry"))
            self.restoreState(settings.value("Layout/Window/State"))
            self.isotope_options.restoreHeaderLayout(
                settings, "Layout/IsotopeOptions/Headers"
            )
            self.outputs.restoreHeaderLayout(settings, "Layout/Outputs/Header")
        else:
            self.defaultLayout()

    def saveLayout(self):
        settings = QtCore.QSettings()
        settings.setValue("Layout/Window/Geometry", self.saveGeometry())
        settings.setValue("Layout/Window/State", self.saveState())
        self.isotope_options.saveHeaderLayout(settings, "Layout/IsotopeOptions/Headers")
        self.outputs.saveHeaderLayout(settings, "Layout/Outputs/Header")

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
        self.tabifyDockWidget(self.isotope_options, self.outputs)

        for dock in self.findChildren(QtWidgets.QDockWidget):
            dock.show()

        size = self.size()
        left_width = max(
            self.instrument_options.sizeHint().width(),
            self.limit_options.sizeHint().width(),
        )
        self.resizeDocks(
            [self.instrument_options, self.limit_options],
            [left_width, left_width, left_width],
            QtCore.Qt.Orientation.Horizontal,
        )
        self.resizeDocks(
            [self.instrument_options, self.limit_options, self.files],
            [
                int(size.height() / 6 * 1),
                int(size.height() / 6 * 3),
                int(size.height() / 6 * 2),
            ],
            QtCore.Qt.Orientation.Vertical,
        )
        self.resizeDocks(
            [self.files, self.isotope_options],
            [left_width, size.width() - left_width],
            QtCore.Qt.Orientation.Horizontal,
        )

        self.isotope_options.defaultLayout()
        self.outputs.defaultLayout()

    def reset(self):
        self.files.clear()
        self.graph.clear()
        self.outputs.clear()
        self.isotope_options.clear()
        self.toolbar.clear()
        self.setCurrentMethod(self.defaultMethod())

    def setColorScheme(self, action: QtGui.QAction):
        scheme = action.text().replace("&", "")
        scheme = scheme.rstrip(".")
        QtCore.QSettings().setValue("ColorScheme", scheme)

        self.redraw()

    def setGraphFont(self):
        settings = QtCore.QSettings()
        current = QtGui.QFont(  # type: ignore
            settings.value("GraphFont/Family", "SansSerif"),
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

    def closeEvent(self, event: QtGui.QCloseEvent):
        self.saveLayout()
        super().closeEvent(event)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        for url in event.mimeData().urls():
            if is_spcal_path(url.toLocalFile()):
                event.acceptProposedAction()
                return
            elif is_spcal_session_path(url.toLocalFile()):
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent):
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if is_spcal_path(path):
                self.dialogLoadFile(path)
                event.accept()
            elif is_spcal_session_path(path):
                self.restoreSession(path)
                event.accept()
                break

    def exceptHook(
        self, etype: type, value: BaseException, tb: TracebackType | None = None
    ):  # pragma: no cover
        """Redirect errors to the log."""
        if etype is KeyboardInterrupt:
            logger.info("Keyboard interrupt, exiting.")
            sys.exit(1)
        logger.exception("Uncaught exception", exc_info=(etype, value, tb))  # type: ignore
        QtWidgets.QMessageBox.critical(self, "Uncaught Exception", str(value))
