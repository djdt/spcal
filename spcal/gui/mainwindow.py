import logging
import sys
from pathlib import Path
from types import TracebackType

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.batch import BatchProcessDialog
from spcal.gui.dialogs.calculator import CalculatorDialog
from spcal.gui.dialogs.response import ResponseDialog
from spcal.gui.dialogs.tools import MassFractionCalculatorDialog, ParticleDatabaseDialog
from spcal.gui.graphs import color_schemes
from spcal.gui.inputs import ReferenceWidget, SampleWidget
from spcal.gui.log import LoggingDialog
from spcal.gui.options import OptionsWidget
from spcal.gui.results import ResultsWidget
from spcal.gui.util import create_action
from spcal.io.session import restoreSession, saveSession

logger = logging.getLogger(__name__)


MAX_RECENT_FILES = 10


from spcal.gui.docks.instrumentoptions import SPCalInstrumentOptionsDock
from spcal.gui.limitoptions import (
    CompoundPoissonOptions,
    GaussianOptions,
    PoissonOptions,
)
from spcal.gui.widgets import CollapsableWidget
from spcal.gui.widgets.units import UnitsWidget
from spcal.gui.widgets.values import ValueWidget
from spcal.siunits import time_units

sf = 4


class SPCalLimitOptionsDock(QtWidgets.QDockWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Limit Options")

        self.window_size = ValueWidget(
            1000, format=("f", 0), validator=QtGui.QIntValidator(3, 1000000)
        )
        self.window_size.setEditFormat(0, format="f")
        self.window_size.setToolTip("Size of window for moving thresholds.")
        self.window_size.setEnabled(False)
        self.check_window = QtWidgets.QCheckBox("Use window")
        self.check_window.setToolTip(
            "Calculate threhold for each point using data from surrounding points."
        )
        self.check_window.toggled.connect(self.window_size.setEnabled)

        layout_window_size = QtWidgets.QHBoxLayout()
        layout_window_size.addWidget(self.window_size, 1)
        layout_window_size.addWidget(self.check_window, 1)

        self.limit_method = QtWidgets.QComboBox()
        self.limit_method.addItems(
            [
                "Automatic",
                "Highest",
                "Compound Poisson",
                "Gaussian",
                "Poisson",
                "Manual Input",
            ]
        )
        self.limit_method.setItemData(
            0,
            "Automatically determine the best method.",
            QtCore.Qt.ToolTipRole,
        )
        self.limit_method.setItemData(
            1, "Use the highest of Gaussian and Poisson.", QtCore.Qt.ToolTipRole
        )
        self.limit_method.setItemData(
            2,
            "Estimate ToF limits using a compound distribution based on the "
            "number of accumulations and the single ion distribution..",
            QtCore.Qt.ToolTipRole,
        )
        self.limit_method.setItemData(
            3, "Threshold using the mean and standard deviation.", QtCore.Qt.ToolTipRole
        )
        self.limit_method.setItemData(
            4,
            "Threshold using Formula C from the MARLAP manual.",
            QtCore.Qt.ToolTipRole,
        )
        self.limit_method.setItemData(
            5,
            "Manually define limits in the sample and reference tabs.",
            QtCore.Qt.ToolTipRole,
        )
        self.limit_method.setToolTip(
            self.limit_method.currentData(QtCore.Qt.ToolTipRole)
        )
        self.limit_method.currentIndexChanged.connect(
            lambda i: self.limit_method.setToolTip(
                self.limit_method.itemData(i, QtCore.Qt.ToolTipRole)
            )
        )

        self.check_iterative = QtWidgets.QCheckBox("Iterative")
        self.check_iterative.setToolTip("Iteratively filter on non detections.")

        self.button_advanced_options = QtWidgets.QPushButton("Advanced Options...")
        # self.button_advanced_options.pressed.connect(self.dialogAdvancedOptions)

        self.gaussian = GaussianOptions()
        self.poisson = PoissonOptions()
        self.compound = CompoundPoissonOptions()

        self.save_button = QtWidgets.QToolButton()
        self.save_button.setAutoRaise(True)
        self.save_button.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly
        )
        self.save_button.setIcon(QtGui.QIcon.fromTheme("document-save"))

        buttons_layout = QtWidgets.QHBoxLayout()
        buttons_layout.addWidget(self.save_button)

        gaussian_collapse = CollapsableWidget("Gaussian Limit Options")
        gaussian_collapse.setWidget(self.gaussian)
        poisson_collapse = CollapsableWidget("Poisson Limit Options")
        poisson_collapse.setWidget(self.poisson)
        compound_collapse = CollapsableWidget("Compound Poisson Limit Options")
        compound_collapse.setWidget(self.compound)

        layout = QtWidgets.QFormLayout()
        layout.addRow(buttons_layout)

        layout.addRow("Window size:", layout_window_size)
        layout.addRow("Method:", self.limit_method)

        layout.addRow(gaussian_collapse)
        layout.addRow(poisson_collapse)
        layout.addRow(compound_collapse)
        # layout.addStretch(1)
        layout.addRow(self.button_advanced_options)
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setWidget(widget)

class SPCalMainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SPCal")
        self.resize(1000, 800)

        self.log = LoggingDialog()
        self.log.setWindowTitle("SPCal Log")

        # self.tabs = QtWidgets.QTabWidget()
        # self.tabs.currentChanged.connect(self.onTabChanged)
        #
        # self.options = OptionsWidget()
        # self.sample = SampleWidget(self.options)
        # self.reference = ReferenceWidget(self.options)
        # self.results = ResultsWidget(self.options, self.sample, self.reference)
        #
        # self.options.useManualLimits.connect(self.sample.io.setLimitsEditable)
        # self.options.useManualLimits.connect(self.reference.io.setLimitsEditable)
        #
        # self.sample.io.requestIonicResponseTool.connect(self.dialogIonicResponse)
        # self.reference.io.requestIonicResponseTool.connect(self.dialogIonicResponse)
        #
        # self.sample.dataLoaded.connect(self.syncSampleAndReference)
        # self.reference.dataLoaded.connect(self.syncSampleAndReference)
        # self.sample.dataLoaded.connect(self.updateRecentFiles)
        # self.reference.dataLoaded.connect(self.updateRecentFiles)
        #
        # self.options.optionsChanged.connect(self.onInputsChanged)
        # self.sample.optionsChanged.connect(self.onInputsChanged)
        # self.sample.detectionsChanged.connect(self.onInputsChanged)
        # self.reference.optionsChanged.connect(self.onInputsChanged)
        # self.reference.detectionsChanged.connect(self.onInputsChanged)
        #
        # self.options.optionsChanged.connect(self.results.requestUpdate)
        # self.sample.optionsChanged.connect(self.results.requestUpdate)
        # self.sample.detectionsChanged.connect(self.results.requestUpdate)
        # self.reference.optionsChanged.connect(self.results.requestUpdate)
        # self.reference.detectionsChanged.connect(self.results.requestUpdate)
        #
        # self.sample.namesEdited.connect(self.updateNames)
        # self.reference.namesEdited.connect(self.updateNames)
        # self.results.io.namesEdited.connect(self.updateNames)
        #
        # self.tabs.addTab(self.options, "Options")
        # self.tabs.addTab(self.sample, "Sample")
        # self.tabs.addTab(self.reference, "Reference")
        # self.tabs.addTab(self.results, "Results")
        #
        # self.tabs.setTabEnabled(self.tabs.indexOf(self.reference), False)
        # self.tabs.setTabEnabled(self.tabs.indexOf(self.results), False)
        #
        self.instrument_options = SPCalInstrumentOptionsDock()
        self.options = SPCalLimitOptionsDock()
        self.addDockWidget(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.instrument_options
        )
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.options)
        widget = QtWidgets.QWidget()

        layout = QtWidgets.QVBoxLayout()
        # layout.addWidget(self.tabs, 1)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.createMenuBar()
        self.updateRecentFiles()

    def updateNames(self, names: dict[str, str]) -> None:
        self.sample.updateNames(names)
        self.reference.updateNames(names)
        self.results.updateNames(names)

    def createMenuBar(self) -> None:
        # File
        self.action_open_sample = create_action(
            "document-open",
            "&Open Sample File",
            "Import SP data from a CSV file.",
            lambda: self.sample.dialogLoadFile(),
        )
        self.action_open_recent = QtGui.QActionGroup(self)
        self.action_open_recent.triggered.connect(
            lambda a: self.sample.dialogLoadFile(a.text().replace("&", ""))
        )
        self.action_open_reference = create_action(
            "document-open",
            "Open &Reference File",
            "Import reference SP data from a CSV file.",
            lambda: self.reference.dialogLoadFile(),
        )

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

        menufile.addSeparator()
        menufile.addAction(self.action_open_reference)
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
            QtWidgets.QMessageBox.Information,
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

    def dialogBatchProcess(self) -> BatchProcessDialog:
        self.results.updateResults()  # Force an update
        dlg = BatchProcessDialog(
            [],
            self.sample,
            self.reference,
            self.options,
            self.results,
            parent=self,
        )
        dlg.open()
        return dlg

    def dialogCalculator(self) -> CalculatorDialog:
        dlg = CalculatorDialog(self.sample.names, self.sample.current_expr, parent=self)
        dlg.expressionAdded.connect(self.sample.addExpression)
        dlg.expressionRemoved.connect(self.sample.removeExpression)
        dlg.expressionAdded.connect(self.reference.addExpression)
        dlg.expressionRemoved.connect(self.reference.removeExpression)
        dlg.open()
        return dlg

    def dialogExportData(self) -> None:
        path, filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Data As",
            "",
            "Numpy Archive (*.npz);;CSV Document(*.csv);;All files(*)",
        )
        if path == "":
            return

        filter_suffix = filter[filter.rfind(".") : -1]

        path = Path(path)
        if filter_suffix != "":  # append suffix if missing
            path = path.with_suffix(filter_suffix)

        names = self.sample.responses.dtype.names
        if path.suffix.lower() == ".csv":
            trim = self.sample.trimRegion(names[0])
            header = " ".join(name for name in names)
            np.savetxt(
                path,
                self.sample.responses[trim[0] : trim[1]],
                delimiter="\t",
                comments="",
                header=header,
                fmt="%.16g",
            )
        elif path.suffix.lower() == ".npz":
            np.savez_compressed(
                path,
                **{name: self.sample.trimmedResponse(name) for name in names},
            )
        else:
            raise ValueError("dialogExportData: file suffix must be '.npz' or '.csv'.")

    def dialogMassFractionCalculator(self) -> MassFractionCalculatorDialog:
        dlg = MassFractionCalculatorDialog(parent=self)
        dlg.open()
        return dlg

    def dialogParticleDatabase(self) -> ParticleDatabaseDialog:
        dlg = ParticleDatabaseDialog(parent=self)
        dlg.open()
        return dlg

    def dialogIonicResponse(self) -> ResponseDialog:
        dlg = ResponseDialog(parent=self)
        dlg.responsesSelected.connect(self.sample.io.setResponses)
        dlg.responsesSelected.connect(self.reference.io.setResponses)
        dlg.open()
        return dlg

    def dialogSaveSession(self) -> None:
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

    def onInputsChanged(self) -> None:
        # Reference tab is neb method requires
        self.tabs.setTabEnabled(
            self.tabs.indexOf(self.reference),
            self.options.efficiency_method.currentText()
            in ["Reference Particle", "Mass Response"],
        )
        self.tabs.setTabEnabled(
            self.tabs.indexOf(self.results),
            self.readyForResults(),
        )
        self.action_open_batch.setEnabled(self.readyForResults())
        self.action_export.setEnabled(self.readyForResults())

    def onTabChanged(self, index: int) -> None:
        if index == self.tabs.indexOf(self.results):
            if self.results.isUpdateRequired():
                self.results.updateResults()

    def readyForResults(self) -> bool:
        return self.sample.isComplete() and (
            self.options.efficiency_method.currentText() in ["Manual Input"]
            or self.reference.isComplete()
        )

    def resetInputs(self) -> None:
        self.tabs.setCurrentIndex(0)

        self.results.resetInputs()
        self.sample.resetInputs()
        self.reference.resetInputs()
        self.options.resetInputs()

    def setColorScheme(self, action: QtGui.QAction) -> None:
        scheme = action.text().replace("&", "")
        QtCore.QSettings().setValue("colorscheme", scheme)

        self.sample.redraw()
        self.reference.redraw()
        self.results.redraw()

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
            self.options.setSignificantFigures(value)
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

    def updateRecentFiles(self, new_path: Path | None = None) -> None:
        settings = QtCore.QSettings()
        num = settings.beginReadArray("RecentFiles")
        paths = []
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
        if etype == KeyboardInterrupt:
            logger.info("Keyboard interrupt, exiting.")
            sys.exit(1)
        logger.exception("Uncaught exception", exc_info=(etype, value, tb))
        QtWidgets.QMessageBox.critical(self, "Uncaught Exception", str(value))
