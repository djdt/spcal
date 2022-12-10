import logging
import sys
from types import TracebackType

from PySide6 import QtGui, QtWidgets

from spcal import __version__
from spcal.gui.batch import BatchProcessDialog
from spcal.gui.dialogs import MassFractionCalculatorDialog, ParticleDatabaseDialog
from spcal.gui.graphs import color_schemes
from spcal.gui.inputs import ReferenceWidget, SampleWidget
from spcal.gui.log import LoggingDialog
from spcal.gui.options import OptionsWidget
from spcal.gui.results import ResultsWidget
from spcal.gui.util import create_action

logger = logging.getLogger(__name__)


class SPCalWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SPCal")
        self.resize(1000, 800)

        self.log = LoggingDialog()
        self.log.setWindowTitle("SPCal Log")

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.currentChanged.connect(self.onTabChanged)

        self.options = OptionsWidget()
        self.sample = SampleWidget(self.options)
        self.reference = ReferenceWidget(self.options)
        self.results = ResultsWidget(self.options, self.sample, self.reference)

        self.sample.dataImported.connect(self.syncSampleAndReference)
        self.reference.dataImported.connect(self.syncSampleAndReference)

        self.options.optionsChanged.connect(self.onInputsChanged)
        self.sample.optionsChanged.connect(self.onInputsChanged)
        self.sample.detectionsChanged.connect(self.onInputsChanged)
        self.reference.optionsChanged.connect(self.onInputsChanged)
        self.reference.detectionsChanged.connect(self.onInputsChanged)

        self.tabs.addTab(self.options, "Options")
        self.tabs.addTab(self.sample, "Sample")
        self.tabs.addTab(self.reference, "Reference")
        self.tabs.addTab(self.results, "Results")

        self.tabs.setTabEnabled(self.tabs.indexOf(self.reference), False)
        self.tabs.setTabEnabled(self.tabs.indexOf(self.results), False)

        widget = QtWidgets.QWidget()

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.tabs, 1)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.createMenuBar()

    def createMenuBar(self) -> None:
        # File
        self.action_open_sample = create_action(
            "document-open",
            "&Open Sample File",
            "Import SP data from a CSV file.",
            lambda: self.sample.dialogLoadFile(),
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

        self.action_close = create_action(
            "window-close", "Quit", "Exit SPCal.", self.close
        )

        # Edit
        self.action_clear = create_action(
            "edit-reset",
            "Reset Inputs",
            "Resets all the option, sample and reference inputs.",
            self.resetInputs,
        )
        self.action_mass_fraction_calculator = create_action(
            "folder-calculate",
            "Molar Ratio Calculator",
            "Calculate the molar ratios of elements in a compound.",
            self.dialogMassFractionCalculator,
        )
        self.action_particle_database = create_action(
            "folder-database",
            "Density Database",
            "Search for compound densities.",
            self.dialogParticleDatabase,
        )

        # View
        self.action_color_scheme = QtGui.QActionGroup(self)
        for scheme in color_schemes.keys():
            action = self.action_color_scheme.addAction(scheme)
            action.setCheckable(True)
        self.action_color_scheme.actions()[0].setChecked(True)
        self.action_color_scheme.triggered.connect(self.setColorScheme)

        # Help
        self.action_log = create_action(
            "dialog-information",
            "Show &Log",
            "Show the error and information log.",
            self.log.open,
        )

        self.action_about = create_action(
            "help-about", "About", "About SPCal.", self.about
        )

        menufile = self.menuBar().addMenu("&File")
        menufile.addAction(self.action_open_sample)
        menufile.addAction(self.action_open_reference)
        menufile.addSeparator()
        menufile.addAction(self.action_open_batch)
        menufile.addSeparator()
        menufile.addAction(self.action_close)

        menuedit = self.menuBar().addMenu("&Edit")
        menuedit.addAction(self.action_clear)
        menuedit.addSeparator()
        menuedit.addAction(self.action_mass_fraction_calculator)
        menuedit.addAction(self.action_particle_database)

        menuview = self.menuBar().addMenu("&View")

        menu_cs = menuview.addMenu("Color Scheme")
        menu_cs.setIcon(QtGui.QIcon.fromTheme("color-management"))
        menu_cs.setStatusTip("Change the colorscheme of the input and result graphs.")
        menu_cs.setToolTip("Change the colorscheme of the input and result graphs.")
        menu_cs.addActions(self.action_color_scheme.actions())

        menuhelp = self.menuBar().addMenu("&Help")
        menuhelp.addAction(self.action_log)
        menuhelp.addAction(self.action_about)

    def about(self) -> QtWidgets.QDialog:
        dlg = QtWidgets.QMessageBox(
            QtWidgets.QMessageBox.Information,
            "About SPCal",
            (
                "sp/scICP-MS processing.\n"
                f"Version {__version__}\n"
                "Developed by the Atomic Medicine Initiative.\n"
                "https://github.com/djdt/spcal"
            ),
            parent=self,
        )
        dlg.open()
        return dlg

    def dialogBatchProcess(self) -> BatchProcessDialog:
        dlg = BatchProcessDialog(
            [], self.sample, self.reference, self.options, parent=self
        )
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

    def onTabChanged(self, index: int) -> None:
        if index == self.tabs.indexOf(self.results):
            names = list(self.sample.detections.dtype.names)
            self.results.updateResults()

    def readyForResults(self) -> bool:
        return self.sample.isComplete() and (
            self.options.efficiency_method.currentText() in ["Manual Input"]
            or self.reference.isComplete()
        )

    def resetInputs(self) -> None:
        self.tabs.setCurrentIndex(0)

        self.options.resetInputs()
        self.sample.resetInputs()
        self.reference.resetInputs()

    def setColorScheme(self, action: QtGui.QAction) -> None:
        scheme = action.text()
        self.sample.setColorScheme(scheme)
        self.reference.setColorScheme(scheme)
        self.results.setColorScheme(scheme)

    def syncSampleAndReference(self) -> None:
        # Sync response
        for io in self.sample.io.widgets():
            if io.name in self.reference.io:
                ref_io = self.reference.io[io.name]
                sample_value = io.response.baseValue()
                ref_value = ref_io.response.baseValue()
                if sample_value is not None and ref_value is None:
                    ref_io.response.setBaseValue(sample_value)
                elif sample_value is None and ref_value is not None:
                    io.response.setBaseValue(ref_value)
                elif sample_value is not None and ref_value is not None:
                    io.setBaseValue(None)
                    ref_io.setBaseValue(None)

                io.syncOutput(ref_io, "response")

    def exceptHook(
        self, etype: type, value: BaseException, tb: TracebackType
    ) -> None:  # pragma: no cover
        """Redirect errors to the log."""
        if etype == KeyboardInterrupt:
            logger.info("Keyboard interrupt, exiting.")
            sys.exit(1)
        logger.exception("Uncaught exception", exc_info=(etype, value, tb))
        QtWidgets.QMessageBox.critical(self, "Uncaught Exception", str(value))
