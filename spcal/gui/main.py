import sys
import logging

from PySide6 import QtGui, QtWidgets

from spcal import __version__

from spcal.gui.batch import BatchProcessDialog
from spcal.gui.inputs import SampleWidget, ReferenceWidget
from spcal.gui.log import LoggingDialog
from spcal.gui.options import OptionsWidget
from spcal.gui.results import ResultsWidget

from types import TracebackType

logger = logging.getLogger(__name__)

class NanoPartWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SPCal")

        self.log = LoggingDialog()
        self.log.setWindowTitle("SPCal Log")

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.currentChanged.connect(self.onTabChanged)

        self.options = OptionsWidget()
        self.sample = SampleWidget(self.options)
        self.reference = ReferenceWidget(self.options)
        self.results = ResultsWidget(self.options, self.sample, self.reference)

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
        action_open_sample = QtGui.QAction("Open Sample", self)
        action_open_sample.triggered.connect(self.sample.dialogLoadFile)

        action_open_reference = QtGui.QAction("Open Reference", self)
        action_open_reference.triggered.connect(self.reference.dialogLoadFile)

        self.action_batch_process = QtGui.QAction("Batch Dialog", self)
        self.action_batch_process.triggered.connect(self.dialogBatchProcess)
        self.action_batch_process.setEnabled(False)

        action_clear = QtGui.QAction("Reset Inputs", self)
        action_clear.triggered.connect(self.resetInputs)

        action_close = QtGui.QAction("Quit", self)
        action_close.triggered.connect(self.close)

        action_log = QtGui.QAction("&Show Log", self)
        action_log.setToolTip("Show the SPCal event and error log.")
        action_log.triggered.connect(self.log.open)

        action_about = QtGui.QAction("About", self)
        action_about.triggered.connect(self.about)

        action_draw_all= QtGui.QAction("Draw All", self)
        action_draw_all.setToolTip("Draw all data points.")
        action_draw_all.toggled.connect(lambda: self.sample.setDrawMode("all"))
        action_draw_all.toggled.connect(lambda: self.reference.setDrawMode("all"))

        action_draw_background= QtGui.QAction("Draw Above Background", self)
        action_draw_background.setToolTip("Only draw points above the background threshold.")
        action_draw_background.toggled.connect(lambda: self.sample.setDrawMode("background"))
        action_draw_background.toggled.connect(lambda: self.reference.setDrawMode("background"))

        action_draw_detections= QtGui.QAction("Draw Dectections", self)
        action_draw_detections.setToolTip("Only draw detections, greatly speeds up drawing.")
        action_draw_detections.toggled.connect(lambda: self.sample.setDrawMode("detections"))
        action_draw_detections.toggled.connect(lambda: self.reference.setDrawMode("detections"))
        
        action_group_draw_mode = QtGui.QActionGroup(self)
        action_group_draw_mode.addAction(action_draw_all)
        action_group_draw_mode.addAction(action_draw_background)
        action_group_draw_mode.addAction(action_draw_detections)

        for action in action_group_draw_mode.actions():
            action.setCheckable(True)
        action_draw_all.setChecked(True)

        menufile = self.menuBar().addMenu("&File")
        menufile.addAction(action_open_sample)
        menufile.addAction(action_open_reference)
        menufile.addSeparator()
        menufile.addAction(self.action_batch_process)
        menufile.addSeparator()
        menufile.addAction(action_close)

        menuedit = self.menuBar().addMenu("&Edit")
        menuedit.addAction(action_clear)

        menuview = self.menuBar().addMenu("&View")
        menuview.addActions(action_group_draw_mode.actions())

        menuhelp = self.menuBar().addMenu("&Help")
        menuhelp.addAction(action_log)
        menuhelp.addAction(action_about)

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
        self.action_batch_process.setEnabled(self.readyForResults())

    def onTabChanged(self, index: int) -> None:
        if index == self.tabs.indexOf(self.results):
            self.results.updateResults()

    def readyForResults(self) -> bool:
        return len(self.sample.detections) > 0

    def resetInputs(self) -> None:
        self.tabs.setCurrentIndex(0)

        self.options.resetInputs()
        self.sample.resetInputs()
        self.reference.resetInputs()

    def exceptHook(
        self, etype: type, value: BaseException, tb: TracebackType
    ) -> None:  # pragma: no cover
        """Redirect errors to the log."""
        if etype == KeyboardInterrupt:
            logger.info("Keyboard interrupt, exiting.")
            sys.exit(1)
        logger.exception("Uncaught exception", exc_info=(etype, value, tb))
        QtWidgets.QMessageBox.critical(self, "Uncaught Exception", str(value))
