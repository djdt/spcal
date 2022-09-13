import sys
import logging

from PySide2 import QtWidgets

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
        action_open_sample = QtWidgets.QAction("Open Sample", self)
        action_open_sample.triggered.connect(self.sample.dialogLoadFile)

        action_open_reference = QtWidgets.QAction("Open Reference", self)
        action_open_reference.triggered.connect(self.reference.dialogLoadFile)

        self.action_batch_process = QtWidgets.QAction("Batch Dialog", self)
        self.action_batch_process.triggered.connect(self.dialogBatchProcess)
        self.action_batch_process.setEnabled(False)

        action_clear = QtWidgets.QAction("Reset Inputs", self)
        action_clear.triggered.connect(self.resetInputs)

        action_close = QtWidgets.QAction("Quit", self)
        action_close.triggered.connect(self.close)

        action_log = QtWidgets.QAction("&Show Log", self)
        action_log.setToolTip("Show the SPCal event and error log.")
        action_log.triggered.connect(self.log.open)

        action_about = QtWidgets.QAction("About", self)
        action_about.triggered.connect(self.about)

        action_draw_detections_only = QtWidgets.QAction("Draw Dectections Only", self)
        action_draw_detections_only.setCheckable(True)
        action_draw_detections_only.setChecked(False)
        action_draw_detections_only.setToolTip("Speed up drawing by only drawing detections.")
        action_draw_detections_only.toggled.connect(self.sample.setDrawDetectionsOnly)
        action_draw_detections_only.toggled.connect(self.reference.setDrawDetectionsOnly)

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
        menuview.addAction(action_draw_detections_only)

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
        return self.sample.detections.size > 0

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
