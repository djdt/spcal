from PySide2 import QtWidgets

from spcal import __version__

from spcal.gui.batch import BatchProcessDialog
from spcal.gui.options import OptionsWidget
from spcal.gui.inputs import SampleWidget, ReferenceWidget
from spcal.gui.results import ResultsWidget


class NanoPartWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SPCal")

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

        action_clear = QtWidgets.QAction("Reset", self)
        action_clear.triggered.connect(self.resetInputs)

        action_close = QtWidgets.QAction("Quit", self)
        action_close.triggered.connect(self.close)

        action_about = QtWidgets.QAction("About", self)
        action_about.triggered.connect(self.about)

        menufile = self.menuBar().addMenu("&File")
        menufile.addAction(action_open_sample)
        menufile.addAction(action_open_reference)
        menufile.addSeparator()
        menufile.addAction(self.action_batch_process)
        menufile.addSeparator()
        menufile.addAction(action_close)

        menuedit = self.menuBar().addMenu("&Edit")
        menuedit.addAction(action_clear)

        menuhelp = self.menuBar().addMenu("&Help")
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
