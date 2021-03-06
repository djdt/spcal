from PySide2 import QtWidgets

from nanopart.gui.batch import BatchProcessDialog
from nanopart.gui.options import OptionsWidget
from nanopart.gui.inputs import SampleWidget, ReferenceWidget
from nanopart.gui.results import ResultsWidget


class NanoPartWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.currentChanged.connect(self.onTabChanged)

        self.options = OptionsWidget()
        self.sample = SampleWidget(self.options)
        self.reference = ReferenceWidget(self.options)
        self.results = ResultsWidget(self.options, self.sample, self.reference)

        # self.reference.efficiency.textChanged.connect(self.options.setEfficiency)

        self.options.optionsChanged.connect(self.onInputsChanged)
        self.sample.optionsChanged.connect(self.onInputsChanged)
        self.sample.detectionsChanged.connect(self.onInputsChanged)
        self.reference.optionsChanged.connect(self.onInputsChanged)
        self.reference.detectionsChanged.connect(self.onInputsChanged)

        self.tabs.addTab(self.options, "Options")
        self.tabs.addTab(self.sample, "Sample")
        self.tabs.addTab(self.reference, "Reference")
        self.tabs.addTab(self.results, "Results")
        self.tabs.setTabEnabled(self.tabs.indexOf(self.results), False)

        widget = QtWidgets.QWidget()

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.tabs, 1)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.createMenuBar()

        self.sample.loadFile("/home/tom/MEGA/Scripts/np/Sample 50 nm.csv")
        # self.reference.loadFile("/home/tom/MEGA/Scripts/np/Reference 50 nm.csv")
        self.options.uptake.setBaseValue(0.000001567)
        self.options.response.setBaseValue(20e9)
        self.options.efficiency.setText("0.062")
        self.sample.density.setBaseValue(19.32e3)

        dlg = self.dialogBatchProcess()
        dlg.files.addItem("/home/tom/MEGA/Scripts/np/Sample 15 nm.csv")
        dlg.files.addItem("/home/tom/MEGA/Scripts/np/Sample 15 nm.csv")
        dlg.files.addItem("/home/tom/MEGA/Scripts/np/Sample 15 nm.csv")
        dlg.files.addItem("/home/tom/MEGA/Scripts/np/Sample 15 nm.csv")
        dlg.files.addItem("/home/tom/MEGA/Scripts/np/Sample 15 nm.csv")
        dlg.files.addItem("/home/tom/MEGA/Scripts/np/Sample 15 nm.csv")
        dlg.files.addItem("/home/tom/MEGA/Scripts/np/Sample 15 nm.csv")
        dlg.files.addItem("/home/tom/MEGA/Scripts/np/Sample 15 nm.csv")
        dlg.files.addItem("/home/tom/MEGA/Scripts/np/Sample 15 nm.csv")
        dlg.files.addItem("/home/tom/MEGA/Scripts/np/Sample 15 nm.csv")
        dlg.files.addItem("/home/tom/MEGA/Scripts/np/Sample 15 nm.csv")
        dlg.files.addItem("/home/tom/MEGA/Scripts/np/Sample 15 nm.csv")
        dlg.files.addItem("/home/tom/MEGA/Scripts/np/Sample 15 nm.csv")
        dlg.files.addItem("/home/tom/MEGA/Scripts/np/Sample 15 nm.csv")
        dlg.files.addItem("/home/tom/MEGA/Scripts/np/Sample 15 nm.csv")
        dlg.files.addItem("/home/tom/MEGA/Scripts/np/Sample 15 nm.csv")
        dlg.files.addItem("/home/tom/MEGA/Scripts/np/Sample 15 nm.csv")
        dlg.files.addItem("/home/tom/MEGA/Scripts/np/Sample 15 nm.csv")
        dlg.files.addItem("/home/tom/MEGA/Scripts/np/Sample 15 nm.csv")
        dlg.files.addItem("/home/tom/MEGA/Scripts/np/Sample 15 nm.csv")
        dlg.files.addItem("/home/tom/MEGA/Scripts/np/Sample 15 nm.csv")
        # dlg.startProcess()

    def createMenuBar(self) -> None:
        action_open_sample = QtWidgets.QAction("Open Sample", self)
        action_open_sample.triggered.connect(self.sample.dialogLoadFile)

        action_open_reference = QtWidgets.QAction("Open Reference", self)
        action_open_reference.triggered.connect(self.reference.dialogLoadFile)

        self.action_batch_process = QtWidgets.QAction("Batch Process Files", self)
        self.action_batch_process.triggered.connect(self.dialogBatchProcess)
        self.action_batch_process.setEnabled(False)

        action_close = QtWidgets.QAction("Quit", self)
        action_close.triggered.connect(self.close)

        menufile = self.menuBar().addMenu("&File")
        menufile.addAction(action_open_sample)
        menufile.addAction(action_open_reference)
        menufile.addSeparator()
        menufile.addAction(self.action_batch_process)
        menufile.addAction(action_close)

    def dialogBatchProcess(self) -> BatchProcessDialog:
        dlg = BatchProcessDialog([], self.sample, self.reference, self.options, parent=self)
        dlg.open()
        return dlg

    def onInputsChanged(self) -> None:
        # Reference tab is neb method requires
        self.tabs.setTabEnabled(
            self.tabs.indexOf(self.reference),
            self.options.efficiency_method.currentText()
            in ["Reference", "Mass Response (None)"],
        )
        self.tabs.setTabEnabled(
            self.tabs.indexOf(self.results),
            self.readyForResults(),
        )
        self.action_batch_process.setEnabled(self.readyForResults())

    # def onSampleComplete(self) -> None:
    #     complete = self.sample.isComplete()
    #     self.tabs.setTabEnabled(self.tabs.indexOf(self.results), complete)

    def onTabChanged(self, index: int) -> None:
        if index == self.tabs.indexOf(self.results):
            self.results.updateResults()

    def readyForResults(self) -> bool:
        return all(
            [
                self.options.isComplete(),
                self.sample.isComplete(),
                self.reference.isComplete()
                or not self.tabs.isTabEnabled(self.tabs.indexOf(self.reference)),
            ]
        )
