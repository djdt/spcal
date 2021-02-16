from PySide2 import QtCore, QtGui, QtWidgets

from nanopart.gui.npoptions import NPOptionsWidget
from nanopart.gui.npinputs import NPSampleWidget, NPReferenceWidget
from nanopart.gui.npresults import NPResultsWidget


class NanoPartWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.currentChanged.connect(self.onTabChanged)

        self.options = NPOptionsWidget()
        self.sample = NPSampleWidget(self.options)
        self.reference = NPReferenceWidget(self.options)
        self.results = NPResultsWidget(self.options, self.sample)

        self.reference.efficiency.textChanged.connect(self.options.setEfficiency)

        self.options.optionsChanged.connect(self.onInputsChanged)
        self.sample.optionsChanged.connect(self.onInputsChanged)
        self.sample.detectionsChanged.connect(self.onInputsChanged)

        # Sync instrument wide parameters
        # self.sample.uptake.sync(self.reference.uptake)
        # self.sample.dwelltime.sync(self.reference.dwelltime)
        # self.sample.response.sync(self.reference.response)
        # self.sample.density.sync(self.reference.density)
        # self.reference.efficiency.textChanged.connect(self.sample.efficiency.setText)

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

        self.sample.loadFile("/home/tom/MEGA/Scripts/np/Sample 50 nm.csv")
        self.options.uptake.setBaseValue(0.000001566666666)
        self.options.response.setBaseValue(20e9)
        self.options.efficiency.setText("0.062")
        self.sample.density.setBaseValue(19.32e3)

    def onInputsChanged(self) -> None:
        results_enabled = self.options.isComplete() and self.sample.isComplete()
        self.tabs.setTabEnabled(self.tabs.indexOf(self.results), results_enabled)

    # def onSampleComplete(self) -> None:
    #     complete = self.sample.isComplete()
    #     self.tabs.setTabEnabled(self.tabs.indexOf(self.results), complete)

    def onTabChanged(self, index: int) -> None:
        if index == self.tabs.indexOf(self.results):
            self.results.updateResultsNanoParticle()
