from PySide2 import QtCore, QtGui, QtWidgets

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
        self.results = ResultsWidget(self.options, self.sample)

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

        self.sample.loadFile("/home/tom/MEGA/Scripts/np/Sample 15 nm.csv")
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
