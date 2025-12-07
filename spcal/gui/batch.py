from PySide6 import QtCore, QtGui, QtWidgets

from spcal.datafile import SPCalDataFile, SPCalNuDataFile
from spcal.processing.method import SPCalProcessingMethod


class BatchOptions(QtWidgets.QGroupBox):
    def __init__(self, title: str, parent: QtWidgets.QWidget | None = None):
        super().__init__(title, parent=parent)

        # self.setLayout()


class NuBatchOptions(BatchOptions):
    def __init__(
        self, data_file: SPCalNuDataFile, parent: QtWidgets.QWidget | None = None
    ):
        super().__init__("Nu Instruments", parent)

        self.check_chunked = QtWidgets.QCheckBox("Process files in chunks.")
        self.chunk_size = QtWidgets.QSpinBox()
        self.chunk_size.setRange(1, 10000)
        self.chunk_size.setSingleStep(100)
        self.chunk_size.setValue(data_file.integ_range[1] - data_file.integ_range[0])


class BatchProcessingDialog(QtWidgets.QDialog):
    def __init__(
        self, method: SPCalProcessingMethod, parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(parent)
        self.setWindowTitle("Batch Processing")

