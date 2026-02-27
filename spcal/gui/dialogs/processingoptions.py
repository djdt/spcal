from PySide6 import QtCore, QtWidgets
from spcal.gui.docks.processingoptions import SPCalProcessingOptionsWidget
from spcal.processing.options import SPCalProcessingOptions


class ProcessingOptionsDialog(QtWidgets.QDialog):
    optionsChanged = QtCore.Signal(SPCalProcessingOptions)

    def __init__(
        self, options: SPCalProcessingOptions, parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(parent)
        self.setWindowTitle("SPCal Processing Options")

        self.old_options = options
        self.options = SPCalProcessingOptionsWidget(options)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.options, 1)
        layout.addWidget(self.button_box, 0)
        self.setLayout(layout)

    def accept(self):
        new_options = self.options.processingOptions()
        if new_options != self.old_options:
            self.optionsChanged.emit(new_options)
        super().accept()
