import numpy as np
from PySide6 import QtCore, QtWidgets

from spcal.gui.modelviews import BasicTable


class SingleIonDialog(QtWidgets.QDialog):
    distributionSelected = QtCore.Signal(np.ndarray)

    def __init__(self, dist: np.ndarray, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Single Ion Distribution")

        self.table = BasicTable()

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.table, 1)
        layout.addWidget(self.button_box, 0)
        self.setLayout(layout)

    def accept(self) -> None:
        self.distributionSelected.emit([])
        super().accept()
