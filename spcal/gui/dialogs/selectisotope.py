from PySide6 import QtCore, QtWidgets

from spcal.datafile import SPCalDataFile
from spcal.gui.widgets import PeriodicTableSelector


class SelectIsotopesDialog(QtWidgets.QDialog):
    isotopesSelected = QtCore.Signal(SPCalDataFile)

    def __init__(
        self, data_file: SPCalDataFile, parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(parent)
        self.setWindowTitle("Select Isotopes")

        self.table = PeriodicTableSelector(
            data_file.isotopes, data_file.selected_isotopes
        )
        self.data_file = data_file

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(
            QtWidgets.QLabel(str(data_file.path)),
            0,
            QtCore.Qt.AlignmentFlag.AlignHCenter,
        )
        layout.addWidget(self.table, 1)
        layout.addWidget(self.button_box, 0)
        self.setLayout(layout)

    def completeChanged(self):
        complete = self.isComplete()
        self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(
            complete
        )

    def isComplete(self) -> bool:
        return len(self.table.selectedIsotopes()) > 0

    def accept(self):
        self.data_file.selected_isotopes = self.table.selectedIsotopes()
        self.isotopesSelected.emit(self.data_file)
        super().accept()
