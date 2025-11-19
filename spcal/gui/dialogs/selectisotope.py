from PySide6 import QtCore, QtWidgets

from spcal.datafile import SPCalDataFile
from spcal.gui.dialogs.nontarget import NonTargetScreeningDialog
from spcal.gui.widgets import PeriodicTableSelector


class ScreeningOptionsDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Screening Options")

        self.screening_ppm = QtWidgets.QSpinBox()
        self.screening_ppm.setRange(0, 999999)
        self.screening_ppm.setValue(100)
        self.screening_ppm.setSingleStep(10)
        self.screening_ppm.setSuffix(" ppm")
        self.screening_ppm.setToolTip(
            "The number of detection (as particles per million events) required to pass the screen."
        )

        self.screening_size = QtWidgets.QSpinBox()
        self.screening_size.setRange(0, 100000000)
        self.screening_size.setValue(1000000)
        self.screening_size.setSingleStep(100000)
        self.screening_size.setToolTip(
            "Number of events to screen, a larger number takes longer."
        )

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )

        layout = QtWidgets.QFormLayout()
        layout.addRow("Screening Threshold", self.screening_ppm)
        layout.addRow("Screening Size", self.screening_size)
        layout.addWidget(self.button_box)
        self.setLayout(layout)


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
        self.button_box.addButton(
            "Screen", QtWidgets.QDialogButtonBox.ButtonRole.ResetRole
        )
        self.button_box.clicked.connect(self.onButtonClicked)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(
            QtWidgets.QLabel(str(data_file.path)),
            0,
            QtCore.Qt.AlignmentFlag.AlignHCenter,
        )
        layout.addWidget(self.table, 1)
        layout.addWidget(self.button_box, 0)
        self.setLayout(layout)

    def onButtonClicked(self, button: QtWidgets.QAbstractButton):
        sb = self.button_box.standardButton(button)
        role = self.button_box.buttonRole(button)
        if role == QtWidgets.QDialogButtonBox.ButtonRole.ResetRole:  # screen
            self.dialogScreen()

        if sb == QtWidgets.QDialogButtonBox.StandardButton.Ok:
            self.accept()
        else:
            self.reject()

    def dialogScreen(self):
        dlg = ScreeningOptionsDialog()
        dlg.exec()

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
