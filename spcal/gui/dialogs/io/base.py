from pathlib import Path

from PySide6 import QtCore, QtWidgets

from spcal.datafile import SPCalDataFile
from spcal.gui.dialogs.selectisotope import ScreeningOptionsDialog
from spcal.gui.widgets import ElidedLabel
from spcal.processing.method import SPCalProcessingMethod


class ImportDialogBase(QtWidgets.QDialog):
    dataImported = QtCore.Signal(SPCalDataFile)

    def __init__(
        self,
        path: str | Path,
        title: str,
        screening_method: SPCalProcessingMethod | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)

        self.screening_method = screening_method
        self.screening_ppm = 100
        self.screening_size = 1000000

        self.file_path = Path(path)
        self.setWindowTitle(f"{title}: {self.file_path.name}")

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(
            False
        )
        if screening_method is not None:
            self.button_box.addButton(
                "Screen", QtWidgets.QDialogButtonBox.ButtonRole.ResetRole
            )
        self.button_box.clicked.connect(self.onButtonClicked)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        box_info = QtWidgets.QGroupBox("Information")
        self.box_info_layout = QtWidgets.QFormLayout()
        self.box_info_layout.addRow(
            "File Path:", ElidedLabel(str(self.file_path.absolute()))
        )
        box_info.setLayout(self.box_info_layout)

        box_options = QtWidgets.QGroupBox("Import Options")
        self.box_options_layout = QtWidgets.QFormLayout()
        box_options.setLayout(self.box_options_layout)

        box_layout = QtWidgets.QHBoxLayout()
        box_layout.addWidget(box_info, 1)
        box_layout.addWidget(box_options, 1)

        self.layout_body = QtWidgets.QVBoxLayout()

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(box_layout)
        layout.addLayout(self.layout_body)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

    def onButtonClicked(self, button: QtWidgets.QAbstractButton):
        if (
            self.button_box.buttonRole(button)
            == QtWidgets.QDialogButtonBox.ButtonRole.ResetRole
        ):
            self.dialogScreening()

    def dialogScreening(self):
        dlg = ScreeningOptionsDialog(
            self.screening_ppm, self.screening_size, parent=self
        )
        dlg.screeningOptionsSelected.connect(self.setScreeningParameters)
        dlg.screeningOptionsSelected.connect(self.screenDataFile)
        dlg.open()

    def setScreeningParameters(self, ppm: int, size: int):
        self.screening_ppm = ppm
        self.screening_size = size

    def screenDataFile(self, screening_target_ppm: int, screening_size: int):
        raise NotImplementedError

    def completeChanged(self):
        complete = self.isComplete()
        self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(
            complete
        )

    def isComplete(self) -> bool:
        return True
