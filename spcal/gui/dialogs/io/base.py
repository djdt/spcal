from pathlib import Path

from PySide6 import QtCore, QtWidgets

from spcal.datafile import SPCalDataFile
from spcal.gui.widgets import ElidedLabel


class ImportDialogBase(QtWidgets.QDialog):
    dataImported = QtCore.Signal(SPCalDataFile)

    def __init__(
        self,
        path: str | Path,
        title: str,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)

        self.file_path = Path(path)
        self.setWindowTitle(f"{title}: {self.file_path.name}")

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(
            False
        )
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

    def completeChanged(self) -> None:
        complete = self.isComplete()
        self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(
            complete
        )

    def isComplete(self) -> bool:
        return True
