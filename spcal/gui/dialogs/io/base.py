from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from spcal.datafile import SPCalDataFile
from spcal.gui.widgets.elidedlabel import ElidedLabel
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
        self.screening_replace_isotopes = True

        self.file_path = Path(path)
        self.setWindowTitle(f"{title}: {self.file_path.name}")

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
            | QtWidgets.QDialogButtonBox.StandardButton.Reset
        )
        self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(
            False
        )
        if screening_method is not None:
            button_screen = QtWidgets.QPushButton("Screen")
            button_screen.setIcon(QtGui.QIcon.fromTheme("edit-find"))
            self.button_box.addButton(
                button_screen, QtWidgets.QDialogButtonBox.ButtonRole.ActionRole
            )
        self.button_box.clicked.connect(self.onButtonClicked)

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
        sb = self.button_box.standardButton(button)
        if sb == QtWidgets.QDialogButtonBox.StandardButton.Ok:
            self.accept()
        elif sb == QtWidgets.QDialogButtonBox.StandardButton.Reset:
            self.reset()
        elif sb == QtWidgets.QDialogButtonBox.StandardButton.Cancel:
            self.reject()

        if (
            self.button_box.buttonRole(button)
            == QtWidgets.QDialogButtonBox.ButtonRole.ActionRole
        ):
            self.dialogScreening()

    def dialogScreening(self):
        # local to prevent circular imports
        from spcal.gui.dialogs.selectisotope import ScreeningOptionsDialog

        dlg = ScreeningOptionsDialog(
            self.screening_ppm, self.screening_size, parent=self
        )
        dlg.screeningOptionsSelected.connect(self.setScreeningParameters)
        dlg.screeningOptionsSelected.connect(self.screenDataFile)
        dlg.open()

    def setScreeningParameters(self, ppm: int, size: int, replace: bool):
        self.screening_ppm = ppm
        self.screening_size = size
        self.screening_replace_isotopes = replace

    def screenDataFile(
        self, screening_target_ppm: int, screening_size: int, replace_isotopes: bool
    ):
        raise NotImplementedError

    def completeChanged(self):
        complete = self.isComplete()
        self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(
            complete
        )

    def isComplete(self) -> bool:
        return True

    def reset(self):
        pass
