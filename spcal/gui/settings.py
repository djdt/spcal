from PySide6 import QtCore, QtGui, QtWidgets

# from typing import Dict
from spcal.gui.graphs import color_schemes

default_settings = {"display": {"scheme": "IBM Carbon", "sigfigs": "4"}}


class SettingsDialog(QtWidgets.QDialog):
    """Display current settings."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)

        settings = QtCore.QSettings()
        if settings.status() != QtCore.QSettings.Status.NoError:
            raise RuntimeError("SettingsDialog: QSettings returned error.")

        self.setWindowTitle(f"{settings.applicationName()} Settings")

        self.scheme = QtWidgets.QComboBox()
        self.scheme.addItems(list(color_schemes.keys()))
        self.scheme.setCurrentText(
            settings.value("display/scheme", default_settings["display"]["scheme"])
        )

        self.sigfigs = QtWidgets.QLineEdit(
            settings.value("display/sigfigs", default_settings["display"]["sigfigs"])
        )
        self.sigfigs.setValidator(QtGui.QIntValidator(1, 10))
        self.sigfigs.textChanged.connect(self.completeChanged)

        box_display = QtWidgets.QGroupBox("Display")
        box_display.setLayout(QtWidgets.QFormLayout())
        box_display.layout().addRow("Color Scheme", self.scheme)
        box_display.layout().addRow("Sig. Figures", self.sigfigs)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(box_display)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

    def isComplete(self) -> bool:
        if not self.sigfigs.hasAcceptableInput():
            return False
        return True

    def completeChanged(self) -> None:
        self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(
            self.isComplete()
        )

    def accept(self) -> None:
        settings = QtCore.QSettings()
        if settings.status() != QtCore.QSettings.Status.NoError:
            raise RuntimeError("SettingsDialog: QSettings returned error.")

        if self.sigfigs.text() != default_settings["display"]["sigfigs"]:
            settings.setValue("display/sigfigs", self.sigfigs.text())

        if self.scheme.currentText() != default_settings["display"]["scheme"]:
            settings.setValue("display/scheme", self.scheme.currentText())
        # for group in self.lineedits.keys():
        #     for key, le in self.lineedits[group].items():
        #         if le.text() != default_settings[group][key]:
        #             settings.setValue(group + "/" + key, le.text())

        super().accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication()
    app.setApplicationName("test")
    app.setOrganizationName("test")

    dlg = SettingsDialog()
    dlg.show()
    app.exec()
