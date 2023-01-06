from PySide6 import QtGui, QtWidgets


class ValidColorLineEdit(QtWidgets.QLineEdit):
    def __init__(
        self,
        text: str = "",
        color_valid: QtGui.QColor | None = None,
        color_invalid: QtGui.QColor | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(text, parent)
        self.active = True

        if color_valid is None:
            color_valid = self.palette().color(QtGui.QPalette.Base)
        self.color_valid = color_valid

        if color_invalid is None:
            color_invalid = QtGui.QColor.fromRgb(255, 172, 172)
        self.color_invalid = color_invalid

        self.textChanged.connect(self.revalidate)

    def setActive(self, active: bool) -> None:
        self.active = active
        self.revalidate()

    def setEnabled(self, enabled: bool) -> None:
        super().setEnabled(enabled)
        self.revalidate()

    def setValidator(self, validator: QtGui.QValidator) -> None:
        super().setValidator(validator)
        self.revalidate()

    def revalidate(self) -> None:
        self.setValid(self.hasAcceptableInput() or not self.isEnabled())

    def setValid(self, valid: bool) -> None:
        palette = self.palette()
        if valid or not self.active:
            palette.setColor(QtGui.QPalette.Base, self.color_valid)
        else:
            palette.setColor(QtGui.QPalette.Base, self.color_invalid)
        self.setPalette(palette)
