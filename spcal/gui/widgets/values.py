"""Widget that displays a value with formatting."""
from typing import Dict

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.widgets.validcolorle import ValidColorLineEdit


class ValueWidget(ValidColorLineEdit):
    valueChanged = QtCore.Signal(object)  # object for None
    errorChanged = QtCore.Signal(object)  # object for None

    def __init__(
        self,
        value: float | None = None,
        validator: QtGui.QValidator | None = None,
        view_format: str = ".6g",
        color_invalid: QtGui.QColor | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(color_invalid=color_invalid, parent=parent)

        self._value: float | None = None
        self._error: float | None = None

        self.edit_format: str = ".16g"
        self.view_format = view_format

        if validator is None:
            validator = QtGui.QDoubleValidator(0.0, 1e99, 16)
        self.setValidator(validator)

        self.textEdited.connect(self.updateValueFromText)
        self.valueChanged.connect(self.updateTextFromValue)
        self.errorChanged.connect(self.updateTextFromValue)

    def value(self) -> float | None:
        return self._value

    def setValue(self, value: float | None) -> None:
        if self._value != value:
            self._value = value
            self.valueChanged.emit(value)

    def error(self) -> float | None:
        return self._error

    def setError(self, error: float | None) -> None:
        if self._error != error:
            self._error = error
            self.errorChanged.emit(error)

    def focusInEvent(self, event: QtGui.QFocusEvent) -> None:
        self.updateTextFromValue()
        super().focusInEvent(event)

    def focusOutEvent(self, event: QtGui.QFocusEvent) -> None:
        self.updateTextFromValue()
        super().focusOutEvent(event)

    def updateValueFromText(self) -> None:
        self.valueChanged.disconnect(self.updateTextFromValue)
        if self.text() == "":
            self.setValue(None)
        elif self.hasAcceptableInput():
            self.setValue(float(self.text()))
        self.valueChanged.connect(self.updateTextFromValue)

    def updateTextFromValue(self) -> None:
        format = self.edit_format if self.hasFocus() else self.view_format
        value = self.value()
        text = f"{value:{format}}" if value is not None else ""
        self.setText(text)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)

        if self.hasFocus() or self._error is None:  # don't draw if editing or missing
            return

        self.ensurePolished()

        rect = self.cursorRect()
        pos = rect.topRight() - QtCore.QPoint(rect.width() / 2, 0)

        layout = QtGui.QTextLayout(f" ± {self._error:{self.view_format}}", self.font())
        layout.beginLayout()
        line = layout.createLine()
        line.setLineWidth(self.width() - pos.x())
        line.setPosition(pos)
        layout.endLayout()

        painter = QtGui.QPainter(self)
        layout.draw(painter, QtCore.QPoint(0, 0))
        painter.end()


if __name__ == "__main__":
    app = QtWidgets.QApplication()

    le = ValueWidget()
    le.setError(2.232)

    dlg = QtWidgets.QDialog()
    dlg.setLayout(QtWidgets.QHBoxLayout())
    dlg.layout().addWidget(le)
    dlg.layout().addWidget(QtWidgets.QLineEdit("not me"))

    dlg.show()
    app.exec()
