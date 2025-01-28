"""Widget that displays a value with formatting."""

from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.widgets.validcolorle import ValidColorLineEdit


class ValueWidget(ValidColorLineEdit):
    valueChanged = QtCore.Signal(object)  # object for None
    errorChanged = QtCore.Signal(object)  # object for None
    valueEdited = QtCore.Signal(object)  # object for None

    def __init__(
        self,
        value: float | None = None,
        validator: QtGui.QValidator | None = None,
        format: int | tuple[str, int] = 6,
        color_invalid: QtGui.QColor | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(color_invalid=color_invalid, parent=parent)

        self._value: float | None = None
        self._last_edit_value: float | None = None
        self._error: float | None = None

        if validator is None:
            validator = QtGui.QDoubleValidator(0.0, 1e99, 12)
        self.setValidator(validator)

        self.view_format = ("g", format) if isinstance(format, int) else format
        self.edit_format = ("g", 16)

        self.textEdited.connect(self.updateValueFromText)
        self.valueChanged.connect(self.updateTextFromValue)
        self.editingFinished.connect(self.checkValueEdited)

        self.setValue(value)

    def checkValueEdited(self) -> None:
        if self._value != self._last_edit_value:
            self._last_edit_value = self._value
            self.valueEdited.emit(self._value)

    def setViewFormat(self, precision: int, format: str = "g") -> None:
        self.view_format = (format, precision)
        self.updateTextFromValue()

    def setEditFormat(self, precision: int, format: str = "g") -> None:
        self.edit_format = (format, precision)

    def value(self) -> float | None:
        return self._value

    def setValue(self, value: float | None) -> None:
        if value != value:  # Check for NaN
            value = None
        if self._value != value:
            self._value = value
            self.valueChanged.emit(value)

    def error(self) -> float | None:
        return self._error

    def setError(self, error: float | None) -> None:
        if error != error:  # Check for NaN
            error = None
        if self._error != error:
            self._error = error
            self.errorChanged.emit(error)

    def focusInEvent(self, event: QtGui.QFocusEvent) -> None:
        self.updateTextFromValue()
        super().focusInEvent(event)

    def focusOutEvent(self, event: QtGui.QFocusEvent) -> None:
        self.updateTextFromValue()
        super().focusOutEvent(event)

    def isEditMode(self) -> bool:
        return self.hasFocus() and self.isEnabled() and not self.isReadOnly()

    def updateValueFromText(self) -> None:
        self.valueChanged.disconnect(self.updateTextFromValue)
        if self.text() == "":
            self.setValue(None)
        elif self.hasAcceptableInput():
            self.setValue(self.locale().toDouble(self.text())[0])
        self.valueChanged.connect(self.updateTextFromValue)

    def updateTextFromValue(self) -> None:
        format = self.edit_format if self.isEditMode() else self.view_format
        value = self.value()
        if value is None:
            self.setText("")
        else:
            self.setText(self.locale().toString(float(value), format[0], format[1]))

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)

        if self.isEditMode() or self._error is None:  # don't draw if editing or missing
            return

        self.ensurePolished()

        panel = QtWidgets.QStyleOptionFrame()
        self.initStyleOption(panel)
        fm = panel.fontMetrics

        rect = self.style().subElementRect(
            QtWidgets.QStyle.SubElement.SE_LineEditContents, panel, self
        )
        rect = rect.marginsRemoved(self.textMargins())
        rect.setX(rect.x() + fm.horizontalAdvance(self.text()))

        err_str = self.locale().toString(
            float(self._error), self.view_format[0], self.view_format[1]
        )
        text = fm.elidedText(
            f" ± {err_str}", QtCore.Qt.TextElideMode.ElideRight, rect.width()
        )

        painter = QtGui.QPainter(self)
        painter.setPen(self.palette().text().color())
        painter.drawText(rect, self.alignment(), text)
