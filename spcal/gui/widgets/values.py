"""Widget that displays a value with formatting."""

from typing import Callable

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.objects import DoubleOrEmptyValidator


class ValueWidget(QtWidgets.QAbstractSpinBox):
    valueChanged = QtCore.Signal(object)
    errorChanged = QtCore.Signal(object)

    def __init__(
        self,
        value: float | None = None,
        error: float | None = None,
        min: float = 0.0,
        max: float = np.inf,
        step: float | Callable[[float, int], float] = 1.0,
        sigfigs: int = 6,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self._value = value
        self._error = error
        self.min = min
        self.max = max
        self.step = step

        self.sigfigs = sigfigs

        # locale group separators break the double validator
        local = self.locale()
        local.setNumberOptions(
            local.numberOptions() | QtCore.QLocale.NumberOption.OmitGroupSeparator
        )
        self.setLocale(local)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

        self.lineEdit().setValidator(DoubleOrEmptyValidator(min, max, 12, parent=self))
        self.lineEdit().textEdited.connect(self.valueFromText)

        self.valueChanged.connect(self.textFromValue)
        self.errorChanged.connect(self.textFromValue)

    def error(self) -> float | None:
        return self._error

    def setError(self, error: float | None):
        if error is None or np.isnan(error):
            error = None
        if self._error != error:
            self._error = error
            self.errorChanged.emit(error)

    def value(self) -> float | None:
        return self._value

    def setValue(self, value: float | None):
        if value is None or np.isnan(value):
            value = None
        if self._value != value:
            self._value = value
            self.valueChanged.emit(value)

    def setRange(self, min: float, max: float):
        self.min, self.max = min, max
        validator = self.lineEdit().validator()
        assert isinstance(validator, QtGui.QDoubleValidator)
        validator.setRange(min, max)
        value = self.value()
        if value is not None and value > max:
            self.setValue(max)
        elif value is not None and value < min:
            self.setValue(min)

    def setSigFigs(self, sigfigs: int):
        self.sigfigs = sigfigs
        validator = self.lineEdit().validator()
        assert isinstance(validator, QtGui.QDoubleValidator)
        validator.setDecimals(sigfigs)
        self.textFromValue()

    def setStep(self, step: float | Callable[[float, int], float]):
        self.step = step

    def stepBy(self, steps: int):
        if self._value is None:
            return
        if isinstance(self.step, float | int):
            new_value = self._value + steps * self.step
        else:
            new_value = self.step(self._value, steps)
        if new_value > self.max:
            new_value = self.max
        elif new_value < self.min:
            new_value = self.min
        self.setValue(new_value)

    def stepEnabled(self) -> QtWidgets.QAbstractSpinBox.StepEnabledFlag:
        enabled = QtWidgets.QAbstractSpinBox.StepEnabledFlag.StepNone
        if self.isReadOnly():
            return enabled
        if self._value is not None:
            if self._value < self.max:
                enabled |= QtWidgets.QAbstractSpinBox.StepEnabledFlag.StepUpEnabled
            if self._value > self.min:
                enabled |= QtWidgets.QAbstractSpinBox.StepEnabledFlag.StepDownEnabled
        return enabled

    def textFromValue(self):
        if self._value is None:
            text = ""
        elif self.hasFocus() and self.isEnabled() and not self.isReadOnly():
            text = self.locale().toString(float(self._value), "g", 12)  # type: ignore
        else:
            text = self.locale().toString(float(self._value), "g", self.sigfigs)  # type: ignore
            if self._error is not None:
                text += " ± "
                text += self.locale().toString(float(self._error), "g", self.sigfigs)  # type: ignore

        self.lineEdit().setText(text)

    def valueFromText(self, text: str):
        if text == "":
            self.setValue(None)
        else:
            text = self.fixup(text)
            value, ok = self.locale().toDouble(text)
            if ok:
                value = min(value, self.max)
                value = max(value, self.min)
                self.valueChanged.disconnect(self.textFromValue)
                self.setValue(value)
                self.valueChanged.connect(self.textFromValue)

    def fixup(self, input: str) -> str:
        return self.lineEdit().validator().fixup(input)

    def focusInEvent(self, event: QtGui.QFocusEvent):
        super().focusInEvent(event)
        self.textFromValue()
        self.selectAll()  # select all text

    def focusOutEvent(self, event: QtGui.QFocusEvent):
        super().focusOutEvent(event)
        self.textFromValue()

    def showEvent(self, event: QtGui.QShowEvent):
        self.textFromValue()
        super().showEvent(event)
