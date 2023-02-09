"""Widget that displays a value with a coresponding unit.
[ line edit ] [combo box]
"""
from typing import Dict

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.widgets.validcolorle import ValidColorLineEdit


class UnitsWidget(QtWidgets.QWidget):
    valueChanged = QtCore.Signal()
    baseValueChanged = QtCore.Signal(float)
    baseErrorChanged = QtCore.Signal(float)

    def __init__(
        self,
        units: Dict[str, float],
        default_unit: str | None = None,
        value: float | None = None,
        validator: QtGui.QDoubleValidator | QtGui.QValidator | None = None,
        formatter: str = ".6g",
        color_invalid: QtGui.QColor | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self._base_value: float | None = None
        self._base_error: float | None = None
        self._previous_unit: str | None = None
        self._units: Dict[str, float] = {}

        self.formatter = formatter
        if validator is None:
            validator = QtGui.QDoubleValidator(0.0, 1e99, 10)
        self.valid_base_range = validator.bottom(), validator.top()

        self.lineedit = ValidColorLineEdit(color_invalid=color_invalid)
        self.lineedit.textEdited.connect(self._updateValueFromText)

        self.lineedit.setValidator(validator)

        self.valueChanged.connect(self._updateTextFromValue)

        self.combo = QtWidgets.QComboBox()
        self.combo.currentTextChanged.connect(self.unitChanged)

        self.setUnits(units)
        if default_unit is not None:
            if self.combo.currentText() == default_unit:  # pragma: no cover
                self.unitChanged(default_unit)
            else:
                self.setUnit(default_unit)
        self.setBaseValue(value)

        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.lineedit, 1)
        layout.addWidget(self.combo, 0)
        self.setLayout(layout)

    def focusOutEvent(
        self, event: QtGui.QFocusEvent
    ) -> None:  # pragma: no cover, covered by clearFocus
        super().focusOutEvent(event)
        self._updateValueFromText()

    def baseValue(self) -> float | None:
        return self._base_value

    def setBaseValue(self, base: float | None) -> None:
        if self._base_value != base:
            self._base_value = base
            self.baseValueChanged.emit(base)
            self.valueChanged.emit()

    def value(self) -> float | None:
        if self._base_value is None:
            return None
        return self._base_value / self._units[self.combo.currentText()]

    def setValue(self, value: float | str | None) -> None:
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                value = None
        if value is None:
            self.setBaseValue(None)
        else:
            self.setBaseValue(value * self._units[self.combo.currentText()])

    def baseError(self) -> float | None:
        return self._base_error

    def setBaseError(self, error: float | None) -> None:
        if self._base_error != error:
            self._base_error = error
            self.valueChanged.emit()
            self.baseErrorChanged.emit(error)

    def error(self) -> float | None:
        if self._base_error is None:
            return None
        return self._base_error / self._units[self.combo.currentText()]

    def setError(self, error: float | str | None) -> None:
        if isinstance(error, str):
            try:
                error = float(error)
            except ValueError:
                error = None
        if error is None:
            self.setBaseError(None)
        else:
            self.setBaseError(error * self._units[self.combo.currentText()])

    def unit(self) -> str:
        return self.combo.currentText()

    def setUnit(self, unit: str) -> None:
        self.combo.setCurrentText(unit)

    def setBestUnit(self) -> str:
        base = self.baseValue()
        if base is not None:
            idx = max(np.searchsorted(list(self._units.values()), base) - 1, 0)
            self.combo.setCurrentIndex(idx)
        return self.combo.currentText()

    def setUnits(self, units: dict) -> None:
        self._units = units
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItems(units.keys())
        self.combo.blockSignals(False)
        self._previous_unit = self.combo.currentText()

    def unitChanged(self, unit: str) -> None:
        self.valueChanged.emit()

        bottom = self.valid_base_range[0] / self._units[unit]
        top = self.valid_base_range[1] / self._units[unit]

        self.lineedit.validator().setBottom(bottom)
        self.lineedit.validator().setTop(top)

        self._previous_unit = unit

    def _updateTextFromValue(self) -> None:
        value = self.value()
        if value is None:
            self.lineedit.setText("")
        elif self._base_error is None:
            self.lineedit.setText(f"{value:{self.formatter}}")
        else:
            error = self.error()
            self.lineedit.setText(
                f"{value:{self.formatter}} ± {error:{self.formatter}}"
            )

    def _updateValueFromText(self) -> None:
        value = self.lineedit.text()
        self.valueChanged.disconnect(self._updateTextFromValue)
        self.setValue(value)
        self.valueChanged.connect(self._updateTextFromValue)

    def sync(self, other: "UnitsWidget") -> None:
        self.baseValueChanged.connect(other.setBaseValue)
        other.baseValueChanged.connect(self.setBaseValue)
        self.baseErrorChanged.connect(other.setBaseError)
        other.baseErrorChanged.connect(self.setBaseError)
        # self.lineedit.textChanged.connect(other.lineedit.setText)
        # other.lineedit.textChanged.connect(self.lineedit.setText)
        self.combo.currentTextChanged.connect(other.combo.setCurrentText)
        other.combo.currentTextChanged.connect(self.combo.setCurrentText)

    # Reimplementations
    def hasAcceptableInput(self) -> bool:
        return self.lineedit.hasAcceptableInput()

    def setReadOnly(self, readonly: bool) -> None:
        self.lineedit.setReadOnly(readonly)
        self.lineedit.setActive(not readonly)

    def setToolTip(self, text: str) -> None:
        self.lineedit.setToolTip(text)
        self.combo.setToolTip(text)

    def setEnabled(self, enabled: bool) -> None:
        self.lineedit.setEnabled(enabled)
        self.combo.setEnabled(enabled)

    def isEnabled(self) -> bool:
        return self.lineedit.isEnabled() and self.combo.isEnabled()
