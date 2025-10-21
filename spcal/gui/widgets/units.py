"""Widget that displays a value with a coresponding unit.
[ line edit ] [combo box]
"""

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.widgets.values import ValueWidget


class UnitsWidget(QtWidgets.QWidget):
    baseValueChanged = QtCore.Signal(object)
    baseErrorChanged = QtCore.Signal(object)
    unitChanged = QtCore.Signal(str)

    def __init__(
        self,
        units: dict[str, float],
        default_unit: str | None = None,
        base_value: float | None = None,
        validator: QtGui.QDoubleValidator | QtGui.QValidator | None = None,
        format: int | tuple[str, int] = 6,
        color_invalid: QtGui.QColor | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self._base_value: float | None = None
        self._base_error: float | None = None
        self._units: dict[str, float] = {}

        self.lineedit = ValueWidget(
            validator=validator,
            format=format,
            color_invalid=color_invalid,
        )
        self.valid_base_range = (
            self.lineedit.validator().bottom(),
            self.lineedit.validator().top(),
        )

        # link base and value
        self.baseValueChanged.connect(self.updateValueFromBase)
        self.lineedit.valueChanged.connect(self.updateBaseFromValue)

        self.baseErrorChanged.connect(self.updateErrorFromBase)
        self.lineedit.errorChanged.connect(self.updateBaseFromError)

        self.combo = QtWidgets.QComboBox()
        self.combo.currentTextChanged.connect(self.onUnitChanged)

        self.setUnits(units)
        if default_unit is not None:
            if self.combo.currentText() == default_unit:  # pragma: no cover
                self.onUnitChanged(default_unit)
            else:
                self.setUnit(default_unit)
        self.setBaseValue(base_value)

        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.lineedit, 3)
        layout.addWidget(self.combo, 1)
        self.setLayout(layout)

    def setViewFormat(self, precision: int, format: str = "g") -> None:
        self.lineedit.setViewFormat(precision, format)

    def setEditFormat(self, precision: int, format: str = "g") -> None:
        self.lineedit.setEditFormat(precision, format)

    def value(self) -> float | None:
        return self.lineedit.value()

    def setValue(self, value: float | None) -> None:
        self.lineedit.setValue(value)

    def error(self) -> float | None:
        return self.lineedit.error()

    def setError(self, error: float | None) -> None:
        self.lineedit.setError(error)

    def baseValue(self) -> float | None:
        return self._base_value

    def setBaseValue(self, value: float | None) -> None:
        if self._base_value != value:
            self._base_value = value
            self.baseValueChanged.emit(value)

    def baseError(self) -> float | None:
        return self._base_error

    def setBaseError(self, error: float | None) -> None:
        if self._base_error != error:
            self._base_error = error
            self.baseErrorChanged.emit(error)

    def onUnitChanged(self, unit: str) -> None:
        bottom = self.valid_base_range[0] / self._units[unit]
        top = self.valid_base_range[1] / self._units[unit]

        self.lineedit.validator().setBottom(bottom)
        self.lineedit.validator().setTop(top)
        self.updateValueFromBase()
        self.updateErrorFromBase()
        self.unitChanged.emit(unit)

    def updateValueFromBase(self) -> None:
        self.lineedit.valueChanged.disconnect(self.updateBaseFromValue)
        base = self.baseValue()
        if base is not None:
            base = base / self._units[self.unit()]
        self.setValue(base)
        self.lineedit.valueChanged.connect(self.updateBaseFromValue)

    def updateBaseFromValue(self) -> None:
        value = self.value()
        if value is not None:
            value = value * self._units[self.unit()]
        self.setBaseValue(value)

    def updateErrorFromBase(self) -> None:
        error = self.baseError()
        if error is not None:
            error = error / self._units[self.unit()]
        self.setError(error)

    def updateBaseFromError(self) -> None:
        error = self.error()
        if error is not None:
            error = error * self._units[self.unit()]
        self.setBaseError(error)

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
