"""Widget that displays a value with a coresponding unit.
[ line edit ] [combo box]
"""

import numpy as np
from PySide6 import QtCore, QtWidgets

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
        base_value_min: float = 0.0,
        base_value_max: float = np.inf,
        sigfigs: int = 6,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self._base_value: float | None = None
        self._base_error: float | None = None
        self._units: dict[str, float] = {}

        self.base_value_min = base_value_min
        self.base_value_max = base_value_max

        self._value = ValueWidget(
            min=base_value_min, max=base_value_max, sigfigs=sigfigs, parent=self
        )

        # link base and value
        self.baseValueChanged.connect(self.valueFromBase)
        self._value.valueChanged.connect(self.baseFromValue)

        self.baseErrorChanged.connect(self.errorFromBase)
        self._value.errorChanged.connect(self.baseFromError)

        self.combo = QtWidgets.QComboBox(parent=self)
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
        layout.addWidget(self._value, 3)
        layout.addWidget(self.combo, 1)
        self.setLayout(layout)

    def value(self) -> float | None:
        return self._value.value()

    def setValue(self, value: float | None) -> None:
        self._value.setValue(value)

    def error(self) -> float | None:
        return self._value.error()

    def setError(self, error: float | None) -> None:
        self._value.setError(error)

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
        min = self.base_value_min / self._units[unit]
        max = self.base_value_max / self._units[unit]

        self._value.setRange(min, max)
        self.valueFromBase(self.baseValue())
        self.errorFromBase(self.baseError())
        self.unitChanged.emit(unit)

    def setSigFigs(self, sigfigs: int):
        self._value.setSigFigs(sigfigs)

    def valueFromBase(self, base: float | None) -> None:
        self._value.valueChanged.disconnect(self.baseFromValue)
        if base is not None:
            base = base / self._units[self.unit()]
        self.setValue(base)
        self._value.valueChanged.connect(self.baseFromValue)

    def baseFromValue(self, value: float | None) -> None:
        if value is not None:
            value = value * self._units[self.unit()]
        self.setBaseValue(value)

    def errorFromBase(self, error: float | None) -> None:
        if error is not None:
            error = error / self._units[self.unit()]
        self.setError(error)

    def baseFromError(self, error: float | None) -> None:
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
            self.combo.setCurrentIndex(int(idx))
        return self.combo.currentText()

    def setUnits(self, units: dict) -> None:
        self._units = units
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItems(list(units.keys()))
        self.combo.blockSignals(False)

    # # Reimplementations
    def hasAcceptableInput(self) -> bool:
        return self._value.hasAcceptableInput()

    def setReadOnly(self, readonly: bool) -> None:
        self._value.setReadOnly(readonly)

    def setToolTip(self, text: str) -> None:
        self._value.setToolTip(text)
        self.combo.setToolTip(text)

    # def setEnabled(self, enabled: bool) -> None:
    #     self._value.setEnabled(enabled)
    #     self.combo.setEnabled(enabled)

    # def isEnabled(self) -> bool:
    #     return self.lineedit.isEnabled() and self.combo.isEnabled()


if __name__ == "__main__":
    app = QtWidgets.QApplication()
    unit = UnitsWidget({"a": 1.0, "b": 10.0}, base_value_max=2.0)
    unit.setReadOnly(True)
    unit.setEnabled(False)
    unit.resize(200, 60)
    unit.show()
    app.exec()
