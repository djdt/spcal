from PySide2 import QtCore, QtGui, QtWidgets
import numpy as np

from nanopart.gui.widgets import ValidColorLineEdit

from typing import Dict, Tuple, Union


class UnitsWidget(QtWidgets.QWidget):
    valueChanged = QtCore.Signal()

    def __init__(
        self,
        units: Dict[str, float],
        default_unit: str = None,
        value: float = None,
        validator: Tuple[float, float, int] = (0.0, 1e99, 10),
        invalid_color: QtGui.QColor = None,
        update_value_with_unit: bool = False,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)

        self.units = units
        self.update_value_with_unit = update_value_with_unit
        self.valid_range = validator[0], validator[1]

        self.lineedit = ValidColorLineEdit(color_bad=invalid_color)
        self.lineedit.editingFinished.connect(self.valueChanged)
        self.lineedit.setValidator(QtGui.QDoubleValidator(*validator))

        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(units.keys())
        if default_unit is not None:
            if self.combo.currentText() == default_unit:
                self.unitChanged(default_unit)
            else:
                self.setUnit(default_unit)
        self.combo.currentTextChanged.connect(self.unitChanged)

        self._previous_unit = self.combo.currentText()
        self.setBaseValue(value)

        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.lineedit, 1)
        layout.addWidget(self.combo, 0)
        self.setLayout(layout)

    def value(self) -> float:
        return float(self.lineedit.text())

    def setValue(self, value: Union[float, str]) -> None:
        if isinstance(value, float):
            decimals = self.lineedit.validator().decimals()
            self.lineedit.setText(f"{value:.{decimals}g}")
            if not self.lineedit.hasAcceptableInput():
                self.lineedit.setText(f"{value:.{decimals}f}")
        else:
            self.lineedit.setText(value)

    def baseValue(self) -> float:
        unit = self.combo.currentText()
        if not self.lineedit.hasAcceptableInput():
            return None
        return self.value() * self.units[unit]

    def setBaseValue(self, base: float) -> None:
        if base is None:
            self.setValue("")
        else:
            unit = self.combo.currentText()
            self.setValue(base / self.units[unit])

    def unit(self) -> str:
        return self.combo.currentText()

    def setUnit(self, unit: str) -> None:
        self.combo.setCurrentText(unit)

    def setBestUnit(self) -> str:
        base = self.baseValue()
        idx = max(np.searchsorted(list(self.units.values()), base) - 1, 0)
        self.combo.setCurrentIndex(idx)
        return self.combo.currentText()

    def unitChanged(self, unit: str) -> None:
        if self.update_value_with_unit and self.lineedit.hasAcceptableInput():
            base = float(self.lineedit.text()) * self.units[self._previous_unit]
            self.setBaseValue(base)
        else:
            self.valueChanged.emit()

        bottom = self.valid_range[0] / self.units[unit]
        top = self.valid_range[1] / self.units[unit]

        self.lineedit.validator().setBottom(bottom)
        self.lineedit.validator().setTop(top)

        self._previous_unit = unit

    def sync(self, other: "UnitsWidget") -> None:
        self.lineedit.textChanged.connect(other.lineedit.setText)
        other.lineedit.textChanged.connect(self.lineedit.setText)
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
        self.lineedit.revalidate()
