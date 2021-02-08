from PySide2 import QtCore, QtGui, QtWidgets

from typing import Dict, Tuple


class RangeSlider(QtWidgets.QSlider):
    value2Changed = QtCore.Signal(int)

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.setOrientation(QtCore.Qt.Horizontal)

        self._value2 = 99
        self._pressed = False

    def left(self) -> int:
        return min(self.value(), self.value2())

    def setLeft(self, value: int) -> None:
        if self.value() < self._value2:
            self.setValue(value)
        else:
            self.setValue2(value)

    def right(self) -> int:
        return max(self.value(), self.value2())

    def setRight(self, value: int) -> None:
        if self.value() > self._value2:
            self.setValue(value)
        else:
            self.setValue2(value)

    def values(self) -> Tuple[int, int]:
        return self.value(), self.value2()

    def setValues(self, left: int, right: int) -> None:
        self.setValue(left)
        self.setValue2(right)

    def value2(self) -> int:
        return self._value2

    def setValue2(self, value: int) -> None:
        self._value2 = value
        self.value2Changed.emit(self._value2)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        option = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(option)
        groove = self.style().subControlRect(
            QtWidgets.QStyle.CC_Slider, option, QtWidgets.QStyle.SC_SliderGroove, self
        )
        handle = self.style().subControlRect(
            QtWidgets.QStyle.CC_Slider, option, QtWidgets.QStyle.SC_SliderHandle, self
        )
        pos = self.style().sliderPositionFromValue(
            self.minimum(), self.maximum(), self.value2(), groove.width()
        )

        handle.moveCenter(QtCore.QPoint(pos, handle.center().y()))
        if handle.contains(event.pos()):
            event.accept()
            self._pressed = True
            self.setSliderDown(True)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._pressed:
            option = QtWidgets.QStyleOptionSlider()
            self.initStyleOption(option)
            groove = self.style().subControlRect(
                QtWidgets.QStyle.CC_Slider,
                option,
                QtWidgets.QStyle.SC_SliderGroove,
                self,
            )
            handle = self.style().subControlRect(
                QtWidgets.QStyle.CC_Slider,
                option,
                QtWidgets.QStyle.SC_SliderHandle,
                self,
            )
            value = self.style().sliderValueFromPosition(
                self.minimum(), self.maximum(), event.pos().x(), groove.width()
            )
            handle.moveCenter(event.pos())
            handle = handle.marginsAdded(
                QtCore.QMargins(
                    handle.width(), handle.width(), handle.width(), handle.width()
                )
            )
            if self.hasTracking():
                self.setValue2(value)
                self.repaint(handle)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._pressed:
            self._pressed = False
            option = QtWidgets.QStyleOptionSlider()
            self.initStyleOption(option)
            groove = self.style().subControlRect(
                QtWidgets.QStyle.CC_Slider,
                option,
                QtWidgets.QStyle.SC_SliderGroove,
                self,
            )
            handle = self.style().subControlRect(
                QtWidgets.QStyle.CC_Slider,
                option,
                QtWidgets.QStyle.SC_SliderHandle,
                self,
            )
            value = self.style().sliderValueFromPosition(
                self.minimum(), self.maximum(), event.pos().x(), groove.width()
            )
            handle.moveCenter(event.pos())
            handle = handle.marginsAdded(
                QtCore.QMargins(
                    handle.width(), handle.width(), handle.width(), handle.width()
                )
            )
            value = self.style().sliderValueFromPosition(
                self.minimum(), self.maximum(), event.pos().x(), groove.width()
            )
            self.setSliderDown(False)
            self.setValue2(value)
            self.update()

        super().mouseReleaseEvent(event)

    def paintEvent(
        self,
        event: QtGui.QPaintEvent,
    ) -> None:
        painter = QtGui.QPainter(self)
        option = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(option)
        option.activeSubControls = QtWidgets.QStyle.SC_None

        if self.isSliderDown():
            option.state |= QtWidgets.QStyle.State_Sunken
            option.activeSubControls = QtWidgets.QStyle.SC_ScrollBarSlider
        else:
            pos = self.mapFromGlobal(QtGui.QCursor.pos())
            option.activeSubControls = self.style().hitTestComplexControl(
                QtWidgets.QStyle.CC_Slider, option, pos, self
            )

        groove = self.style().subControlRect(
            QtWidgets.QStyle.CC_Slider, option, QtWidgets.QStyle.SC_SliderGroove, self
        )
        start = self.style().sliderPositionFromValue(
            self.minimum(), self.maximum(), self.left(), groove.width()
        )
        end = self.style().sliderPositionFromValue(
            self.minimum(), self.maximum(), self.right(), groove.width()
        )

        # Draw grooves
        option.subControls = QtWidgets.QStyle.SC_SliderGroove

        option.sliderPosition = self.maximum() - self.minimum() - self.left()
        option.upsideDown = not option.upsideDown

        cliprect = QtCore.QRect(groove)
        cliprect.setRight(end)
        painter.setClipRegion(QtGui.QRegion(cliprect))

        self.style().drawComplexControl(
            QtWidgets.QStyle.CC_Slider, option, painter, self
        )

        option.upsideDown = not option.upsideDown
        option.sliderPosition = self.right()
        cliprect.setLeft(start)
        cliprect.setRight(groove.right())
        painter.setClipRegion(QtGui.QRegion(cliprect))

        self.style().drawComplexControl(
            QtWidgets.QStyle.CC_Slider, option, painter, self
        )

        painter.setClipRegion(QtGui.QRegion())
        painter.setClipping(False)

        # Draw handles
        option.subControls = QtWidgets.QStyle.SC_SliderHandle

        option.sliderPosition = self.left()
        self.style().drawComplexControl(
            QtWidgets.QStyle.CC_Slider, option, painter, self
        )
        option.sliderPosition = self.right()
        self.style().drawComplexControl(
            QtWidgets.QStyle.CC_Slider, option, painter, self
        )


class ValidColorLineEdit(QtWidgets.QLineEdit):
    def __init__(self, text: str = "", parent: QtWidgets.QWidget = None):
        super().__init__(text, parent)
        self.textChanged.connect(self.revalidate)
        self.color_good = self.palette().color(QtGui.QPalette.Base)
        self.color_bad = QtGui.QColor.fromRgb(255, 172, 172)

    def setValidator(self, validator: QtGui.QValidator) -> None:
        super().setValidator(validator)
        self.revalidate()

    def revalidate(self) -> None:
        self.setValid(self.hasAcceptableInput())

    def setValid(self, valid: bool) -> None:
        palette = self.palette()
        if valid:
            color = self.color_good
        else:
            color = self.color_bad
        palette.setColor(QtGui.QPalette.Base, color)
        self.setPalette(palette)


class UnitsWidget(QtWidgets.QWidget):
    changed = QtCore.Signal()

    def __init__(
        self,
        units: Dict[str, float],
        value: float = None,
        unit: str = None,
        update_value_with_unit: bool = False,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)

        self.units = units
        self.update_value_with_unit = update_value_with_unit

        self.le = ValidColorLineEdit(str(value or ""))
        self.le.setValidator(QtGui.QDoubleValidator(0.0, 1e99, 10))
        self.le.textChanged.connect(self.changed)

        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(units.keys())
        if unit is not None:
            self.combo.setCurrentText(unit)
        self.combo.currentTextChanged.connect(self.unitChanged)

        self.previous_unit = self.combo.currentText()

        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.le, 1)
        layout.addWidget(self.combo, 0)
        self.setLayout(layout)

    def unitChanged(self, unit: str) -> None:
        if self.update_value_with_unit and self.le.hasAcceptableInput():
            base = float(self.le.text()) * self.units[self.previous_unit]
            self.setBaseValue(base)
        else:
            self.changed.emit()
        self.previous_unit = unit

    def sync(self, other: "UnitsWidget") -> None:
        self.le.textChanged.connect(other.le.setText)
        other.le.textChanged.connect(self.le.setText)
        self.combo.currentTextChanged.connect(other.combo.setCurrentText)
        other.combo.currentTextChanged.connect(self.combo.setCurrentText)

    def baseValue(self) -> float:
        unit = self.combo.currentText()
        return float(self.le.text()) * self.units[unit]

    def setBaseValue(self, base: float) -> None:
        unit = self.combo.currentText()
        self.le.setText(f"{base / self.units[unit]:.6g}")
