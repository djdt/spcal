from PySide6 import QtCore, QtGui, QtWidgets


class RangeSlider(QtWidgets.QSlider):
    value2Changed = QtCore.Signal(int)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setOrientation(QtCore.Qt.Horizontal)

        self._value2 = 99
        self._pressed = False

    def left(self) -> int:
        return min(self.value(), self.value2())

    def setLeft(self, value: int) -> None:
        if self.value() < self._value2:  # pragma: no cover, trivial
            self.setValue(value)
        else:
            self.setValue2(value)

    def right(self) -> int:
        return max(self.value(), self.value2())

    def setRight(self, value: int) -> None:
        if self.value() > self._value2:
            self.setValue(value)
        else:  # pragma: no cover, trivial
            self.setValue2(value)

    def values(self) -> tuple[int, int]:
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
        if handle.contains(event.position().toPoint()):
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
                self.minimum(),
                self.maximum(),
                event.position().toPoint().x(),
                groove.width(),
            )
            handle.moveCenter(event.position().toPoint())
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
                self.minimum(),
                self.maximum(),
                event.position().toPoint().x(),
                groove.width(),
            )
            handle.moveCenter(event.position().toPoint())
            handle = handle.marginsAdded(
                QtCore.QMargins(
                    handle.width(), handle.width(), handle.width(), handle.width()
                )
            )
            value = self.style().sliderValueFromPosition(
                self.minimum(),
                self.maximum(),
                event.position().toPoint().x(),
                groove.width(),
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
