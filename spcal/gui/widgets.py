from PySide6 import QtCore, QtGui, QtWidgets

from typing import Optional, Tuple


class DragDropRedirectFilter(QtCore.QObject):  # pragma: no cover
    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.DragEnter:
            self.parent().dragEnterEvent(event)
            return True
        elif event.type() == QtCore.QEvent.DragLeave:
            self.parent().dragLeaveEvent(event)
            return True
        elif event.type() == QtCore.QEvent.DragMove:
            self.parent().dragMoveEvent(event)
            return True
        elif event.type() == QtCore.QEvent.Drop:
            self.parent().dropEvent(event)
            return True
        return bool(super().eventFilter(obj, event))


class ElidedLabel(QtWidgets.QWidget):
    def __init__(self, text: str = "", parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._text = text
        self._elide = QtCore.Qt.ElideLeft
        self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)

    def elide(self) -> QtCore.Qt.TextElideMode:
        return self._elide

    def setElide(self, elide: QtCore.Qt.TextElideMode) -> None:
        self._elide = elide

    def text(self) -> str:
        return self._text

    def setText(self, text: str) -> None:
        self._text = text
        self.updateGeometry()

    def sizeHint(self) -> QtCore.QSize:
        fm = self.fontMetrics()
        return fm.boundingRect(self._text).size()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        fm = painter.fontMetrics()

        # width + 1 to prevent elide when text width = widget width
        elided = fm.elidedText(self._text, self._elide, self.width() + 1)
        painter.drawText(
            self.contentsRect(),
            QtCore.Qt.AlignVCenter
            | QtCore.Qt.TextSingleLine
            | QtCore.Qt.TextShowMnemonic,
            elided,
        )


class RangeSlider(QtWidgets.QSlider):
    value2Changed = QtCore.Signal(int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
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
    def __init__(
        self,
        text: str = "",
        color_good: Optional[QtGui.QColor] = None,
        color_bad: Optional[QtGui.QColor] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(text, parent)
        self.active = True
        self.textChanged.connect(self.revalidate)
        if color_good is None:
            color_good = self.palette().color(QtGui.QPalette.Base)
        if color_bad is None:
            color_bad = QtGui.QColor.fromRgb(255, 172, 172)
        self.color_good = color_good
        self.color_bad = color_bad

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
            color = self.color_good
        else:
            color = self.color_bad
        palette.setColor(QtGui.QPalette.Base, color)
        self.setPalette(palette)
