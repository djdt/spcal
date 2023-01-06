from PySide6 import QtCore, QtGui, QtWidgets
from pytestqt.qtbot import QtBot

from spcal.gui.widgets.rangeslider import RangeSlider
from spcal.gui.widgets.validcolorle import ValidColorLineEdit


def test_range_slider(qtbot: QtBot):
    slider = RangeSlider()
    qtbot.add_widget(slider)
    with qtbot.wait_exposed(slider):
        slider.show()

    slider.setValues(0, 100)

    option = QtWidgets.QStyleOptionSlider()
    slider.initStyleOption(option)
    groove = slider.style().subControlRect(
        QtWidgets.QStyle.CC_Slider, option, QtWidgets.QStyle.SC_SliderGroove, slider
    )
    handle = slider.style().subControlRect(
        QtWidgets.QStyle.CC_Slider, option, QtWidgets.QStyle.SC_SliderHandle, slider
    )

    qtbot.mousePress(
        slider,
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
        QtCore.QPoint(
            slider.style().sliderPositionFromValue(
                slider.minimum(), slider.maximum(), slider.value2(), groove.width()
            ),
            handle.center().y(),
        ),
    )
    qtbot.mouseMove(
        slider,
        QtCore.QPoint(
            slider.style().sliderPositionFromValue(
                slider.minimum(), slider.maximum(), 60, groove.width()
            ),
            handle.center().y(),
        ),
    )
    qtbot.mouseRelease(
        slider,
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
        QtCore.QPoint(
            slider.style().sliderPositionFromValue(
                slider.minimum(), slider.maximum(), 60, groove.width()
            ),
            handle.center().y(),
        ),
    )
    assert slider.values() == (0, 60)
    assert slider.left() == 0
    assert slider.right() == 60

    qtbot.mousePress(
        slider,
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
        QtCore.QPoint(
            slider.style().sliderPositionFromValue(
                slider.minimum(), slider.maximum(), slider.value(), groove.width()
            ),
            handle.center().y(),
        ),
    )
    qtbot.mouseMove(
        slider,
        QtCore.QPoint(
            slider.style().sliderPositionFromValue(
                slider.minimum(), slider.maximum(), 74, groove.width()
            ),
            handle.center().y(),
        ),
    )
    qtbot.mouseRelease(
        slider,
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
        QtCore.QPoint(
            slider.style().sliderPositionFromValue(
                slider.minimum(), slider.maximum(), 74, groove.width()
            ),
            handle.center().y(),
        ),
    )
    assert slider.values() == (80, 60)  # QRangeSlider value is +7 ?
    assert slider.left() == 60
    assert slider.right() == 80

    slider.setLeft(10)
    assert slider.values() == (80, 10)
    slider.setRight(90)
    assert slider.values() == (90, 10)


def test_valid_color_line_edit(qtbot: QtBot):
    le = ValidColorLineEdit("")
    qtbot.add_widget(le)
    with qtbot.wait_exposed(le):
        le.show()

    assert le.palette().color(QtGui.QPalette.Base) == le.color_valid
    le.setValidator(QtGui.QIntValidator(0, 100))
    assert le.palette().color(QtGui.QPalette.Base) == le.color_invalid
    le.setText("5")
    assert le.palette().color(QtGui.QPalette.Base) == le.color_valid
    le.setText("a")
    assert le.palette().color(QtGui.QPalette.Base) == le.color_invalid
    le.setActive(False)
    assert le.palette().color(QtGui.QPalette.Base) == le.color_valid
    le.setActive(True)
    le.setEnabled(False)
    assert le.palette().color(QtGui.QPalette.Base) == le.color_valid
