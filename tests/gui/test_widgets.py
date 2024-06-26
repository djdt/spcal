from PySide6 import QtCore, QtGui, QtWidgets
from pytestqt.qtbot import QtBot

from spcal.gui.widgets.checkablecombobox import CheckableComboBox
from spcal.gui.widgets.editablecombobox import EditableComboBox
from spcal.gui.widgets.rangeslider import RangeSlider
from spcal.gui.widgets.validcolorle import ValidColorLineEdit


def test_checkable_combo_box(qtbot: QtBot):
    cb = CheckableComboBox()
    qtbot.add_widget(cb)
    with qtbot.wait_exposed(cb):
        cb.show()

    cb.addItems(["a", "b", "c"])
    assert cb.count() == 3
    item = cb.model().item(0)
    assert item.isCheckable()

    cb.setCheckedItems(["b"])
    assert cb.checkedItems() == ["b"]

    cb.setCheckedItems(["b", "c"])
    assert cb.checkedItems() == ["b", "c"]

    # cannot test delegates with click


def test_editable_combo_box(qtbot: QtBot):
    cb = EditableComboBox()
    qtbot.add_widget(cb)
    with qtbot.wait_exposed(cb):
        cb.show()

    cb.addItems(["a", "b", "c", "d", "e"])

    with qtbot.wait_signal(cb.textsEdited):
        qtbot.keyClick(cb, QtCore.Qt.Key.Key_Backspace)
        qtbot.keyClick(cb, QtCore.Qt.Key.Key_Z)
        qtbot.keyClick(cb, QtCore.Qt.Key.Key_Enter)

    items = [cb.itemText(i) for i in range(cb.count())]
    assert items == ["z", "b", "c", "d", "e"]

    with qtbot.assertNotEmitted(cb.textsEdited):
        qtbot.keyClick(cb, QtCore.Qt.Key.Key_Backspace)
        qtbot.keyClick(cb, QtCore.Qt.Key.Key_A)
        cb.setCurrentIndex(1)

    dlg = cb.openEnableDialog()
    dlg.texts.item(1).setCheckState(QtCore.Qt.CheckState.Unchecked)
    dlg.accept()
    assert not QtCore.Qt.ItemFlags.ItemIsEnabled & cb.model().flags(

        cb.model().index(1, 0)
    )


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
    # Changed in a Qt update, not worth testing
    # assert slider.values() == (80, 60)  # QRangeSlider value is +7 ?
    # assert slider.left() == 60
    # assert slider.right() == 80

    slider.setLeft(10)
    assert slider.values()[1] == 10
    slider.setRight(90)
    assert slider.values()[0] == 90


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
