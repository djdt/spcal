from PySide6 import QtWidgets
from pytestqt.qtbot import QtBot

from spcal.gui.objects import (
    DoubleOrEmptyValidator,
    DoubleOrPercentValidator,
)


def test_double_or_empty_validator(qtbot: QtBot):
    le = QtWidgets.QLineEdit("")
    le.setValidator(DoubleOrEmptyValidator(-1.0, 1.0, 4))
    qtbot.add_widget(le)

    # empty
    assert le.hasAcceptableInput()
    le.setText("2.0")
    assert not le.hasAcceptableInput()
    le.setText("0.00001")
    assert not le.hasAcceptableInput()


def test_double_or_percent_validator(qtbot: QtBot):
    le = QtWidgets.QLineEdit("")
    le.setValidator(
        DoubleOrPercentValidator(
            -1.0, 1.0, decimals=2, percent_bottom=0.0, percent_top=10.0
        )
    )
    qtbot.add_widget(le)

    # empty
    assert not le.hasAcceptableInput()
    le.setText("1.0")
    assert le.hasAcceptableInput()
    le.setText("2.0")
    assert not le.hasAcceptableInput()
    le.setText("0.5000")
    assert not le.hasAcceptableInput()
    le.setText("2%")
    assert le.hasAcceptableInput()
    le.setText("2.0%")
    assert le.hasAcceptableInput()
    le.setText("2.0 %")
    assert not le.hasAcceptableInput()
    le.setText("2.0%%")
    assert not le.hasAcceptableInput()
    le.setText("20%")
    assert not le.hasAcceptableInput()
