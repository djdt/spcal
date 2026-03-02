import re

from PySide6 import QtCore, QtGui, QtWidgets

from spcal.npdb import db

from spcal.gui.objects import DoubleOrEmptyValidator
from spcal.gui.widgets.values import ValueWidget
from spcal.gui.modelviews.values import ValueWidgetDelegate


class MassFractionValidator(DoubleOrEmptyValidator):
    regex = re.compile("([A-Z][a-z]?)([0-9\\.]*)")

    def validate(self, input: str, pos: int) -> tuple[QtGui.QValidator.State, str, int]:
        valid = super().validate(input, pos)
        if valid[0] == QtGui.QValidator.State.Invalid:
            return QtGui.QValidator.State.Intermediate, input, pos
        return valid

    def searchInput(self, input: str) -> list[tuple[str, float]]:
        found = []
        pos = 0
        while pos < len(input):
            m = MassFractionValidator.regex.match(input, pos)
            if m is None or m.group(1) not in db["elements"]["Symbol"]:
                return []
            found.append((m.group(1), float(m.group(2) or 1.0)))
            pos = m.end()
        return found

    def fixup(self, input: str) -> str:
        matches = self.searchInput(input)
        if len(matches) == 0:
            return input

        mw = 0.0
        for symbol, number in matches:
            mw += db["elements"]["MW"][db["elements"]["Symbol"] == symbol][0] * number

        first = (
            db["elements"]["MW"][db["elements"]["Symbol"] == matches[0][0]][0]
            * matches[0][1]
        )
        return f"{first / mw:.12g}"


class MassFractionDelegate(ValueWidgetDelegate):
    def createEditor(
        self,
        parent: QtWidgets.QWidget,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ) -> QtWidgets.QWidget:
        editor = super().createEditor(parent, option, index)
        assert isinstance(editor, ValueWidget)
        editor.lineEdit().setValidator(MassFractionValidator(0.0, 1.0, 12))
        return editor


#
#
# class DensityValidator(DoubleOrEmptyValidator):
#     def __init__(
#         self,
#         bottom: float,
#         top: float,
#         decimals: int,
#         unit: str | None = None,
#         parent: QtCore.QObject | None = None,
#     ):
#         super().__init__(bottom, top, decimals, parent)
#         if unit is None:
#             unit = "g/cm³"
#         self.unit = unit
#
#     def validate(self, input: str, pos: int) -> tuple[QtGui.QValidator.State, str, int]:
#         valid = super().validate(input, pos)
#         if valid[0] == QtGui.QValidator.State.Invalid:
#             return QtGui.QValidator.State.Intermediate, input, pos
#         return valid
#
#     def fixup(self, input: str) -> str:
#         if input in db["inorganic"]["Name"]:
#             density = db["inorganic"]["Density"][db["inorganic"]["Name"] == input][0]
#         elif input in db["polymer"]["Name"]:
#             density = db["polymer"]["Density"][db["polymer"]["Name"] == input][0]
#         else:
#             return input
#
#         density *= 1000.0
#         density *= density_units[self.unit]
#
#         return f"{density:.12g}"
#
#
# class DensityDelegate(ValueWidgetDelegate):
#     def createEditor(
#         self,
#         parent: QtWidgets.QWidget,
#         option: QtWidgets.QStyleOptionViewItem,
#         index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
#     ) -> QtWidgets.QWidget:
#         editor = super().createEditor(parent, option, index)
#         assert isinstance(editor, ValueWidget)
#         editor.lineEdit().setValidator(
#             DensityValidator(
#                 editor.min, editor.max, 12, unit=index.data(CurrentUnitRole)
#             )
#         )
#         return editor
#
#
