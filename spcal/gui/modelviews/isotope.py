from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.modelviews import IsotopeRole
from spcal.isotope import (
    ISOTOPE_TABLE,
    RECOMMENDED_ISOTOPES,
    REGEX_ISOTOPE,
    SPCalIsotope,
    SPCalIsotopeBase,
)


class IsotopeModel(QtCore.QAbstractListModel):
    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self.isotopes: list[SPCalIsotopeBase] = []

    def rowCount(
        self,
        parent: QtCore.QModelIndex
        | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),
    ) -> int:
        return len(self.isotopes)

    def data(
        self,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if not index.isValid():
            return None

        isotope = self.isotopes[index.row()]

        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            return str(isotope)
        elif role == IsotopeRole:
            return isotope


class IsotopeComboBox(QtWidgets.QComboBox):
    isotopeChanged = QtCore.Signal(SPCalIsotopeBase)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent=parent)
        self.currentIndexChanged.connect(self.onIndexChanged)

    def addIsotope(self, isotope: SPCalIsotopeBase):
        index = self.count()
        self.insertItem(index, str(isotope))
        self.setItemData(index, isotope, IsotopeRole)

    def addIsotopes(self, isotopes: list[SPCalIsotopeBase]):
        for isotope in isotopes:
            self.addIsotope(isotope)

    def currentIsotope(self) -> SPCalIsotopeBase:
        return self.isotope(self.currentIndex())

    def setCurrentIsotope(self, isotope: SPCalIsotopeBase):
        self.setCurrentIndex(self.findData(isotope, role=IsotopeRole))

    def isotope(self, index: int) -> SPCalIsotopeBase:
        return self.itemData(index, role=IsotopeRole)

    def onIndexChanged(self, index: int):
        self.isotopeChanged.emit(self.isotope(index))


class IsotopeComboDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(
        self, isotopes: list[SPCalIsotopeBase], parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(parent)
        self.isotopes = isotopes

    def createEditor(
        self,
        parent: QtWidgets.QWidget,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ) -> QtWidgets.QWidget:
        editor = IsotopeComboBox(parent)
        editor.addIsotopes(self.isotopes)
        editor.setCurrentIsotope(index.data(IsotopeRole))
        return editor

    def setEditorData(
        self,
        editor: QtWidgets.QWidget,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ):
        assert isinstance(editor, IsotopeComboBox)
        editor.setCurrentIsotope(index.data(IsotopeRole))

    def setModelData(
        self,
        editor: QtWidgets.QWidget,
        model: QtCore.QAbstractItemModel,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ):
        assert isinstance(editor, IsotopeComboBox)
        model.setData(index, editor.currentIsotope(), IsotopeRole)
        model.setData(
            index, str(editor.currentIsotope()), QtCore.Qt.ItemDataRole.DisplayRole
        )


class IsotopeNameValidator(QtGui.QValidator):
    def validate(self, input: str, pos: int) -> tuple[QtGui.QValidator.State, str, int]:
        match = REGEX_ISOTOPE.fullmatch(input)
        if match is None:
            return QtGui.QValidator.State.Intermediate, input, pos
        symbol = match.group(2)
        if match.group(1) is not None:
            isotope = int(match.group(1))
        elif match.group(3) is not None:
            isotope = int(match.group(3))
        else:
            return QtGui.QValidator.State.Intermediate, input, pos

        if (symbol, isotope) in ISOTOPE_TABLE:
            return QtGui.QValidator.State.Acceptable, input, pos
        else:
            return QtGui.QValidator.State.Intermediate, input, pos

    def fixup(self, input: str) -> str:
        match = REGEX_ISOTOPE.fullmatch(input)
        if match is None:
            return input
        if (
            len(match.group(2)) == 2
            and match.group(1) is None
            and match.group(3) is None
        ):
            if match.group(2) in RECOMMENDED_ISOTOPES:
                return input + str(RECOMMENDED_ISOTOPES[match.group(2)])
        elif match.group(1) is not None:
            return match.group(1) + match.group(2)
        elif match.group(3) is not None:
            return match.group(2) + match.group(3)

        return input


class IsotopeNameDelegate(QtWidgets.QItemDelegate):
    ISOTOPE_COMPLETER_STRINGS = list(
        f"{symbol}{isotope}" for symbol, isotope in ISOTOPE_TABLE.keys()
    ) + list(f"{isotope}{symbol}" for symbol, isotope in ISOTOPE_TABLE.keys())

    def createEditor(
        self,
        parent: QtWidgets.QWidget,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ) -> QtWidgets.QWidget:
        editor = QtWidgets.QLineEdit(parent=parent)
        editor.setText(index.data(QtCore.Qt.ItemDataRole.EditRole))
        editor.setValidator(IsotopeNameValidator())
        editor.setCompleter(
            QtWidgets.QCompleter(IsotopeNameDelegate.ISOTOPE_COMPLETER_STRINGS)
        )
        return editor

    def setEditorData(
        self,
        editor: QtWidgets.QWidget,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ):
        assert isinstance(editor, QtWidgets.QLineEdit)
        editor.setText(index.data(QtCore.Qt.ItemDataRole.EditRole))

    def setModelData(
        self,
        editor: QtWidgets.QWidget,
        model: QtCore.QAbstractItemModel,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ):
        assert isinstance(editor, QtWidgets.QLineEdit)
        model.setData(index, editor.text(), QtCore.Qt.ItemDataRole.EditRole)
        try:
            isotope = SPCalIsotope.fromString(editor.text())
            model.setData(index, isotope, IsotopeRole)
            model.setData(
                index,
                QtGui.QPalette.ColorRole.Text,
                QtCore.Qt.ItemDataRole.ForegroundRole,
            )
        except NameError:
            model.setData(
                index,
                QtGui.QPalette.ColorRole.Accent,
                QtCore.Qt.ItemDataRole.ForegroundRole,
            )
