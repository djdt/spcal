from typing import Any

from PySide6 import QtCore, QtWidgets

from spcal.gui.modelviews import IsotopeRole
from spcal.isotope import SPCalIsotopeBase


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
        editor = IsotopeComboBox()
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
