from typing import Any

from PySide6 import QtCore, QtWidgets

from spcal.isotope import SPCalIsotope


class IsotopeModel(QtCore.QAbstractListModel):
    IsotopeRole = QtCore.Qt.ItemDataRole.UserRole

    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self.isotopes: list[SPCalIsotope] = []

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
        elif role == IsotopeModel.IsotopeRole:
            return isotope


class IsotopeComboBox(QtWidgets.QComboBox):
    isotopeChanged = QtCore.Signal(SPCalIsotope)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent=parent)
        self.currentIndexChanged.connect(self.onIndexChanged)

    def addIsotope(self, isotope: SPCalIsotope):
        self.addItem(str(isotope), userData=isotope)

    def addIsotopes(self, isotopes: list[SPCalIsotope]):
        for isotope in isotopes:
            self.addIsotope(isotope)

    def currentIsotope(self) -> SPCalIsotope:
        return self.isotope(self.currentIndex())

    def isotope(self, index: int) -> SPCalIsotope:
        return self.itemData(index)

    def onIndexChanged(self, index: int):
        self.isotopeChanged.emit(self.isotope(index))
