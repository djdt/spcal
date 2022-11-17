from PySide6 import QtCore, QtWidgets


class ParticleDatabaseModel(QtCore.QAbstractTableModel):
    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(parent)

class SearchRowProxyModel(QtCore.QSortFilterProxyModel):
    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self._search_string = ""

    def setSearchString(self, string: str) -> None:
        self._search_string = string
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QtCore.QModelIndex) -> bool:
        if self._search_string == "":
            return True
        idx = self.sourceModel().index(source_row, 0, source_parent)
        return self._search_string.lower() in self.sourceModel().data(idx).lower()

    def headerData(self, section: int, QtCore.Qt.Orientation, role: int) -> QtCore.QVariant:
        return self.sourceModel().headerData(section, orientation, role)
