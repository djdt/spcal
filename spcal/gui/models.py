from typing import Any

import numpy as np
from PySide6 import QtCore


class NumpyRecArrayTableModel(QtCore.QAbstractTableModel):
    """Access a numpy structured array through a table.

    Args:
        array: 1d array
        fill_values: default value for missing data for each type
        column_formats: dict of column names and formatting strings for display
        parent: parent object
    """

    def __init__(
        self,
        array: np.ndarray,
        fill_values: dict[str, Any] | None = None,
        column_formats: dict[str, str] | None = None,
        column_flags: dict[str, QtCore.Qt.ItemFlag] | None = None,
        parent: QtCore.QObject | None = None,
    ):
        assert array.ndim == 1
        assert array.dtype.names is not None

        super().__init__(parent)

        self.array = array

        self.fill_values = {"f": np.nan, "U": "", "i": -1, "u": 0}
        if fill_values is not None:
            self.fill_values.update(fill_values)

        self.column_formats = {}
        if column_formats is not None:
            self.column_formats.update(column_formats)

        self.column_flags = {}
        if column_flags is not None:
            self.column_flags.update(column_flags)

    # Rows and Columns
    def columnCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self.array.dtype.names)  # type: ignore

    def rowCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return self.array.shape[0]

    # Data
    def data(
        self, index: QtCore.QModelIndex, role: int = QtCore.Qt.DisplayRole
    ) -> str | None:
        if not index.isValid():
            return None

        if role in (QtCore.Qt.DisplayRole, QtCore.Qt.EditRole):
            name = self.array.dtype.names[index.column()]
            value = self.array[name][index.row()]
            if np.isreal(value) and np.isnan(value):
                return ""
            return self.column_formats.get(name, "{}").format(value)
        else:  # pragma: no cover
            return None

    def setData(
        self, index: QtCore.QModelIndex, value: Any, role: QtCore.Qt.ItemFlag
    ) -> bool:
        name = self.array.dtype.names[index.column()]
        if role == QtCore.Qt.EditRole:
            if value == "":
                value = self.fill_values[self.array[name].dtype.kind]
            self.array[name][index.row()] = value
            self.dataChanged.emit(index, index, [role])
            return True
        return False

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlags:
        if not index.isValid():  # pragma: no cover
            return 0

        name = self.array.dtype.names[index.column()]
        return self.column_flags.get(
            name, super().flags(index) | QtCore.Qt.ItemIsEditable
        )

    # Header
    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: QtCore.Qt.ItemDataRole,
    ) -> str | None:
        if role != QtCore.Qt.DisplayRole:  # pragma: no cover
            return None

        if orientation == QtCore.Qt.Orientation.Horizontal:
            return self.array.dtype.names[section]
        else:
            return str(section)

    def insertRows(
        self, pos: int, rows: int, parent: QtCore.QModelIndex = QtCore.QModelIndex()
    ) -> bool:
        self.beginInsertRows(parent, pos, pos + rows - 1)
        empty = np.array(
            [
                tuple(
                    self.fill_values[d.kind]
                    for d, v in self.array.dtype.fields.values()
                )
            ],
            dtype=self.array.dtype,
        )
        self.array = np.insert(self.array, pos, np.full(rows, empty))
        self.endInsertRows()
        return True


class SearchColumnsProxyModel(QtCore.QSortFilterProxyModel):
    def __init__(self, columns: list[int], parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self._search_string = ""
        self.columns = columns

    def setSearchString(self, string: str) -> None:
        self._search_string = string
        self.invalidateFilter()

    def filterAcceptsRow(
        self, source_row: int, source_parent: QtCore.QModelIndex
    ) -> bool:
        if self._search_string == "":
            return True
        tokens = self._search_string.lower().split(" ")
        for column in self.columns:
            idx = self.sourceModel().index(source_row, column, source_parent)
            if all(x in self.sourceModel().data(idx).lower() for x in tokens):
                return True
        return False
