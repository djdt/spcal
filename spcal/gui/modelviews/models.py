from typing import Any

import numpy as np
from PySide6 import QtCore


class NumpyRecArrayTableModel(QtCore.QAbstractTableModel):
    """Access a numpy structured array through a table.

    Args:
        array: 1d array
        orientation: direction of array data, vertical = names in columns
        fill_values: default value for missing data for each type
        name_formats: dict of array names and formatting strings for display
        name_flags: dict of array names and model flags
        parent: parent object
    """

    def __init__(
        self,
        array: np.ndarray,
        orientation: QtCore.Qt.Orientation = QtCore.Qt.Orientation.Vertical,
        fill_values: dict[str, Any] | None = None,
        name_formats: dict[str, str] | None = None,
        name_flags: dict[str, QtCore.Qt.ItemFlag] | None = None,
        parent: QtCore.QObject | None = None,
    ):
        assert array.ndim == 1
        assert array.dtype.names is not None

        super().__init__(parent)

        self.array = array
        self.orientation = orientation

        self.fill_values = {"f": np.nan, "U": "", "i": -1, "u": 0}
        if fill_values is not None:
            self.fill_values.update(fill_values)

        self.name_formats = {}
        if name_formats is not None:
            self.name_formats.update(name_formats)

        self.name_flags = {}
        if name_flags is not None:
            self.name_flags.update(name_flags)

    # Rows and Columns
    def columnCount(
        self,
        parent: QtCore.QModelIndex
        | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),
    ) -> int:
        if parent.isValid():
            return 0
        if self.orientation == QtCore.Qt.Orientation.Horizontal:
            return self.array.shape[0]
        else:
            return len(self.array.dtype.names)  # type: ignore

    def rowCount(
        self,
        parent: QtCore.QModelIndex
        | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),
    ) -> int:
        if parent.isValid():
            return 0
        if self.orientation == QtCore.Qt.Orientation.Horizontal:
            return len(self.array.dtype.names)  # type: ignore
        else:
            return self.array.shape[0]

    # Data
    def data(
        self,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> str | None:
        if not index.isValid() or self.array.dtype.names is None:
            return None

        row, column = index.row(), index.column()
        if self.orientation == QtCore.Qt.Orientation.Horizontal:
            row, column = column, row

        if role in (
            QtCore.Qt.ItemDataRole.DisplayRole,
            QtCore.Qt.ItemDataRole.EditRole,
        ):
            name = self.array.dtype.names[column]
            value = self.array[name][row]
            if np.isreal(value) and np.isnan(value):
                return ""
            return self.name_formats.get(name, "{}").format(value)
        else:  # pragma: no cover
            return None

    def setData(
        self,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        value: Any,
        role: int = QtCore.Qt.ItemDataRole.EditRole,
    ) -> bool:
        if not index.isValid() or self.array.dtype.names is None:
            return False

        row, column = index.row(), index.column()
        if self.orientation == QtCore.Qt.Orientation.Horizontal:
            row, column = column, row

        name = self.array.dtype.names[column]
        if role == QtCore.Qt.ItemDataRole.EditRole:
            if value == "":
                value = self.fill_values[self.array[name].dtype.kind]
            self.array[name][row] = value
            self.dataChanged.emit(index, index, [role])
            return True
        return False

    def flags(
        self, index: QtCore.QModelIndex | QtCore.QPersistentModelIndex
    ) -> QtCore.Qt.ItemFlag:
        if not index.isValid() or self.array.dtype.names is None:  # pragma: no cover
            return QtCore.Qt.ItemFlag.NoItemFlags

        idx = (
            index.column()
            if self.orientation == QtCore.Qt.Orientation.Vertical
            else index.row()
        )

        name = self.array.dtype.names[idx]
        return self.name_flags.get(
            name, super().flags(index) | QtCore.Qt.ItemFlag.ItemIsEditable
        )

    # Header
    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> str | None:
        if role != QtCore.Qt.ItemDataRole.DisplayRole:  # pragma: no cover
            return None

        if orientation == self.orientation:
            return str(section)
        else:
            assert self.array.dtype.names is not None
            return self.array.dtype.names[section]

    def insertColumns(
        self,
        pos: int,
        columns: int,
        parent: QtCore.QModelIndex
        | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),
    ) -> bool:
        if self.orientation == QtCore.Qt.Orientation.Vertical:
            raise NotImplementedError("name insert is not implemented")

        self.beginInsertColumns(parent, pos, pos + columns - 1)
        empty = np.array(
            [
                tuple(
                    self.fill_values[d.kind]
                    for d, v in self.array.dtype.fields.values()  # type: ignore
                )
            ],
            dtype=self.array.dtype,
        )
        self.array = np.insert(self.array, pos, np.full(columns, empty))
        self.endInsertColumns()
        return True

    def insertRows(
        self,
        pos: int,
        rows: int,
        parent: QtCore.QModelIndex
        | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),
    ) -> bool:
        if self.orientation == QtCore.Qt.Orientation.Horizontal:
            raise NotImplementedError("name insert is not implemented")

        self.beginInsertRows(parent, pos, pos + rows - 1)
        empty = np.array(
            [
                tuple(
                    self.fill_values[d.kind]
                    for d, v in self.array.dtype.fields.values()  # type: ignore
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

    def setSearchString(self, string: str):
        self.beginFilterChange()
        self._search_string = string
        self.endFilterChange()

    def filterAcceptsRow(
        self,
        source_row: int,
        source_parent: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ) -> bool:
        if self._search_string == "":
            return True
        tokens = self._search_string.lower().split(" ")
        for column in self.columns:
            idx = self.sourceModel().index(source_row, column, source_parent)
            if all(x in self.sourceModel().data(idx).lower() for x in tokens):
                return True
        return False
