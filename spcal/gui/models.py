from typing import Any, Dict, List, Type

import numpy as np
from PySide6 import QtCore


class NumpyRecArrayTableModel(QtCore.QAbstractTableModel):
    """Access a numpy structured array through a table.

    Args:
        array: ndim > 2
        axes: axes to view as (column, row)
        column_formats: dict of column names and formatting strings for display
        parent: parent object
    """

    def __init__(
        self,
        array: np.ndarray,
        fill_values: Dict[Type, Any] | None = None,
        column_formats: Dict[str, str] | None = None,
        parent: QtCore.QObject | None = None,
    ):
        assert array.ndim == 1
        assert array.dtype.names is not None

        super().__init__(parent)

        _column_formats = {}
        if column_formats is not None:
            _column_formats.update(column_formats)
        _fill_values = {float: np.nan, str: "", int: -1}
        if fill_values is not None:
            _fill_values.update(fill_values)

        self.array = array

        self.fill_values = _fill_values
        self.column_formats = _column_formats

    # Rows and Columns
    def columnCount(self, parent: QtCore.QModelIndex | None = None) -> int:
        return len(self.array.dtype.names)  # type: ignore

    def rowCount(self, parent: QtCore.QModelIndex | None = None) -> int:
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
            return self.column_formats.get(name, "{}").format(value)
        else:  # pragma: no cover
            return None

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlags:
        if not index.isValid():  # pragma: no cover
            return QtCore.Qt.ItemIsEnabled

        return super().flags(index)

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


class SearchColumnsProxyModel(QtCore.QSortFilterProxyModel):
    def __init__(self, columns: List[int], parent: QtCore.QObject | None = None):
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
