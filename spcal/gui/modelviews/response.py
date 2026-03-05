import logging
from typing import Any

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.datafile import SPCalDataFile
from spcal.gui.modelviews import DataFileRole, IsotopeRole
from spcal.isotope import SPCalIsotopeBase

logger = logging.getLogger(__name__)


class ConcentrationModel(QtCore.QAbstractTableModel):
    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self.isotopes: list[SPCalIsotopeBase] = []
        self.concentrations: dict[
            SPCalDataFile, dict[SPCalIsotopeBase, float | None]
        ] = {}

    def columnCount(
        self,
        parent: QtCore.QModelIndex
        | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),
    ) -> int:
        if parent.isValid():
            return 0
        return len(self.isotopes)

    def rowCount(
        self,
        parent: QtCore.QModelIndex
        | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),
    ) -> int:
        if parent.isValid():
            return 0
        return len(self.concentrations)

    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ):
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if orientation == QtCore.Qt.Orientation.Horizontal:
                return str(self.isotopes[section])
            else:
                return list(self.concentrations.keys())[section].path.stem

    def flags(
        self, index: QtCore.QModelIndex | QtCore.QPersistentModelIndex
    ) -> QtCore.Qt.ItemFlag:
        flags = super().flags(index)
        if index.isValid():
            data_file = list(self.concentrations.keys())[index.row()]
            isotope = self.isotopes[index.column()]
            if isotope in data_file.selected_isotopes:
                flags |= QtCore.Qt.ItemFlag.ItemIsEditable
            else:
                flags &= ~QtCore.Qt.ItemFlag.ItemIsEnabled
        return flags

    def data(
        self,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        role: int = QtCore.Qt.ItemDataRole.EditRole,
    ) -> Any:
        if not index.isValid():
            return None

        if role in [
            QtCore.Qt.ItemDataRole.DisplayRole,
            QtCore.Qt.ItemDataRole.EditRole,
        ]:
            data_file = list(self.concentrations.keys())[index.row()]
            isotope = self.isotopes[index.column()]
            return self.concentrations[data_file].get(isotope, None)
        elif role == QtCore.Qt.ItemDataRole.BackgroundRole:
            if not index.flags() & QtCore.Qt.ItemFlag.ItemIsEditable:
                return QtGui.QBrush(
                    QtWidgets.QApplication.palette().color(
                        QtGui.QPalette.ColorGroup.Inactive,
                        QtGui.QPalette.ColorRole.Window,
                    )
                )
        elif role == DataFileRole:
            return list(self.concentrations.keys())[index.row()]
        elif role == IsotopeRole:
            return self.isotopes[index.column()]

    def setData(
        self,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        value: Any,
        role: int = QtCore.Qt.ItemDataRole.EditRole,
    ) -> bool:
        if not index.isValid():
            return False
        if QtCore.Qt.ItemDataRole.EditRole:
            data_file = list(self.concentrations.keys())[index.row()]
            isotope = self.isotopes[index.column()]
            if value is not None:
                value = float(value)
            self.concentrations[data_file][isotope] = value
            self.dataChanged.emit(index, index)
            return True

        return False


class IntensityModel(QtCore.QAbstractTableModel):
    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self.isotopes: list[SPCalIsotopeBase] = []
        self.intensities: dict[SPCalDataFile, dict[SPCalIsotopeBase, float | None]] = {}
        self.exclusion_regions: dict[SPCalDataFile, list[tuple[float, float]]] = {}

    def columnCount(
        self,
        parent: QtCore.QModelIndex
        | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),
    ) -> int:
        if parent.isValid():
            return 0
        return len(self.isotopes)

    def rowCount(
        self,
        parent: QtCore.QModelIndex
        | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),
    ) -> int:
        if parent.isValid():
            return 0
        return len(self.intensities)

    def flags(
        self, index: QtCore.QModelIndex | QtCore.QPersistentModelIndex
    ) -> QtCore.Qt.ItemFlag:
        flags = super().flags(index)
        if index.isValid():
            data_file = list(self.intensities.keys())[index.row()]
            isotope = self.isotopes[index.column()]
            if isotope in data_file.selected_isotopes:
                flags |= QtCore.Qt.ItemFlag.ItemIsEditable
            else:
                flags &= ~QtCore.Qt.ItemFlag.ItemIsEnabled
        return flags

    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ):
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if orientation == QtCore.Qt.Orientation.Horizontal:
                return str(self.isotopes[section])
            else:
                return list(self.intensities.keys())[section].path.stem

    def data(
        self,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        role: int = QtCore.Qt.ItemDataRole.EditRole,
    ) -> Any:
        if not index.isValid():
            return None

        if role in [
            QtCore.Qt.ItemDataRole.DisplayRole,
            QtCore.Qt.ItemDataRole.EditRole,
        ]:
            data_file = list(self.intensities.keys())[index.row()]
            isotope = self.isotopes[index.column()]
            if isotope not in self.intensities[data_file]:
                if isotope in data_file.selected_isotopes:
                    mask = np.ones(data_file[isotope].shape, dtype=bool)
                    for start, end in self.exclusion_regions.get(data_file, []):
                        istart, iend = np.searchsorted(data_file.times, [start, end])
                        mask[istart:iend] = False
                    val = float(np.nanmean(data_file[isotope], where=mask))
                else:
                    val = None
                self.intensities[data_file][isotope] = val
            return self.intensities[data_file][isotope]
        elif role == QtCore.Qt.ItemDataRole.BackgroundRole:
            if not index.flags() & QtCore.Qt.ItemFlag.ItemIsEditable:
                return QtGui.QBrush(
                    QtWidgets.QApplication.palette().color(
                        QtGui.QPalette.ColorGroup.Inactive,
                        QtGui.QPalette.ColorRole.Window,
                    )
                )
        elif role == DataFileRole:
            return list(self.intensities.keys())[index.row()]
        elif role == IsotopeRole:
            return self.isotopes[index.column()]
