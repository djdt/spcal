from typing import Any

from PySide6 import QtCore

from spcal.gui.modelviews import (
    BaseValueErrorRole,
    BaseValueRole,
    IsotopeOptionRole,
    IsotopeRole,
)
from spcal.gui.modelviews.units import UnitsModel
from spcal.isotope import SPCalIsotopeBase
from spcal.processing.options import SPCalIsotopeOptions
from spcal.siunits import (
    density_units,
    mass_concentration_units,
    mass_units,
    response_units,
    size_units,
)


class IsotopeOptionModel(UnitsModel):
    COLUMN_LABELS = {
        0: "Density",
        1: "Response",
        2: "Mass Fraction",
        3: "Diameter",
        4: "Concentration",
        5: "Mass Response",
    }
    COLUMN_TOOLTIPS = {
        0: "Particle density, meaure externally or lookup in the density database",
        1: "The signal produced per mass of the measured element",
        2: "The fraction of measured element in a particle, can be entered as a molecular formula",
        3: "The diameter of a reference particle, for calculating transport efficiency",
        4: "The mass concentration of a reference particle solution",
        5: "The average signal per mass of a reference particle",
    }

    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(
            list(IsotopeOptionModel.COLUMN_LABELS.values()),
            ["g/cm³", "L/µg", "", "nm", "µg/L", "ag"],
            [
                density_units,
                response_units,
                {},
                size_units,
                mass_concentration_units,
                mass_units,
            ],
            units_tooltips=list(IsotopeOptionModel.COLUMN_TOOLTIPS.values()),
            parent=parent,
        )

        self.isotope_options: dict[SPCalIsotopeBase, SPCalIsotopeOptions] = {}

    def rowCount(
        self,
        parent: QtCore.QModelIndex
        | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),
    ) -> int:
        if parent.isValid():
            return 0
        return len(self.isotope_options)

    def columnCount(
        self,
        parent: QtCore.QModelIndex
        | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),
    ) -> int:
        if parent.isValid():
            return 0
        return len(IsotopeOptionModel.COLUMN_LABELS)

    def flags(
        self, index: QtCore.QModelIndex | QtCore.QPersistentModelIndex
    ) -> QtCore.Qt.ItemFlag:
        if not index.isValid():
            return QtCore.Qt.ItemFlag()
        flags = super().flags(index)
        if IsotopeOptionModel.COLUMN_LABELS[index.column()] != "Mass Response":
            flags ^= QtCore.Qt.ItemFlag.ItemIsEditable
        return flags

    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if orientation == QtCore.Qt.Orientation.Vertical:
            if role in [
                QtCore.Qt.ItemDataRole.DisplayRole,
                QtCore.Qt.ItemDataRole.EditRole,
            ]:
                return str(list(self.isotope_options.keys())[section])
        return super().headerData(section, orientation, role)

    def data(
        self,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if not index.isValid():  # pragma: no cover
            return None

        isotope = list(self.isotope_options.keys())[index.row()]
        name = IsotopeOptionModel.COLUMN_LABELS[index.column()]
        if role == IsotopeRole:
            return isotope
        elif role == IsotopeOptionRole:
            return self.isotope_options[isotope]
        elif role == BaseValueRole:
            if name == "Density":
                return self.isotope_options[isotope].density
            elif name == "Response":
                return self.isotope_options[isotope].response
            elif name == "Mass Fraction":
                return self.isotope_options[isotope].mass_fraction
            elif name == "Diameter":
                return self.isotope_options[isotope].diameter
            elif name == "Concentration":
                return self.isotope_options[isotope].concentration
            elif name == "Mass Response":
                return self.isotope_options[isotope].mass_response
            else:
                raise ValueError(f"unknown column name '{name}'")
        elif role == BaseValueErrorRole:
            return None
        else:
            return super().data(index, role)

    def setData(
        self,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        value: Any,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> bool:
        if not index.isValid():
            return False

        isotope = list(self.isotope_options.keys())[index.row()]
        name = IsotopeOptionModel.COLUMN_LABELS[index.column()]

        if role == IsotopeRole:
            return False  # cannot set isotope
        elif role == IsotopeOptionRole:
            self.isotope_options[isotope] = value
            tl, br = (
                self.index(index.row(), 0),
                self.index(index.row(), self.columnCount() - 1),
            )
            self.dataChanged.emit(tl, br, [role])
            return True
        elif role in [BaseValueRole]:
            if name == "Density":
                self.isotope_options[isotope].density = value
            elif name == "Response":
                self.isotope_options[isotope].response = value
            elif name == "Mass Fraction":
                self.isotope_options[isotope].mass_fraction = value
            elif name == "Diameter":
                self.isotope_options[isotope].diameter = value
            elif name == "Concentration":
                self.isotope_options[isotope].concentration = value
            elif name == "Mass Response":
                self.isotope_options[isotope].mass_response = value
            else:
                raise ValueError(f"unknown column name '{name}'")
            self.dataChanged.emit(index, index, [role])
            return True
        else:
            return super().setData(index, value, role)
