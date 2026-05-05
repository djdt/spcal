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
    COLUMNS = {
        0: (
            "Density",
            "Particle density, meaure externally or lookup in the density database",
        ),
        1: ("Response", "The signal produced per mass of the measured element"),
        2: (
            "Mass Fraction",
            "The fraction of measured element in a particle, can be entered as a molecular formula",
        ),
        3: (
            "Diameter",
            "The diameter of a reference particle, for calculating transport efficiency",
        ),
        4: ("Concentration", "The mass concentration of a reference particle solution"),
        5: ("Mass Response", "The average signal per mass of a reference particle"),
    }

    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(
            list(n for n, _ in IsotopeOptionModel.COLUMNS.values()),
            ["g/cm³", "L/µg", "", "nm", "µg/L", "ag"],
            [
                density_units,
                response_units,
                {},
                size_units,
                mass_concentration_units,
                mass_units,
            ],
            units_tooltips=list(tt for _, tt in IsotopeOptionModel.COLUMNS.values()),
            parent=parent,
        )

        self.isotope_options: dict[SPCalIsotopeBase, SPCalIsotopeOptions] = {}

    def rowCount(
        self,
        parent: QtCore.QModelIndex
        | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),
    ) -> int:
        return len(self.isotope_options)

    def columnCount(
        self,
        parent: QtCore.QModelIndex
        | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),
    ) -> int:
        return len(IsotopeOptionModel.COLUMNS)

    def flags(
        self, index: QtCore.QModelIndex | QtCore.QPersistentModelIndex
    ) -> QtCore.Qt.ItemFlag:
        flags = super().flags(index)
        if IsotopeOptionModel.COLUMNS[index.column()] != "Mass Response":
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
        name = IsotopeOptionModel.COLUMNS[index.column()]
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
        name = IsotopeOptionModel.COLUMNS[index.column()]

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
            self.dataChanged.emit(index, index, [role])
            return super().setData(index, value, role)
