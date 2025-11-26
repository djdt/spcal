import logging
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.dialogs.tools import MassFractionCalculatorDialog, ParticleDatabaseDialog
from spcal.gui.modelviews.basic import BasicTableView
from spcal.gui.modelviews.units import UnitsHeaderView, UnitsModel
from spcal.gui.modelviews.values import ValueWidgetDelegate
from spcal.isotope import SPCalIsotope
from spcal.processing import SPCalIsotopeOptions
from spcal.siunits import (
    density_units,
    response_units,
    mass_concentration_units,
    mass_units,
    size_units,
)

logger = logging.getLogger(__name__)


class IsotopeOptionModel(UnitsModel):
    COLUMNS = {
        0: "Density",
        1: "Response",
        2: "Mass Fraction",
        3: "Diameter",
        4: "Concentration",
        5: "Mass Response",
    }

    IsotopeRole = QtCore.Qt.ItemDataRole.UserRole
    IsotopeOptionRole = QtCore.Qt.ItemDataRole.UserRole + 1

    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(parent=parent)

        self.unit_labels = list(IsotopeOptionModel.COLUMNS.values())
        self.current_unit = ["g/cm³", "L/µg", "", "nm", "µg/L", "ag"]
        self.units = [
            density_units,
            response_units,
            {},
            size_units,
            mass_concentration_units,
            mass_units,
        ]

        self.isotope_options: dict[SPCalIsotope, SPCalIsotopeOptions] = {}

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
        if not index.isValid():
            return None

        isotope = list(self.isotope_options.keys())[index.row()]
        name = IsotopeOptionModel.COLUMNS[index.column()]
        if role == IsotopeOptionModel.IsotopeRole:
            return isotope
        elif role == IsotopeOptionModel.IsotopeOptionRole:
            return self.isotope_options[isotope]
        elif role == UnitsModel.BaseValueRole:
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
        elif role == UnitsModel.BaseErrorRole:
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

        if role == IsotopeOptionModel.IsotopeRole:
            return False  # cannot set isotope
        elif role == IsotopeOptionModel.IsotopeOptionRole:
            self.isotope_options[isotope] = value
            tl, br = (
                self.index(index.row(), 0),
                self.index(index.row(), self.columnCount() - 1),
            )
            self.dataChanged.emit(tl, br, [role])
            return True
        elif role in [UnitsModel.BaseValueRole]:
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


class IsotopeOptionTable(BasicTableView):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent=parent)
        self.isotope_model = IsotopeOptionModel()
        self.header = UnitsHeaderView(QtCore.Qt.Orientation.Horizontal)
        self.header.setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.setModel(self.isotope_model)
        self.setHorizontalHeader(self.header)
        self.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents
        )

        for col, name in self.isotope_model.COLUMNS.items():
            delegate = ValueWidgetDelegate()
            if name == "Mass Fraction":
                delegate.max = 1.0
                delegate.step = 0.1
            self.setItemDelegateForColumn(col, delegate)
            if name in ["Diameter", "Concentration", "Mass Response"]:
                self.hideColumn(col)

    def isotope(self, row: int) -> SPCalIsotope:
        return self.isotope_model.data(
            self.isotope_model.index(row, 0), role=IsotopeOptionModel.IsotopeRole
        )

    def rowForIsotope(self, isotope: SPCalIsotope) -> int:
        return list(self.isotope_model.isotope_options.keys()).index(isotope)

    def dialogParticleDatabase(self, index: QtCore.QModelIndex) -> QtWidgets.QDialog:
        def set_density(density: float | None):
            if density is not None:
                density /= 1000.0  # g/cm3 -> kg/m3
            self.isotope_model.setData(index, density, UnitsModel.BaseValueRole)

        dlg = ParticleDatabaseDialog(parent=self)
        dlg.densitySelected.connect(set_density)
        dlg.open()
        return dlg

    def dialogMassFractionCalculator(
        self, index: QtCore.QModelIndex
    ) -> QtWidgets.QDialog:
        def set_major_ratio(ratios: list):
            self.isotope_model.setData(
                index, float(ratios[0][1]), UnitsModel.BaseValueRole
            )

        dlg = MassFractionCalculatorDialog(parent=self)
        dlg.ratiosSelected.connect(set_major_ratio)
        dlg.open()
        return dlg

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent):
        event.accept()
        menu = self.basicTableMenu()

        index = self.indexAt(event.pos())
        if index.isValid() and index.column() == 0:
            action_density = QtGui.QAction(
                QtGui.QIcon.fromTheme("folder-database"), "Lookup Density", self
            )
            action_density.triggered.connect(lambda: self.dialogParticleDatabase(index))
            menu.insertSeparator(menu.actions()[0])
            menu.insertAction(menu.actions()[0], action_density)
        elif index.isValid() and index.column() == 2:
            action_massfrac = QtGui.QAction(
                QtGui.QIcon.fromTheme("folder-calculate"),
                "Calculate Mass Fraction",
                self,
            )
            action_massfrac.triggered.connect(
                lambda: self.dialogMassFractionCalculator(index)
            )
            menu.insertSeparator(menu.actions()[0])
            menu.insertAction(menu.actions()[0], action_massfrac)

        menu.popup(event.globalPos())


class SPCalIsotopeOptionsDock(QtWidgets.QDockWidget):
    optionChanged = QtCore.Signal(SPCalIsotope)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Isotope Options")

        self.table = IsotopeOptionTable()
        self.table.isotope_model.dataChanged.connect(self.onDataChanged)

        self.setWidget(self.table)

    def onDataChanged(self, index: QtCore.QModelIndex):
        self.optionChanged.emit(
            self.table.isotope_model.data(index, IsotopeOptionModel.IsotopeRole)
        )

    def isotopeOptions(self) -> dict[SPCalIsotope, SPCalIsotopeOptions]:
        return self.table.isotope_model.isotope_options

    def setIsotopes(self, isotopes: list[SPCalIsotope]):
        self.table.isotope_model.dataChanged.disconnect(self.onDataChanged)
        self.table.isotope_model.beginResetModel()
        self.table.isotope_model.isotope_options = {
            isotope: SPCalIsotopeOptions(None, None, None) for isotope in isotopes
        }
        self.table.isotope_model.endResetModel()
        self.table.isotope_model.dataChanged.connect(self.onDataChanged)

    def setIsotopeOption(self, isotope: SPCalIsotope, option: SPCalIsotopeOptions):
        self.table.isotope_model.setData(
            self.table.isotope_model.index(self.table.rowForIsotope(isotope), 0),
            option,
            role=IsotopeOptionModel.IsotopeOptionRole,
        )

    def optionForIsotope(self, isotope: SPCalIsotope) -> SPCalIsotopeOptions:
        return self.table.isotope_model.isotope_options[isotope]

    def resetInputs(self):
        self.setIsotopes(list(self.table.isotope_model.isotope_options.keys()))
