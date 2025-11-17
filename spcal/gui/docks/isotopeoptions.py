import logging
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.dialogs.tools import MassFractionCalculatorDialog, ParticleDatabaseDialog
from spcal.gui.modelviews.headers import ComboHeaderView
from spcal.gui.modelviews.units import UnitsTable, UnitsTableView
from spcal.isotope import SPCalIsotope
from spcal.processing import SPCalIsotopeOptions
from spcal.siunits import (
    density_units,
    response_units,
)

logger = logging.getLogger(__name__)


class IsotopeOptionModel(QtCore.QAbstractTableModel):
    COLUMNS = {
        0: "density",
        1: "response",
        2: "mass_fraction",
        3: "diameter",
        4: "concentration",
        5: "mass_response",
    }
    LABELS = [
        "Density",
        "Response",
        "Mass Fraction",
        "Diameter",
        "Concentration",
        "Mass Response",
    ]

    IsotopeRole = QtCore.Qt.ItemDataRole.UserRole
    IsotopeOptionRole = QtCore.Qt.ItemDataRole.UserRole + 1

    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(parent)
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
        if IsotopeOptionModel.COLUMNS[index.column()] != "mass_response":
            flags ^= QtCore.Qt.ItemFlag.ItemIsEditable
        return flags

    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        print(role)
        if role != QtCore.Qt.ItemDataRole.DisplayRole:
            return super().headerData(section, orientation, role)
        elif orientation == QtCore.Qt.Orientation.Horizontal:
            return IsotopeOptionModel.LABELS[section]
        else:
            return str(list(self.isotope_options.keys())[section])

    def data(
        self,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if not index.isValid():
            return None

        isotope = list(self.isotope_options.keys())[index.row()]
        col = index.column()
        if role == IsotopeOptionModel.IsotopeRole:
            return isotope
        elif role == IsotopeOptionModel.IsotopeOptionRole:
            return self.isotope_options[isotope]
        elif role in [
            QtCore.Qt.ItemDataRole.DisplayRole,
            QtCore.Qt.ItemDataRole.EditRole,
        ]:
            return getattr(
                self.isotope_options[isotope], IsotopeOptionModel.COLUMNS[col]
            )
        else:
            return None

    def setData(
        self,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        value: Any,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> bool:
        if not index.isValid():
            return False

        isotope = list(self.isotope_options.keys())[index.row()]
        col = index.column()
        if role == IsotopeOptionModel.IsotopeRole:
            return False  # cannot set isotope
        elif role == IsotopeOptionModel.IsotopeOptionRole:
            self.isotope_options[isotope] = value
            tl, br = (
                self.index(index.row(), 0),
                self.index(index.row(), self.columnCount()),
            )
            self.dataChanged.emit(tl, br, [role])
            return True
        elif role in [
            QtCore.Qt.ItemDataRole.DisplayRole,
            QtCore.Qt.ItemDataRole.EditRole,
        ]:
            setattr(
                self.isotope_options[isotope], IsotopeOptionModel.COLUMNS[col], value
            )
            self.dataChanged.emit(index, index, [role])
            return True
        else:
            return False


class IsotopeOptionTable(UnitsTableView):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(
            [
                ("Density", density_units, "g/cm³", None),
                ("Response", response_units, "L/µg", None),
                ("Mass Fraction", None, None, (0.0, 1.0)),
            ],
            parent=parent,
        )

    def dialogParticleDatabase(self, index: QtCore.QModelIndex) -> QtWidgets.QDialog:
        dlg = ParticleDatabaseDialog(parent=self)
        dlg.densitySelected.connect(
            lambda x: self.model().setData(
                index,
                x * 1000.0 / self.current_units[index.column()],
                QtCore.Qt.ItemDataRole.EditRole,
            )
        )  # to current unit
        dlg.open()
        return dlg

    def dialogMassFractionCalculator(
        self, index: QtCore.QModelIndex
    ) -> QtWidgets.QDialog:
        def set_major_ratio(ratios: list):
            self.model().setData(
                index, float(ratios[0][1]), QtCore.Qt.ItemDataRole.EditRole
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
        self.model = IsotopeOptionModel()
        self.table.setModel(self.model)
        self.table.setHorizontalHeader(
            ComboHeaderView([("Density", density_units, "g/cm³", None)])
        )
        # self.table.setSelectionMode(
        #     QtWidgets.QTableView.SelectionMode.ExtendedSelection
        # )
        # self.table.setSelectionBehavior(
        #     QtWidgets.QTableView.SelectionBehavior.SelectRows
        # )
        # self.table.cellChanged.connect(
        #     lambda r, c: self.optionChanged.emit(self.isotope(r))
        # )

        self.setWidget(self.table)

    def isotope(self, row: int) -> SPCalIsotope:
        return self.model.data(
            self.model.index(row, 0), role=IsotopeOptionModel.IsotopeRole
        )

    def rowForIsotope(self, isotope: SPCalIsotope) -> int:
        return list(self.model.isotope_options.keys()).index(isotope)

    def asIsotopeOptions(self) -> dict[SPCalIsotope, SPCalIsotopeOptions]:
        # options = {}
        return self.model.isotope_options
        # for row in range(self.table.model().rowCount()):
        #     options[self.isotope(row)] = SPCalIsotopeOptions(
        #         self.table.baseValueForItem(row, 0),
        #         self.table.baseValueForItem(row, 1),
        #         self.table.baseValueForItem(row, 2),
        #     )
        # return options

    def setIsotopes(self, isotopes: list[SPCalIsotope]):
        self.table.blockSignals(True)
        self.model.beginResetModel()
        self.model.isotope_options = {
            isotope: SPCalIsotopeOptions(None, None, None) for isotope in isotopes
        }
        self.model.endResetModel()
        # self.table.setRowCount(len(isotopes))
        # for i, iso in enumerate(isotopes):
        #     item = QtWidgets.QTableWidgetItem(
        #         str(iso), type=QtWidgets.QTableWidgetItem.ItemType.UserType
        #     )
        #     item.setData(QtCore.Qt.ItemDataRole.UserRole, iso)
        #     self.table.setVerticalHeaderItem(i, item)
        self.table.blockSignals(False)

    def setIsotopeOption(self, isotope: SPCalIsotope, option: SPCalIsotopeOptions):
        self.model.setData(
            self.model.index(self.rowForIsotope(isotope), 0),
            option,
            role=IsotopeOptionModel.IsotopeOptionRole,
        )
        # for i in range(self.table.rowCount()):
        #     if self.isotope(i) == isotope:
        #         self.table.blockSignals(True)
        #         self.table.setBaseValueForItem(i, 0, option.density)
        #         self.table.setBaseValueForItem(i, 1, option.response)
        #         self.table.setBaseValueForItem(i, 2, option.mass_fraction)
        #         self.table.blockSignals(False)
        #         return
        # raise StopIteration

    def optionForIsotope(self, isotope: SPCalIsotope) -> SPCalIsotopeOptions:
        return self.model.isotope_options[isotope]
        # for i in range(self.table.rowCount()):
        #     if self.isotope(i) == isotope:
        #         return SPCalIsotopeOptions(
        #             density=self.table.baseValueForItem(i, 0),
        #             response=self.table.baseValueForItem(i, 1),
        #             mass_fraction=self.table.baseValueForItem(i, 2),
        #         )
        # raise StopIteration

    def resetInputs(self):
        self.blockSignals(True)
        self.setIsotopes(list(self.model.isotope_options.keys()))
        self.blockSignals(False)
