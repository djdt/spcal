import logging

from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.dialogs.tools import MassFractionCalculatorDialog, ParticleDatabaseDialog
from spcal.gui.widgets.unitstable import UnitsTable
from spcal.processing import SPCalIsotopeOptions
from spcal.siunits import (
    density_units,
    response_units,
)

logger = logging.getLogger(__name__)


class IsotopeOptionTable(UnitsTable):
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

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
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
    optionChanged = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Isotope Options")

        self.table = IsotopeOptionTable()
        self.table.setSelectionMode(
            QtWidgets.QTableView.SelectionMode.ExtendedSelection
        )
        self.table.setSelectionBehavior(
            QtWidgets.QTableView.SelectionBehavior.SelectRows
        )
        self.table.cellChanged.connect(self.findOptionChanged)

        self.setWidget(self.table)

    def findOptionChanged(self, row: int, column: int):
        self.optionChanged.emit(self.table.verticalHeaderItem(row).text())

    def asIsotopeOptions(self) -> dict[str, SPCalIsotopeOptions]:
        options = {}
        for row in range(self.table.rowCount()):
            label = self.table.verticalHeaderItem(row).text()
            options[label] = SPCalIsotopeOptions(
                self.table.baseValueForItem(row, 0),
                self.table.baseValueForItem(row, 1),
                self.table.baseValueForItem(row, 2),
            )
        return options

    def setIsotopes(self, isotopes: list[str]) -> None:
        self.table.setVerticalHeaderLabels(isotopes)

    def setIsotopeOption(self, isotope: str, option: SPCalIsotopeOptions):
        for i in range(self.table.rowCount()):
            if self.table.verticalHeaderItem(i).text() == isotope:
                self.table.setBaseValueForItem(i, 0, option.density)
                self.table.setBaseValueForItem(i, 1, option.response)
                self.table.setBaseValueForItem(i, 2, option.mass_fraction)
                return
        raise StopIteration

    def optionForIsotope(self, isotope: str) -> SPCalIsotopeOptions:
        for i in range(self.table.rowCount()):
            if self.table.verticalHeaderItem(i).text() == isotope:
                return SPCalIsotopeOptions(
                    density=self.table.baseValueForItem(i, 0),
                    response=self.table.baseValueForItem(i, 1),
                    mass_fraction=self.table.baseValueForItem(i, 2),
                )
        raise StopIteration

    def selectedIsotope(self) -> str:
        indicies = self.table.selectedIndexes()
        if len(indicies) == 0:
            return ""
        return self.table.verticalHeaderItem(indicies[0].row()).text()


if __name__ == "__main__":
    app = QtWidgets.QApplication()

    dlg = SPCalIsotopeOptionsDock()
    dlg.setIsotopes(["A197"])

    dlg.show()
    app.exec()
