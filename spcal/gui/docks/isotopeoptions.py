import logging
import re

from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.dialogs.tools import MassFractionCalculatorDialog, ParticleDatabaseDialog
from spcal.gui.modelviews import BasicTable, ComboHeaderView
from spcal.gui.widgets.values import ValueWidget
from spcal.processing import SPCalIsotopeOptions
from spcal.siunits import (
    density_units,
    response_units,
)

logger = logging.getLogger(__name__)


class ValueWidgetDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(
        self,
        sigfigs: int = 6,
        min: float = 0.0,
        max: float = 1e99,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent=parent)
        self.sigfigs = sigfigs
        self.min, self.max = min, max

    def createEditor(
        self,
        parent: QtWidgets.QWidget,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ) -> QtWidgets.QWidget:
        editor = ValueWidget(
            min=self.min, max=self.max, sigfigs=self.sigfigs, parent=parent
        )
        return editor

    def setEditorData(
        self,
        editor: QtWidgets.QWidget,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ):
        assert isinstance(editor, ValueWidget)
        value = index.data(QtCore.Qt.ItemDataRole.EditRole)
        editor.setValue(value)

    def setModelData(
        self,
        editor: QtWidgets.QWidget,
        model: QtCore.QAbstractItemModel,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ):
        assert isinstance(editor, ValueWidget)
        value = editor.value()
        model.setData(index, value, QtCore.Qt.ItemDataRole.EditRole)


class IsotopeOptionTable(BasicTable):
    HEADER_UNITS = {
        "Density": density_units,
        "Response": response_units,
        "Mass Fraction": None,
        # "Diameter": size_units,
        # "Concentration": mass_concentration_units,
    }

    def __init__(self, sf: int, parent: QtWidgets.QWidget | None = None):
        super().__init__(0, 3, parent=parent)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.MinimumExpanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

        header_items = {}
        for i, (key, units) in enumerate(self.HEADER_UNITS.items()):
            if units is None:
                continue
            header_items[i] = [f"{key} ({unit})" for unit in units.keys()]

        self.header = ComboHeaderView(header_items)

        self.setItemDelegateForColumn(0, ValueWidgetDelegate(sf, parent=self))
        self.setItemDelegateForColumn(1, ValueWidgetDelegate(sf, parent=self))
        self.setItemDelegateForColumn(2, ValueWidgetDelegate(sf, 0.0, 1.0, parent=self))

        self.current_units = {0: density_units["g/cm³"], 1: response_units["L/µg"]}

        self.header.sectionChanged.connect(self.adjustSectionValues)
        self.header.setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.setHorizontalHeader(self.header)
        self.setHorizontalHeaderLabels(
            ["Density (g/cm³)", "Response (L/µg)", "Mass Fraction"]
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

    def sizeHint(self) -> QtCore.QSize:
        width = sum(
            self.horizontalHeader().sectionSize(i) for i in range(self.columnCount())
        )
        width += self.verticalHeader().width()
        height = super().sizeHint().height()
        return QtCore.QSize(width, height)

    def unitForSection(self, section: int) -> float:
        text = self.model().headerData(section, QtCore.Qt.Orientation.Horizontal)
        m = re.match("([\\w ]+) \\((.+)\\)", text)
        if m is None:
            return 1.0
        return self.HEADER_UNITS[m.group(1)][m.group(2)]

    def adjustSectionValues(self, section: int) -> None:
        if section not in self.current_units:
            return
        current = self.current_units[section]
        new = self.unitForSection(section)
        self.current_units[section] = new

        for row in range(self.rowCount()):
            item = self.item(row, section)
            if item is not None:
                value = (
                    float(item.data(QtCore.Qt.ItemDataRole.EditRole)) * current / new
                )
                item.setData(QtCore.Qt.ItemDataRole.EditRole, value)

    def baseValueForItem(self, row: int, column: int) -> float | None:
        item = self.item(row, column)
        if item is None:
            return None
        value = item.data(QtCore.Qt.ItemDataRole.EditRole)
        if value is not None:
            value = float(item.data(QtCore.Qt.ItemDataRole.EditRole))
            if column in self.current_units:
                value *= self.current_units[column]
        return value

    def setBaseValueForItem(self, row: int, column: int, value: float | None):
        item = self.item(row, column)
        if item is None:
            raise ValueError(f"missing item at ({row}, {column})")
        if value is not None and column in self.current_units:
            value /= self.current_units[column]
        item.setData(QtCore.Qt.ItemDataRole.EditRole, value)

    def asIsotopeOptions(self) -> dict[str, SPCalIsotopeOptions]:
        options = {}
        for row in range(self.rowCount()):
            label = self.verticalHeaderItem(row).text()
            options[label] = SPCalIsotopeOptions(
                self.baseValueForItem(row, 0),
                self.baseValueForItem(row, 1),
                self.baseValueForItem(row, 2),
            )
        return options

    def setIsotopes(self, isotopes: list[str]) -> None:
        self.blockSignals(True)
        self.setRowCount(len(isotopes))
        self.setVerticalHeaderLabels(isotopes)
        for i in range(self.rowCount()):
            for j in range(self.columnCount()):
                item = QtWidgets.QTableWidgetItem()
                item.setData(QtCore.Qt.ItemDataRole.EditRole, None)
                self.setItem(i, j, item)
                self.update(self.indexFromItem(item))
        self.blockSignals(False)

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

        sf = int(QtCore.QSettings().value("SigFigs", 4))  # type: ignore

        self.table = IsotopeOptionTable(sf)
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

    def setIsotopes(self, isotopes: list[str]):
        self.table.setIsotopes(isotopes)

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
    dlg.table.setIsotopes(["A197"])

    dlg.show()
    app.exec()
