from spcal.datafile import SPCalDataFile
from typing import Any

import bottleneck as bn
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.calc import mode as modefn
from spcal.gui.modelviews import (
    BaseValueErrorRole,
    BaseValueRole,
    IsotopeRole,
    DataFileRole,
)
from spcal.gui.modelviews.basic import BasicTableView
from spcal.gui.modelviews.units import UnitsHeaderView, UnitsModel
from spcal.gui.modelviews.values import ValueWidgetDelegate
from spcal.gui.objects import ContextMenuRedirectFilter
from spcal.gui.util import create_action
from spcal.isotope import SPCalIsotope, SPCalIsotopeBase, SPCalIsotopeExpression
from spcal.processing.result import SPCalProcessingResult
from spcal.siunits import (
    number_concentration_units,
    signal_units,
    mass_concentration_units,
)


class ResultOutputModel(UnitsModel):
    COLUMN_LABELS = {
        0: "Number",
        1: "Concentration",
        2: "Ionic Background",
        3: "Background",
        4: "LOD",
        5: "Mean",
        6: "Median",
        7: "Mode",
    }
    COLUMN_TOOLTIPS = {
        0: "Number of detected particles",
        1: "Particle number or mass concentration",
        2: "Mean concentration of background regions",
        3: "Mean value of all regions without detected particles",
        4: "The limit of detection, different from the detection threshold",
        5: "Mean value of detected particles",
        6: "Median value of detected particles",
        7: "The most frequent value of detected particles",
    }

    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(
            list(ResultOutputModel.COLUMN_LABELS.values()),
            ["", "#/ml", "µg/L", "cts", "cts", "cts", "cts", "cts"],
            [
                {},
                number_concentration_units,
                mass_concentration_units,
                signal_units,
                signal_units,
                signal_units,
                signal_units,
                signal_units,
            ],
            units_tooltips=list(ResultOutputModel.COLUMN_TOOLTIPS.values()),
            parent=parent,
        )

        self.key = "signal"

        self.results: dict[
            SPCalDataFile, dict[SPCalIsotopeBase, SPCalProcessingResult]
        ] = {}

    def resultForSection(
        self, section: int
    ) -> tuple[SPCalDataFile, SPCalIsotopeBase, SPCalProcessingResult]:
        current_section = 0
        for data_file, results in self.results.items():
            for isotope, result in results.items():
                if current_section == section:
                    return data_file, isotope, result
                current_section += 1
        raise StopIteration(f"unable to find result for section {section}")

    def rowCount(
        self,
        parent: QtCore.QModelIndex
        | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),
    ) -> int:
        if parent.isValid():
            return 0
        return sum(len(results) for results in self.results.values())

    def columnCount(
        self,
        parent: QtCore.QModelIndex
        | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),
    ) -> int:
        if parent.isValid():
            return 0
        return len(ResultOutputModel.COLUMN_LABELS)

    def flags(
        self, index: QtCore.QModelIndex | QtCore.QPersistentModelIndex
    ) -> QtCore.Qt.ItemFlag:
        flags = super().flags(index)
        if index.isValid():
            flags ^= QtCore.Qt.ItemFlag.ItemIsEditable
        return flags

    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if section < 0:
            return
        if orientation == QtCore.Qt.Orientation.Vertical:
            data_file, isotope, _ = self.resultForSection(section)
            if role in [
                QtCore.Qt.ItemDataRole.DisplayRole,
                QtCore.Qt.ItemDataRole.EditRole,
            ]:
                if len(self.results) < 2:
                    return f"{isotope}"
                else:
                    return f"{data_file.path.stem} : {isotope}"
            elif role == DataFileRole:
                return data_file
            elif role == IsotopeRole:
                return isotope
        return super().headerData(section, orientation, role)

    def data(
        self,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if not index.isValid():
            return None

        name = ResultOutputModel.COLUMN_LABELS[index.column()]
        data_file, isotope, result = self.resultForSection(index.row())
        if role == IsotopeRole:
            return isotope
        elif role == DataFileRole:
            return data_file
        elif role == BaseValueRole:
            if name == "Number":
                return result.number
            elif name == "Concentration":
                if self.key == "mass":
                    return result.mass_concentration
                else:
                    return result.number_concentration
            elif name == "Ionic Background":
                return result.ionic_background
            else:  # other column
                if not result.canCalibrate(self.key) or result.number == 0:
                    return None
                if name == "Background":
                    return result.calibrateTo(result.background, self.key)
                elif name == "LOD":
                    lod = bn.nanmean(result.limit.limitOfDetection())
                    return result.calibrateTo(float(lod), self.key)
                elif name == "Mean":
                    return np.mean(result.calibrated(self.key))
                elif name == "Median":
                    return np.median(result.calibrated(self.key))
                elif name == "Mode":
                    return modefn(result.calibrated(self.key))
                else:
                    raise ValueError(f"unknown column name {name}")
        elif role == BaseValueErrorRole:
            if name == "Number":
                return result.number_error if result.number_error > 0.0 else None
            elif name in ["Background", "Mean"]:
                if not result.canCalibrate(self.key) or result.number == 0:
                    return None
                if name == "Background":
                    return result.calibrateTo(float(result.background_error), self.key)
                else:
                    return np.std(result.calibrated(self.key))
            return None
        elif role == QtCore.Qt.ItemDataRole.ToolTipRole:
            if name == "LOD":
                return f"{result.limit.name}: " + ", ".join(
                    f"{k}={v}" for k, v in result.limit.parameters.items()
                )
        else:
            return super().data(index, role)


class ResultOutputView(BasicTableView):
    currentRowChanged = QtCore.Signal(SPCalIsotopeBase)
    selectedRowsChanged = QtCore.Signal(list)

    requestAddExpression = QtCore.Signal(SPCalIsotopeExpression)
    requestRemoveIsotopes = QtCore.Signal(list)
    requestRemoveExpressions = QtCore.Signal(list)
    requestRemoveResult = QtCore.Signal(SPCalProcessingResult)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)

        self._previous_rows = []

        self.header = UnitsHeaderView(QtCore.Qt.Orientation.Horizontal)
        self.header.setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )

        self.setHorizontalHeader(self.header)
        self.verticalHeader().installEventFilter(ContextMenuRedirectFilter(self))
        self.verticalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Fixed
        )

        self.setEditTriggers(QtWidgets.QTableView.EditTrigger.NoEditTriggers)
        self.setItemDelegate(ValueWidgetDelegate())

        self.action_sum = create_action(
            "black_sum",
            "Sum Isotopes",
            "Add a calculator expression summing the selected isotopes.",
            self.sumSelectedIsotopes,
        )
        self.action_remove_isotopes = create_action(
            "entry-delete",
            "Remove Isotopes",
            "Delete the selected isotopes and expressions.",
            self.removeSelectedIsotopes,
        )
        self.action_remove_expr = create_action(
            "entry-delete",
            "Remove Expressions",
            "Delete selected expressions.",
            self.removeSelectedExpressions,
        )

    def setModel(self, model: QtCore.QAbstractItemModel | None):
        if not isinstance(model, ResultOutputModel):
            raise ValueError("ResultOutputView requires a ResultOutputModel")
        super().setModel(model)
        # hook up signals
        self.selectionModel().currentChanged.connect(self.onCurrentChanged)
        self.selectionModel().selectionChanged.connect(self.onSelectionChanged)

    def currentRow(self) -> int | None:
        current = self.currentIndex()
        if current.isValid():
            return current.row()
        return None

    def setCurrentRow(self, row: int):
        self.selectionModel().setCurrentIndex(
            self.model().index(row, 0),
            QtCore.QItemSelectionModel.SelectionFlag.Current,
        )

    def onCurrentChanged(self, index: QtCore.QModelIndex):
        print("current changed")
        self.currentRowChanged.emit(index.row())

    def selectedRows(self) -> list[int]:
        return list(set(idx.row() for idx in self.selectedIndexes()))

    def setSelectedRows(self, rows: list[int]):
        selection = QtCore.QItemSelection()
        for row in rows:
            index = self.model().index(row, 0)
            selection.select(index, index)
        self.selectionModel().select(  # TODO: could be improved to retain selected columns
            selection,
            QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect
            | QtCore.QItemSelectionModel.SelectionFlag.Rows,
        )

    def onSelectionChanged(self):
        selected = self.selectedRows()
        if selected != self._previous_rows:
            self._previous_rows = selected
            self.selectedRowsChanged.emit(selected)

    def selectedIsotopes(self) -> list[SPCalIsotopeBase]:
        return list(
            set(
                self.model().index(row, 0).data(IsotopeRole)
                for row in self.selectedRows()
            )
        )

    def setSelectedIsotopes(self, isotopes: list[SPCalIsotopeBase]):
        rows = []
        for row in range(self.model().rowCount()):
            if self.model().index(row, 0).data(IsotopeRole) in isotopes:
                rows.append(row)
        self.setSelectedRows(rows)
        if len(rows) > 0:
            self.setCurrentRow(rows[0])

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent):
        event.accept()
        menu = self.basicTableMenu()
        selected = self.selectedRows()
        menu.addSeparator()
        if len(selected) > 1 and all(isinstance(iso, SPCalIsotope) for iso in selected):
            menu.addAction(self.action_sum)
        if any(isinstance(iso, SPCalIsotopeExpression) for iso in selected):
            menu.addAction(self.action_remove_expr)
        if any(isinstance(iso, SPCalIsotope) for iso in selected):
            menu.addAction(self.action_remove_isotopes)
        menu.popup(event.globalPos())

    def sumSelectedIsotopes(self):
        selected_isotopes = [
            iso for iso in self.selectedIsotopes() if isinstance(iso, SPCalIsotope)
        ]
        if len(selected_isotopes) > 1:
            expr = SPCalIsotopeExpression.sumIsotopes(selected_isotopes)
            self.requestAddExpression.emit(expr)

    def removeSelectedIsotopes(self):
        selected_expr = [
            iso for iso in self.selectedIsotopes() if isinstance(iso, SPCalIsotope)
        ]
        if len(selected_expr) > 0:
            self.requestRemoveIsotopes.emit(selected_expr)

    def removeSelectedExpressions(self):
        selected_expr = [
            iso
            for iso in self.selectedIsotopes()
            if isinstance(iso, SPCalIsotopeExpression)
        ]
        if len(selected_expr) > 0:
            self.requestRemoveExpressions.emit(selected_expr)
