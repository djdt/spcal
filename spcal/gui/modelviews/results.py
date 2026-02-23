from typing import Any

import bottleneck as bn
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.calc import mode as modefn
from spcal.gui.modelviews import (
    BaseValueErrorRole,
    BaseValueRole,
    IsotopeRole,
)
from spcal.gui.modelviews.basic import BasicTableView
from spcal.gui.modelviews.units import UnitsHeaderView, UnitsModel
from spcal.gui.modelviews.values import ValueWidgetDelegate
from spcal.gui.objects import ContextMenuRedirectFilter
from spcal.gui.util import create_action
from spcal.isotope import SPCalIsotope, SPCalIsotopeBase, SPCalIsotopeExpression
from spcal.processing.result import SPCalProcessingResult
from spcal.siunits import number_concentration_units, signal_units


class ResultOutputModel(UnitsModel):
    COLUMNS = {
        0: "Number",
        1: "Concentration",
        2: "Background",
        3: "LOD",
        4: "Mean",
        5: "Median",
        6: "Mode",
    }

    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(
            list(ResultOutputModel.COLUMNS.values()),
            ["", "#/L", "cts", "cts", "cts", "cts", "cts"],
            [
                {},
                number_concentration_units,
                signal_units,
                signal_units,
                signal_units,
                signal_units,
                signal_units,
            ],
            parent=parent,
        )

        self.key = "signal"

        self.results: list[SPCalProcessingResult] = []

    def rowCount(
        self,
        parent: QtCore.QModelIndex
        | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),
    ) -> int:
        if parent.isValid():
            return 0
        return len(self.results)

    def columnCount(
        self,
        parent: QtCore.QModelIndex
        | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),
    ) -> int:
        if parent.isValid():
            return 0
        return len(ResultOutputModel.COLUMNS)

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
        if orientation == QtCore.Qt.Orientation.Vertical:
            if role in [
                QtCore.Qt.ItemDataRole.DisplayRole,
                QtCore.Qt.ItemDataRole.EditRole,
            ]:
                return str(self.results[section].isotope)
            elif role == IsotopeRole:
                return self.results[section].isotope
        return super().headerData(section, orientation, role)

    def data(
        self,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if not index.isValid():
            return None

        name = ResultOutputModel.COLUMNS[index.column()]
        result = self.results[index.row()]
        if role == IsotopeRole:
            return result.isotope
        elif role == BaseValueRole:
            if name == "Number":
                return result.number
            elif name == "Concentration":
                if self.key == "mass":
                    return result.mass_concentration
                else:
                    return result.number_concentration
            else:  # other column
                if not result.canCalibrate(self.key):
                    return None
                if name == "Background":
                    return result.calibrateTo(result.background, self.key)
                elif name == "LOD":
                    lod = bn.nanmean(result.limit.detection_threshold)
                    return result.calibrateTo(float(lod), self.key)
                elif name == "Mean":
                    return np.mean(result.calibrated(self.key))
                elif name == "Median":
                    return np.median(result.calibrated(self.key))
                elif name == "Mode":
                    return modefn(result.calibrated(self.key))
                else:
                    raise ValueError(f"unknown column name {name}")
                return
        elif role == BaseValueErrorRole:
            if name == "Number":
                return result.number_error
            elif name in ["Background", "Mean"]:
                if not result.canCalibrate(self.key):
                    return None
                if name == "Background":
                    return result.calibrateTo(float(result.background_error), self.key)
                else:
                    return np.std(result.calibrated(self.key))
            return None
        elif role == QtCore.Qt.ItemDataRole.ToolTipRole:
            if name == "LOD":
                return f"{result.limit.name}: " + ", ".join(
                    f"{k}={v:.4g}" for k, v in result.limit.parameters.items()
                )
        else:
            return super().data(index, role)


class ResultOutputView(BasicTableView):
    isotopeSelected = QtCore.Signal(SPCalIsotopeBase)
    requestAddExpression = QtCore.Signal(SPCalIsotopeExpression)
    requestRemoveIsotopes = QtCore.Signal(list)
    requestRemoveExpressions = QtCore.Signal(list)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.results_model = ResultOutputModel()

        self.header = UnitsHeaderView(QtCore.Qt.Orientation.Horizontal)
        self.header.setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )

        self.setModel(self.results_model)
        self.setHorizontalHeader(self.header)
        self.verticalHeader().installEventFilter(ContextMenuRedirectFilter(self))
        self.verticalHeader().sectionClicked.connect(self.onHeaderClicked)

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

    def onHeaderClicked(self, section: int):
        isotope = self.results_model.data(
            self.results_model.index(section, 0), IsotopeRole
        )
        self.isotopeSelected.emit(isotope)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent):
        event.accept()
        menu = self.basicTableMenu()
        selected = self.selectedIsotopes()
        menu.addSeparator()
        if len(selected) > 1 and all(isinstance(iso, SPCalIsotope) for iso in selected):
            menu.addAction(self.action_sum)
        if any(isinstance(iso, SPCalIsotopeExpression) for iso in selected):
            menu.addAction(self.action_remove_expr)
        if any(isinstance(iso, SPCalIsotope) for iso in selected):
            menu.addAction(self.action_remove_isotopes)
        menu.popup(event.globalPos())

    def selectedIsotopes(self):
        selected_isotopes = []
        for idx in self.selectedIndexes():
            isotope = idx.data(IsotopeRole)
            if isotope not in selected_isotopes:
                selected_isotopes.append(isotope)
        return selected_isotopes

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
