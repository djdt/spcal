from typing import Any

import bottleneck as bn
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.calc import mode as modefn
from spcal.gui.modelviews import (
    BaseValueErrorRole,
    BaseValueRole,
    CurrentUnitRole,
    IsotopeRole,
    UnitsRole,
)
from spcal.gui.modelviews.basic import BasicTableView
from spcal.gui.modelviews.units import UnitsHeaderView, UnitsModel
from spcal.gui.modelviews.values import ValueWidgetDelegate
from spcal.gui.objects import ContextMenuRedirectFilter
from spcal.gui.util import create_action
from spcal.isotope import SPCalIsotope, SPCalIsotopeBase, SPCalIsotopeExpression
from spcal.processing import SPCalProcessingResult
from spcal.siunits import (
    mass_concentration_units,
    mass_units,
    number_concentration_units,
    signal_units,
    size_units,
    volume_units,
)


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
        super().__init__(parent=parent)

        self.unit_labels = list(ResultOutputModel.COLUMNS.values())
        self.current_unit = ["", "#/L", "cts", "cts", "cts", "cts", "cts"]
        self.units = [
            {},
            number_concentration_units,
            signal_units,
            signal_units,
            signal_units,
            signal_units,
            signal_units,
        ]
        self.key = "signal"

        self.results: dict[SPCalIsotopeBase, SPCalProcessingResult] = {}

    def rowCount(
        self,
        parent: QtCore.QModelIndex
        | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),
    ) -> int:
        return len(self.results)

    def columnCount(
        self,
        parent: QtCore.QModelIndex
        | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),
    ) -> int:
        return len(ResultOutputModel.COLUMNS)

    def flags(
        self, index: QtCore.QModelIndex | QtCore.QPersistentModelIndex
    ) -> QtCore.Qt.ItemFlag:
        flags = super().flags(index)
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
                return str(list(self.results.keys())[section])
            elif role == IsotopeRole:
                return list(self.results.keys())[section]
        return super().headerData(section, orientation, role)

    def data(
        self,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if not index.isValid():
            return None

        isotope = list(self.results.keys())[index.row()]
        name = ResultOutputModel.COLUMNS[index.column()]
        result = self.results[isotope]
        if role == IsotopeRole:
            return isotope
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
                    val = result.background
                elif name == "LOD":
                    val = bn.nanmean(result.limit.detection_threshold)
                elif name == "Mean":
                    val = np.mean(result.calibrated("signal"))
                elif name == "Median":
                    val = np.median(result.calibrated("signal"))
                elif name == "Mode":
                    val = modefn(result.calibrated("signal"))
                else:
                    raise ValueError(f"unknown column name {name}")
                return result.method.calibrateTo(
                    float(val), self.key, isotope, result.event_time
                )
        elif role == BaseValueErrorRole:
            if name == "Number":
                return result.number_error
            elif name in ["Background", "Mean"]:
                if not result.canCalibrate(self.key):
                    return None
                if name == "Background":
                    val = result.background_error
                else:
                    val = np.std(result.calibrated("signal"))
                return result.method.calibrateTo(
                    float(val), self.key, isotope, result.event_time
                )
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


class SPCalOutputsDock(QtWidgets.QDockWidget):
    requestCurrentIsotope = QtCore.Signal(SPCalIsotopeBase)
    requestRemoveIsotopes = QtCore.Signal(list)
    requestAddExpression = QtCore.Signal(SPCalIsotopeExpression)
    requestRemoveExpressions = QtCore.Signal(list)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Results")

        self.table = ResultOutputView()

        self.table.isotopeSelected.connect(self.requestCurrentIsotope)
        self.table.requestRemoveIsotopes.connect(self.requestRemoveIsotopes)
        self.table.requestAddExpression.connect(self.requestAddExpression)
        self.table.requestRemoveExpressions.connect(self.requestRemoveExpressions)

        self.setWidget(self.table)

    def setResults(self, results: dict[SPCalIsotopeBase, SPCalProcessingResult]):
        self.table.results_model.beginResetModel()
        self.table.results_model.results = results
        self.table.results_model.endResetModel()

    def updateOutputsForKey(self, key: str):
        self.table.results_model.key = key
        units = signal_units
        default_unit = "cts"
        conc_units = number_concentration_units
        default_conc_unit = "#/L"

        if key == "mass":
            units = mass_units
            default_unit = "fg"
            default_conc_unit = "µg/L"
            conc_units = mass_concentration_units
        elif key == "size":
            units = size_units
            default_unit = "nm"
        elif key == "volume":
            units = volume_units
            default_unit = "µm³"
        elif key != "signal":
            raise ValueError(f"unknown key '{key}'")

        orientation = self.table.header.orientation()
        self.table.results_model.setHeaderData(
            1, orientation, conc_units, role=UnitsRole
        )
        self.table.results_model.setHeaderData(
            1, orientation, default_conc_unit, role=CurrentUnitRole
        )

        for i in range(2, 6):
            self.table.results_model.setHeaderData(
                i, orientation, units, role=UnitsRole
            )
            self.table.results_model.setHeaderData(
                i, orientation, default_unit, role=CurrentUnitRole
            )

    def setSignificantFigures(self, sf: int):
        delegate = self.table.itemDelegate()
        assert isinstance(delegate, ValueWidgetDelegate)
        delegate.setSigFigs(sf)
        self.table.setItemDelegate(delegate)

    def reset(self):
        self.setResults({})
