from typing import Any
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.calc import mode as modefn
from spcal.gui.modelviews.basic import BasicTableView
from spcal.gui.modelviews.units import UnitsModel, UnitsHeaderView
from spcal.gui.modelviews.values import ValueWidgetDelegate
from spcal.gui.util import create_action
from spcal.isotope import SPCalIsotopeBase
from spcal.processing import SPCalProcessingResult
from spcal.siunits import (
    mass_concentration_units,
    mass_units,
    number_concentration_units,
    signal_units,
    size_units,
    volume_units,
)
import bottleneck as bn


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

    IsotopeRole = QtCore.Qt.ItemDataRole.UserRole
    # NumberRole = QtCore.Qt.ItemDataRole.UserRole + 1
    # ConcentrationRole = QtCore.Qt.ItemDataRole.UserRole + 2
    # BackgroundRole = QtCore.Qt.ItemDataRole.UserRole + 3
    # LODRole = QtCore.Qt.ItemDataRole.UserRole + 4
    # MeanRole = QtCore.Qt.ItemDataRole.UserRole + 5
    # MedianRole = QtCore.Qt.ItemDataRole.UserRole + 6
    # ModeRole = QtCore.Qt.ItemDataRole.UserRole + 7

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
        if role == ResultOutputModel.IsotopeRole:
            return isotope
        elif role == UnitsModel.BaseValueRole:
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
                return result.method.calibrateTo(float(val), self.key, isotope)
        elif role == UnitsModel.BaseErrorRole:
            if name == "Number":
                return result.number_error
            elif name in ["Background", "Mean"]:
                if not result.canCalibrate(self.key):
                    return None
                if name == "Background":
                    val = result.background_error
                else:
                    val = np.std(result.calibrated("signal"))
                return result.method.calibrateTo(float(val), self.key, isotope)
            return None
        elif role == QtCore.Qt.ItemDataRole.ToolTipRole:
            if name == "LOD":
                return f"{result.limit.name}: " + ", ".join(
                    f"{k}={v:.4g}" for k, v in result.limit.parameters.items()
                )
        else:
            return super().data(index, role)


class ResultOutputView(BasicTableView):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.header = UnitsHeaderView(QtCore.Qt.Orientation.Horizontal)
        self.header.setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.setHorizontalHeader(self.header)

        self.setEditTriggers(QtWidgets.QTableView.EditTrigger.NoEditTriggers)
        self.setItemDelegate(ValueWidgetDelegate())

        self.action_sum = create_action("Sum", "Sum", "Sum", self.sumSelectedIsotopes)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent):
        event.accept()
        menu = self.basicTableMenu()
        menu.popup(event.globalPos())

    def sumSelectedIsotopes(self):
        selected_rows = []
        for idx in self.selectedIndexes():
            if idx.row() not in selected_rows:
                selected_rows.append(idx.row())



class SPCalOutputsDock(QtWidgets.QDockWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Results")

        self.model = ResultOutputModel()
        self.table = ResultOutputView()
        self.table.setModel(self.model)

        self.setWidget(self.table)

    def setResults(self, results: dict[SPCalIsotopeBase, SPCalProcessingResult]):
        self.model.beginResetModel()
        self.model.results = results
        self.model.endResetModel()

    def updateOutputsForKey(self, key: str):
        self.model.key = key
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
        self.model.setHeaderData(1, orientation, conc_units, role=UnitsModel.UnitsRole)
        self.model.setHeaderData(
            1, orientation, default_conc_unit, role=UnitsModel.CurrentUnitRole
        )

        for i in range(2, 6):
            self.model.setHeaderData(i, orientation, units, role=UnitsModel.UnitsRole)
            self.model.setHeaderData(
                i, orientation, default_unit, role=UnitsModel.CurrentUnitRole
            )

    def setSignificantFigures(self, sf: int):
        delegate = self.table.itemDelegate()
        assert isinstance(delegate, ValueWidgetDelegate)
        delegate.setSigFigs(sf)
        self.table.setItemDelegate(delegate)

    def reset(self):
        self.model.beginResetModel()
        self.model.results.clear()
        self.model.endResetModel()
