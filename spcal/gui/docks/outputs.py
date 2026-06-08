from spcal.datafile import SPCalDataFile
from PySide6 import QtCore, QtWidgets
import logging

from spcal.gui.modelviews import (
    CurrentUnitRole,
    UnitsRole,
    IsotopeRole,
    DataFileRole,
)
from spcal.gui.modelviews.values import ValueWidgetDelegate
from spcal.gui.modelviews.results import ResultOutputView, ResultOutputModel
from spcal.isotope import SPCalIsotopeExpression, SPCalIsotopeBase
from spcal.processing.result import SPCalProcessingResult
from spcal.siunits import (
    mass_concentration_units,
    mass_units,
    number_concentration_units,
    signal_units,
    size_units,
)

logger = logging.getLogger(__name__)


class SPCalOutputsDock(QtWidgets.QDockWidget):
    requestRemoveIsotopes = QtCore.Signal(list)
    requestAddExpression = QtCore.Signal(SPCalIsotopeExpression)
    requestRemoveExpressions = QtCore.Signal(list)

    activeResultsChanged = QtCore.Signal(object)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("spcal-results-dock")
        self.setWindowTitle("Results")

        self._previous_active = []

        self.model = ResultOutputModel()

        self.view = ResultOutputView()
        self.view.setModel(self.model)

        self.view.requestRemoveIsotopes.connect(self.requestRemoveIsotopes)
        self.view.requestAddExpression.connect(self.requestAddExpression)
        self.view.requestRemoveExpressions.connect(self.requestRemoveExpressions)

        self.view.selectedRowsChanged.connect(self.onSelectedRowsChanged)

        self.setWidget(self.view)
        self.updateOutputsForKey("signal")

    def onSelectedRowsChanged(self):
        active = self.activeResults()
        if active != self._previous_active:
            self._previous_active = active
            self.activeResultsChanged.emit(active)

    def activeResults(
        self,
    ) -> dict[SPCalDataFile, dict[SPCalIsotopeBase, SPCalProcessingResult]]:
        """The selected isotopes, or current if no selection exists"""
        rows = self.view.selectedRows()
        if len(rows) == 0:
            current = self.view.currentRow()
            print("the current row is", current)
            rows = [current] if current is not None else []

        results = {}
        for row in rows:
            data_file, isotope, result = self.model.results[row]
            if data_file not in results:
                results[data_file] = {}
            results[data_file][isotope] = result
        return results

    def setActiveResults(
        self,
        results: dict[SPCalDataFile, dict[SPCalIsotopeBase, SPCalProcessingResult]],
    ):
        rows = []
        for row in range(self.model.rowCount()):
            index = self.model.index(row, 0)
            data_file = index.data(DataFileRole)
            if data_file in results and index.data(IsotopeRole) in results[data_file]:
                rows.append(index.row())
        self.view.setSelectedRows(rows)
        if len(rows) > 0:
            self.view.setCurrentRow(rows[0])

    def results(
        self,
    ) -> dict[SPCalDataFile, dict[SPCalIsotopeBase, SPCalProcessingResult]]:
        """Converts from a flat list to a dict"""
        results = {}
        for df, isotope, result in self.model.results:
            if df not in results:
                results[df] = {}
            results[df][isotope] = result
        return results

    def setResults(
        self,
        results: dict[SPCalDataFile, dict[SPCalIsotopeBase, SPCalProcessingResult]],
    ):
        flattened = [
            (df, isotope, result)
            for df in results
            for isotope, result in results[df].items()
        ]
        self.model.beginResetModel()
        self.model.results = flattened
        if len(results) > 1:
            self.model.multiple_datafiles = True
        self.model.endResetModel()

    def updateOutputsForKey(self, key: str):
        self.model.key = key
        units = signal_units
        default_unit = "cts"
        conc_units = number_concentration_units
        default_conc_unit = "#/ml"

        if key == "mass":
            units = mass_units
            default_unit = "fg"
            default_conc_unit = "µg/L"
            conc_units = mass_concentration_units
        elif key == "size":
            units = size_units
            default_unit = "nm"
        elif key != "signal":
            raise ValueError(f"unknown key '{key}'")

        orientation = self.view.header.orientation()
        self.model.setHeaderData(1, orientation, conc_units, role=UnitsRole)
        self.model.setHeaderData(
            1, orientation, default_conc_unit, role=CurrentUnitRole
        )

        for i in range(3, 8):
            self.model.setHeaderData(i, orientation, units, role=UnitsRole)
            self.model.setHeaderData(i, orientation, default_unit, role=CurrentUnitRole)

    def setSignificantFigures(self, sf: int):
        delegate = self.view.itemDelegate()
        assert isinstance(delegate, ValueWidgetDelegate)
        delegate.setSigFigs(sf)
        self.view.setItemDelegate(delegate)

    def clear(self):
        self.setResults({})

    def saveHeaderLayout(self, settings: QtCore.QSettings, prefix: str):
        orientation = self.view.header.orientation()
        settings.beginWriteArray(prefix)
        for i in range(self.view.header.count()):
            settings.setArrayIndex(i)
            settings.setValue("Hidden", self.view.header.isSectionHidden(i))
            settings.setValue(
                "Unit",
                self.model.headerData(i, orientation, role=CurrentUnitRole),
            )
        settings.endArray()

    def restoreHeaderLayout(self, settings: QtCore.QSettings, prefix: str):
        orientation = self.view.header.orientation()
        count = settings.beginReadArray(prefix)
        if count != self.view.header.count():
            logger.warning("unable to restore headers for outputs, mismatched size")
            settings.endArray()
            return

        for i in range(self.view.header.count()):
            settings.setArrayIndex(i)
            self.view.header.setSectionHidden(i, settings.value("Hidden") == "true")
            self.model.setHeaderData(
                i, orientation, settings.value("Unit"), role=CurrentUnitRole
            )
        settings.endArray()

    def defaultLayout(self):
        for i in range(self.view.header.count()):
            self.view.header.setSectionHidden(i, False)
