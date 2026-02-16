from PySide6 import QtCore, QtWidgets

from spcal.gui.modelviews import (
    CurrentUnitRole,
    UnitsRole,
)
from spcal.gui.modelviews.values import ValueWidgetDelegate
from spcal.gui.modelviews.results import ResultOutputView
from spcal.isotope import SPCalIsotopeBase, SPCalIsotopeExpression
from spcal.processing.result import SPCalProcessingResult
from spcal.siunits import (
    mass_concentration_units,
    mass_units,
    number_concentration_units,
    signal_units,
    size_units,
    volume_units,
)


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
        self.updateOutputsForKey("signal")

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

        for i in range(2, 7):
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
