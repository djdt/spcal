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
        self.setObjectName("spcal-results-dock")
        self.setWindowTitle("Results")

        self.view = ResultOutputView()

        self.view.isotopeSelected.connect(self.requestCurrentIsotope)
        self.view.requestRemoveIsotopes.connect(self.requestRemoveIsotopes)
        self.view.requestAddExpression.connect(self.requestAddExpression)
        self.view.requestRemoveExpressions.connect(self.requestRemoveExpressions)

        self.setWidget(self.view)
        self.updateOutputsForKey("signal")

    def setResults(self, results: list[SPCalProcessingResult]):
        self.view.setResults(results)

    def updateOutputsForKey(self, key: str):
        self.view.results_model.key = key
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

        orientation = self.view.header.orientation()
        self.view.results_model.setHeaderData(
            1, orientation, conc_units, role=UnitsRole
        )
        self.view.results_model.setHeaderData(
            1, orientation, default_conc_unit, role=CurrentUnitRole
        )

        for i in range(2, 7):
            self.view.results_model.setHeaderData(
                i, orientation, units, role=UnitsRole
            )
            self.view.results_model.setHeaderData(
                i, orientation, default_unit, role=CurrentUnitRole
            )

    def setSignificantFigures(self, sf: int):
        delegate = self.view.itemDelegate()
        assert isinstance(delegate, ValueWidgetDelegate)
        delegate.setSigFigs(sf)
        self.view.setItemDelegate(delegate)

    def reset(self):
        self.setResults([])
