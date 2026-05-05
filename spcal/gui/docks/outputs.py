from PySide6 import QtCore, QtWidgets
import logging

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
)

logger = logging.getLogger(__name__)


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
        self.view.results_model.setHeaderData(
            1, orientation, conc_units, role=UnitsRole
        )
        self.view.results_model.setHeaderData(
            1, orientation, default_conc_unit, role=CurrentUnitRole
        )

        for i in range(3, 8):
            self.view.results_model.setHeaderData(i, orientation, units, role=UnitsRole)
            self.view.results_model.setHeaderData(
                i, orientation, default_unit, role=CurrentUnitRole
            )

    def setSignificantFigures(self, sf: int):
        delegate = self.view.itemDelegate()
        assert isinstance(delegate, ValueWidgetDelegate)
        delegate.setSigFigs(sf)
        self.view.setItemDelegate(delegate)

    def clear(self):
        self.setResults([])

    def saveHeaderLayout(self, settings: QtCore.QSettings, prefix: str):
        orientation = self.view.header.orientation()
        settings.beginWriteArray(prefix)
        for i in range(self.view.header.count()):
            settings.setArrayIndex(i)
            settings.setValue("Hidden", self.view.header.isSectionHidden(i))
            settings.setValue(
                "Unit",
                self.view.results_model.headerData(
                    i, orientation, role=CurrentUnitRole
                ),
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
            self.view.results_model.setHeaderData(
                i, orientation, settings.value("Unit"), role=CurrentUnitRole
            )
        settings.endArray()

    def defaultLayout(self):
        for i in range(self.view.header.count()):
            self.view.header.setSectionHidden(i, False)
