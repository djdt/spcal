import logging
from typing import Sequence

from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.dialogs.tools import MassFractionCalculatorDialog, ParticleDatabaseDialog
from spcal.gui.modelviews import (
    BaseValueRole,
    IsotopeOptionRole,
    IsotopeRole,
    CurrentUnitRole,
)
from spcal.gui.modelviews.basic import BasicTableView
from spcal.gui.modelviews.units import UnitsHeaderView
from spcal.gui.modelviews.options import IsotopeOptionModel
from spcal.gui.modelviews.massfraction import MassFractionDelegate
from spcal.gui.modelviews.values import ValueWidgetDelegate
from spcal.gui.objects import ContextMenuRedirectFilter
from spcal.isotope import SPCalIsotopeBase
from spcal.processing.options import SPCalIsotopeOptions

logger = logging.getLogger(__name__)


class IsotopeOptionTable(BasicTableView):
    isotopeSelected = QtCore.Signal(SPCalIsotopeBase)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent=parent)
        self.header = UnitsHeaderView(QtCore.Qt.Orientation.Horizontal)
        self.header.setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.setHorizontalHeader(self.header)
        self.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents
        )
        self.verticalHeader().installEventFilter(ContextMenuRedirectFilter(self))
        self.verticalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Fixed
        )

        self.verticalHeader().sectionClicked.connect(self.onHeaderClicked)

    def setModel(self, model: QtCore.QAbstractItemModel | None):
        if not isinstance(model, IsotopeOptionModel):
            raise ValueError("IsotopeOptionTable requires an IsotopeOptionModel")
        super().setModel(model)
        for col, name in IsotopeOptionModel.COLUMN_LABELS.items():
            if name == "Mass Fraction":
                delegate = MassFractionDelegate(max=1.0, step=0.1)
            # elif name == "Density":
            #     delegate = DensityDelegate()
            else:
                delegate = ValueWidgetDelegate()
            self.setItemDelegateForColumn(col, delegate)

    def onHeaderClicked(self, section: int):
        isotope = self.model().index(section, 0).data(IsotopeRole)
        self.isotopeSelected.emit(isotope)

    def dialogParticleDatabase(self, index: QtCore.QModelIndex) -> QtWidgets.QDialog:
        def set_density(density: float | None):
            self.model().setData(index, density, BaseValueRole)

        dlg = ParticleDatabaseDialog(parent=self)
        dlg.densitySelected.connect(set_density)
        dlg.open()
        return dlg

    def dialogMassFractionCalculator(
        self, index: QtCore.QModelIndex
    ) -> QtWidgets.QDialog:
        def set_major_ratio(ratios: list):
            self.model().setData(index, float(ratios[0][1]), BaseValueRole)

        dlg = MassFractionCalculatorDialog(parent=self)
        dlg.ratiosSelected.connect(set_major_ratio)
        dlg.open()
        return dlg

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent):
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

    def setSignificantFigures(self, sf: int):
        for i in range(self.model().columnCount()):
            delegate = self.itemDelegateForColumn(i)
            assert isinstance(delegate, ValueWidgetDelegate)
            delegate.setSigFigs(sf)
            self.setItemDelegateForColumn(i, delegate)


class SPCalIsotopeOptionsDock(QtWidgets.QDockWidget):
    requestCurrentIsotope = QtCore.Signal(SPCalIsotopeBase)
    optionChanged = QtCore.Signal(SPCalIsotopeBase)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("spcal-isotope-optinos-dock")
        self.setWindowTitle("Isotope Options")

        self.model = IsotopeOptionModel()
        self.model.dataChanged.connect(self.onDataChanged)

        self.table = IsotopeOptionTable()
        self.table.isotopeSelected.connect(self.requestCurrentIsotope)
        self.table.setModel(self.model)

        self.setWidget(self.table)

    def onDataChanged(
        self,
        topleft: QtCore.QModelIndex,
        _bottom_right: QtCore.QModelIndex,
        roles: list[QtCore.Qt.ItemDataRole],
    ):
        isotope = self.model.data(topleft, IsotopeRole)
        if isotope is not None and BaseValueRole in roles:
            self.optionChanged.emit(isotope)

    def isotopeOptions(self) -> dict[SPCalIsotopeBase, SPCalIsotopeOptions]:
        return self.model.isotope_options

    def addIsotopes(self, isotopes: list[SPCalIsotopeBase]):
        self.model.beginResetModel()
        for isotope in isotopes:
            if isotope not in self.model.isotope_options:
                self.model.isotope_options[isotope] = SPCalIsotopeOptions(
                    None, None, None
                )
        self.model.endResetModel()

    def isotopes(self) -> list[SPCalIsotopeBase]:
        return list(self.model.isotope_options.keys())

    def setIsotopes(self, isotopes: Sequence[SPCalIsotopeBase]):
        self.model.dataChanged.disconnect(self.onDataChanged)
        self.model.beginResetModel()
        self.model.isotope_options = {
            isotope: SPCalIsotopeOptions(None, None, None)
            for isotope in sorted(isotopes)
        }
        self.model.endResetModel()
        self.model.dataChanged.connect(self.onDataChanged)

    def setIsotopeOption(self, isotope: SPCalIsotopeBase, option: SPCalIsotopeOptions):
        row = list(self.model.isotope_options.keys()).index(isotope)
        self.model.setData(self.model.index(row, 0), option, role=IsotopeOptionRole)

    def optionForIsotope(self, isotope: SPCalIsotopeBase) -> SPCalIsotopeOptions:
        return self.model.isotope_options[isotope]

    def clear(self):
        self.setIsotopes([])

    def setSignificantFigures(self, sf: int):
        self.table.setSignificantFigures(sf)

    def saveHeaderLayout(self, settings: QtCore.QSettings, prefix: str):
        orientation = self.table.header.orientation()
        settings.beginWriteArray(prefix)
        for i in range(self.table.header.count()):
            settings.setArrayIndex(i)
            settings.setValue("Hidden", self.table.header.isSectionHidden(i))
            settings.setValue(
                "Unit",
                self.model.headerData(i, orientation, role=CurrentUnitRole),
            )
        settings.endArray()

    def restoreHeaderLayout(self, settings: QtCore.QSettings, prefix: str):
        orientation = self.table.header.orientation()
        count = settings.beginReadArray(prefix)
        if count != self.table.header.count():
            logger.warning("unable to restore headers for isotopes, mismatched size")
            settings.endArray()
            return

        for i in range(self.table.header.count()):
            settings.setArrayIndex(i)
            self.table.header.setSectionHidden(i, settings.value("Hidden") == "true")
            self.model.setHeaderData(
                i, orientation, settings.value("Unit"), role=CurrentUnitRole
            )
        settings.endArray()

    def defaultLayout(self):
        for col, name in IsotopeOptionModel.COLUMN_LABELS.items():
            if name in ["Diameter", "Concentration", "Mass Response"]:
                self.table.header.setSectionHidden(col, True)
            else:
                self.table.header.setSectionHidden(col, False)
