from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.modelviews.isotope import IsotopeComboBox
from spcal.gui.util import create_action
from spcal.isotope import SPCalIsotopeBase
from spcal.processing.method import SPCalProcessingMethod


class SPCalToolBar(QtWidgets.QToolBar):
    keyChanged = QtCore.Signal(str)
    isotopeChanged = QtCore.Signal(SPCalIsotopeBase)
    viewChanged = QtCore.Signal(QtGui.QAction)

    def __init__(
        self, view_actions: list[QtGui.QAction], parent: QtWidgets.QWidget | None = None
    ):
        super().__init__("SPCal", parent=parent)

        self.combo_isotope = IsotopeComboBox()
        self.combo_isotope.isotopeChanged.connect(self.isotopeChanged)

        self.combo_key = QtWidgets.QComboBox()
        self.combo_key.currentTextChanged.connect(self.keyChanged)
        self.combo_key.addItems(SPCalProcessingMethod.CALIBRATION_KEYS)

        self.action_all_isotopes = create_action(
            "office-chart-line-stacked",
            "Overlay Isotopes",
            "Plot all isotope signals.",
            self.overlayOptionChanged,
            checkable=True,
        )

        self.action_filter = create_action(
            "view-filter",
            "Filter Detections",
            "Filter detections based on element compositions.",
            None,
        )

        self.action_group_views = QtGui.QActionGroup(self)
        for action in view_actions:
            self.action_group_views.addAction(action)
            self.addAction(action)

        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

        self.addWidget(spacer)

        self.addAction(self.action_filter)
        self.addSeparator()

        self.addAction(self.action_all_isotopes)
        self.addWidget(self.combo_isotope)
        self.addWidget(self.combo_key)

        self.action_group_views.triggered.connect(self.viewChanged)

    def selectedIsotopes(self) -> list[SPCalIsotopeBase]:
        if self.action_all_isotopes.isChecked():
            return [
                self.combo_isotope.isotope(i) for i in range(self.combo_isotope.count())
            ]
        else:
            return [self.combo_isotope.currentIsotope()]

    def overlayOptionChanged(self, checked: bool):
        self.combo_isotope.setEnabled(not checked)

    def setIsotopes(self, isotopes: list[SPCalIsotopeBase]):
        self.combo_isotope.blockSignals(True)
        current = self.combo_isotope.currentIsotope()

        self.combo_isotope.clear()
        self.combo_isotope.addIsotopes(isotopes)
        if current is not None:
            self.combo_isotope.setCurrentIsotope(current)

        self.combo_isotope.blockSignals(False)

        self.action_all_isotopes.setEnabled(len(isotopes) > 1)
    
    def reset(self):
        self.combo_isotope.blockSignals(True)
        self.combo_isotope.clear()
        self.combo_isotope.blockSignals(False)
