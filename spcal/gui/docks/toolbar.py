from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.util import create_action

from spcal.isotope import SPCalIsotope
from spcal.gui.modelviews.isotope import IsotopeComboBox


class SPCalToolBar(QtWidgets.QToolBar):
    isotopeChanged = QtCore.Signal(SPCalIsotope)
    viewChanged = QtCore.Signal(QtGui.QAction)

    def __init__(
        self, view_actions: list[QtGui.QAction], parent: QtWidgets.QWidget | None = None
    ):
        super().__init__("SPCal", parent=parent)

        self.combo_isotope = IsotopeComboBox()
        self.combo_isotope.isotopeChanged.connect(self.isotopeChanged)

        self.action_all_isotopes = create_action(
            "office-chart-line-stacked",
            "Overlay Isotopes",
            "Plot all isotope signals.",
            self.overlayOptionChanged,
            checkable=True,
        )

        self.action_view_signal = create_action(
            "office-chart-line",
            "Signal View",
            "View raw signal and detected particle peaks.",
            None,
            checkable=True,
        )
        self.action_view_histogram = create_action(
            "office-chart-histogram",
            "Results View",
            "View signal and calibrated results as histograms.",
            None,
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

        self.action_group_views.triggered.connect(self.viewChanged)

    def selectedIsotopes(self) -> list[SPCalIsotope]:
        if self.action_all_isotopes.isChecked():
            return [
                self.combo_isotope.isotope(i) for i in range(self.combo_isotope.count())
            ]
        else:
            return [self.combo_isotope.currentIsotope()]

    def overlayOptionChanged(self, checked: bool):
        self.combo_isotope.setEnabled(not checked)

    def setIsotopes(self, isotopes: list[SPCalIsotope]):
        self.action_all_isotopes.setChecked(False)

        self.combo_isotope.blockSignals(True)
        self.combo_isotope.clear()
        self.combo_isotope.addIsotopes(isotopes)
        self.combo_isotope.blockSignals(False)

        self.action_all_isotopes.setEnabled(len(isotopes) > 1)
