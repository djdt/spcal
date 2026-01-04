from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.modelviews.isotope import IsotopeComboBox
from spcal.gui.util import create_action
from spcal.isotope import SPCalIsotopeBase
from spcal.processing.method import SPCalProcessingMethod
from spcal.gui.modelviews import IsotopeRole


class BackgroundIsotope(SPCalIsotopeBase):
    pass


class SPCalOptionsToolBar(QtWidgets.QToolBar):
    keyChanged = QtCore.Signal(str)
    isotopeChanged = QtCore.Signal(SPCalIsotopeBase)

    requestFilterDialog = QtCore.Signal()

    def __init__(
        self,
        title: str = "SPCal Options Toolbar",
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(title, parent=parent)

        self.combo_isotope = IsotopeComboBox()
        self.combo_isotope.isotopeChanged.connect(self.isotopeChanged)

        self.combo_isotope_additional = IsotopeComboBox()
        self.combo_isotope_additional.isotopeChanged.connect(self.isotopeChanged)

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
            self.requestFilterDialog,
        )

        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

        self.addWidget(spacer)

        self.addAction(self.action_filter)
        self.addSeparator()

        self.addAction(self.action_all_isotopes)

        self.isotope_action = self.addWidget(self.combo_isotope)
        self.isotope_additional_action = self.addWidget(self.combo_isotope_additional)
        self.key_action = self.addWidget(self.combo_key)

        self.isotope_additional_action.setVisible(False)

    def onViewChanged(self, view: str):
        if view in ["spectra"]:
            self.isotope_additional_action.setVisible(True)
        else:
            self.isotope_additional_action.setVisible(False)

    def selectedIsotopes(self) -> list[SPCalIsotopeBase]:
        if (
            self.action_all_isotopes.isVisible()
            and self.action_all_isotopes.isChecked()
        ):
            return [
                self.combo_isotope.isotope(i) for i in range(self.combo_isotope.count())
            ]
        else:
            return [self.combo_isotope.currentIsotope()]

    def overlayOptionChanged(self, checked: bool):
        self.combo_isotope.setEnabled(not checked)
        self.isotopeChanged.emit(None)

    def setIsotopes(self, isotopes: list[SPCalIsotopeBase]):
        for combo in [self.combo_isotope, self.combo_isotope_additional]:
            current = combo.currentIsotope()
            combo.blockSignals(True)

            combo.clear()

            if combo == self.combo_isotope_additional:
                combo.insertItem(0, "Background")
                combo.setItemData(0, BackgroundIsotope(), IsotopeRole)
            combo.addIsotopes(isotopes)

            if current is not None:
                combo.setCurrentIsotope(current)

            if combo.currentIndex() == -1:
                combo.setCurrentIndex(0)

            combo.blockSignals(False)

        self.action_all_isotopes.setEnabled(len(isotopes) > 1)

    def reset(self):
        for combo in [self.combo_isotope, self.combo_isotope_additional]:
            combo.blockSignals(True)
            combo.clear()
            combo.blockSignals(False)


class SPCalViewToolBar(QtWidgets.QToolBar):
    viewChanged = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__("View Toolbar", parent=parent)

        self.action_view_composition = create_action(
            "office-chart-pie",
            "Composition View",
            "Cluster and view results as pie or bar charts.",
            lambda: self.viewChanged.emit("composition"),
            checkable=True,
        )
        self.action_view_histogram = create_action(
            "view-object-histogram-linear",
            "Results View",
            "View signal and calibrated results as histograms.",
            lambda: self.viewChanged.emit("histogram"),
            checkable=True,
        )
        self.action_view_particle = create_action(
            "office-chart-line",
            "Particle View",
            "View raw signal and detected particle peaks.",
            lambda: self.viewChanged.emit("particle"),
            checkable=True,
        )
        self.action_view_spectra = create_action(
            "none",
            "Spectra View",
            "View the mass spectra of selected peaks.",
            lambda: self.viewChanged.emit("spectra"),
            checkable=True,
        )
        self.action_view_particle.setChecked(True)

        action_group_views = QtGui.QActionGroup(self)
        for action in [
            self.action_view_histogram,
            self.action_view_particle,
            self.action_view_composition,
            self.action_view_spectra,
        ]:
            action_group_views.addAction(action)
            self.addAction(action)

        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

        self.addWidget(spacer)
