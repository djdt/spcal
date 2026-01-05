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
    requestZoomReset = QtCore.Signal()

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

        self.action_zoom_reset = create_action(
            "zoom-original",
            "Reset Zoom",
            "Reset the zoom to the full graph extent.",
            self.requestZoomReset,
        )

        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

        self.addWidget(spacer)

        self.addAction(self.action_all_isotopes)

        self.isotope_action = self.addWidget(self.combo_isotope)
        self.isotope_additional_action = self.addWidget(self.combo_isotope_additional)
        self.key_action = self.addWidget(self.combo_key)

        self.isotope_additional_action.setVisible(False)

        self.addSeparator()
        self.addAction(self.action_filter)

        self.addSeparator()
        self.addAction(self.action_zoom_reset)

    def onViewChanged(self, view: str):
        if view in ["scatter", "spectra"]:
            self.isotope_additional_action.setVisible(True)
        else:
            self.isotope_additional_action.setVisible(False)

        # insert a fake background isotope
        self.combo_isotope_additional.blockSignals(True)
        if view in ["spectra"] and not isinstance(
            self.combo_isotope_additional.isotope(0), BackgroundIsotope
        ):
            self.combo_isotope_additional.insertItem(0, "Background")
            self.combo_isotope_additional.setItemData(
                0, BackgroundIsotope(), IsotopeRole
            )
            self.combo_isotope_additional.setCurrentIndex(0)
        elif isinstance(self.combo_isotope_additional.isotope(0), BackgroundIsotope):
            self.combo_isotope_additional.removeItem(0)
        self.combo_isotope_additional.blockSignals(False)

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
    requestViewOptionsDialog = QtCore.Signal()

    VIEWS = {
        "particle": ("office-chart-line", "Show signals and detected peaks."),
        "histogram": (
            "view-object-histogram-linear",
            "Show results as signal, mass and size histograms.",
        ),
        "composition": ("office-chart-pie", "Plot particle compositions."),
        "spectra": ("office-chart-bar", "Plot the mass spectra of selected peaks."),
        "scatter": (
            "office-chart-scatter",
            "Plot the signal, mass or size of two isotopes.",
        ),
    }

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__("View Toolbar", parent=parent)

        self.view_actions = {
            name: create_action(
                icon,
                f"{name.capitalize()} View",
                tip,
                self.onViewChanged,
                checkable=True,
            )
            for name, (icon, tip) in SPCalViewToolBar.VIEWS.items()
        }
        next(iter(self.view_actions.values())).setChecked(True)

        self.action_view_options = create_action(
            "configure",
            "Graph Options",
            "Set options specific to the current graph.",
            self.requestViewOptionsDialog,
        )
        self.action_view_options.setEnabled(False)

        action_group_views = QtGui.QActionGroup(self)
        for action in self.view_actions.values():
            action_group_views.addAction(action)
            self.addAction(action)

        self.addSeparator()
        self.addAction(self.action_view_options)

        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

        self.addWidget(spacer)

    def currentView(self) -> str:
        for name, action in self.view_actions.items():
            if action.isChecked():
                return name
        raise StopIteration

    def onViewChanged(self):
        view = self.currentView()
        self.action_view_options.setEnabled(view in ["histogram", "composition"])
        self.viewChanged.emit(view)
