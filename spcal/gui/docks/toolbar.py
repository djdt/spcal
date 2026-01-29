from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.modelviews.isotope import IsotopeComboBox
from spcal.gui.util import create_action
from spcal.isotope import SPCalIsotopeBase
from spcal.processing.method import SPCalProcessingMethod


class ScatterExprLineEdit(QtWidgets.QLineEdit):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)

        self._completer = QtWidgets.QCompleter()
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed
        )

    def insertCompletion(self, completion: str):
        prefix = self._completer.completionPrefix()
        self.setText(self.text()[: self.cursorPosition() - len(prefix)] + completion)

    def setIsotopes(self, isotopes: list[SPCalIsotopeBase]):
        self._completer = QtWidgets.QCompleter([str(isotope) for isotope in isotopes])
        self._completer.setCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)
        self._completer.setWidget(self)
        self._completer.activated.connect(self.insertCompletion)

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if self._completer.popup().isVisible():
            if event.key() in [  # Ignore keys when popup is present
                QtCore.Qt.Key.Key_Enter,
                QtCore.Qt.Key.Key_Return,
                QtCore.Qt.Key.Key_Escape,
                QtCore.Qt.Key.Key_Tab,
                QtCore.Qt.Key.Key_Down,
                QtCore.Qt.Key.Key_Up,
            ]:
                event.ignore()
                return

        super().keyPressEvent(event)

        eow = "~!@#$%^&*()+{}|:\"<>?,./;'[]\\-= "

        current_word = self.text()[
            max(self.text()[: self.cursorPosition()].rfind(char) for char in eow)
            + 1 : self.cursorPosition()
        ]
        if len(current_word) < 2 or current_word == self._completer.currentCompletion():
            self._completer.popup().hide()
        else:
            self._completer.setCompletionPrefix(current_word)
            self._completer.popup().setCurrentIndex(
                self._completer.completionModel().index(0, 0)
            )
            rect = self.cursorRect()
            rect.setWidth(
                self._completer.popup().sizeHintForColumn(0)
                + self._completer.popup().verticalScrollBar().sizeHint().width()
            )
            self._completer.complete(rect)


class SPCalOptionsToolBar(QtWidgets.QToolBar):
    isotopeChanged = QtCore.Signal(SPCalIsotopeBase)
    keyChanged = QtCore.Signal(str)
    scatterOptionsChanged = QtCore.Signal()

    requestViewOptionsDialog = QtCore.Signal()

    def __init__(
        self,
        title: str = "SPCal Options Toolbar",
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(title, parent=parent)

        # default widgets
        self.combo_isotope = IsotopeComboBox()
        self.combo_isotope.isotopeChanged.connect(self.isotopeChanged)

        self.combo_key = QtWidgets.QComboBox()
        self.combo_key.currentTextChanged.connect(self.keyChanged)
        self.combo_key.addItems(SPCalProcessingMethod.CALIBRATION_KEYS)

        self.scatter_x = ScatterExprLineEdit()
        self.scatter_y = ScatterExprLineEdit()

        self.scatter_key_x = QtWidgets.QComboBox()
        self.scatter_key_y = QtWidgets.QComboBox()
        self.scatter_key_x.addItems(SPCalProcessingMethod.CALIBRATION_KEYS)
        self.scatter_key_y.addItems(SPCalProcessingMethod.CALIBRATION_KEYS)

        self.scatter_x.editingFinished.connect(self.scatterOptionsChanged)
        self.scatter_y.editingFinished.connect(self.scatterOptionsChanged)
        self.scatter_key_x.activated.connect(self.scatterOptionsChanged)
        self.scatter_key_y.activated.connect(self.scatterOptionsChanged)

        self.action_all_isotopes = create_action(
            "office-chart-line-stacked",
            "Overlay Isotopes",
            "Plot all isotope signals.",
            self.overlayOptionChanged,
            checkable=True,
        )

        self.action_view_options = create_action(
            "configure",
            "Graph Options",
            "Set options specific to the current graph.",
            self.requestViewOptionsDialog,
        )

        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

        self.addWidget(spacer)

        self.addAction(self.action_all_isotopes)

        self.isotope_action = self.addWidget(self.combo_isotope)

        self.key_action = self.addWidget(self.combo_key)

        self.scatter_y_action = self.addWidget(self.scatter_y)
        self.scatter_key_y_action = self.addWidget(self.scatter_key_y)
        self.scatter_x_action = self.addWidget(self.scatter_x)
        self.scatter_key_x_action = self.addWidget(self.scatter_key_x)

        self.addSeparator()
        self.addAction(self.action_view_options)

        for action in [
            self.action_view_options,
            self.scatter_y_action,
            self.scatter_key_y_action,
            self.scatter_x_action,
            self.scatter_key_x_action,
        ]:
            action.setVisible(False)

    def onViewChanged(self, view: str):
        self.isotope_action.setVisible(view in ["particle", "histogram"])
        self.action_all_isotopes.setVisible(view in ["particle", "histogram"])
        self.action_view_options.setVisible(view not in ["particle", "scatter"])

        self.key_action.setVisible(view not in ["scatter"])
        for widget in [
            self.scatter_x_action,
            self.scatter_y_action,
            self.scatter_key_x_action,
            self.scatter_key_y_action,
        ]:
            widget.setVisible(view in ["scatter"])

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
        for combo in [self.combo_isotope]:
            current = combo.currentIsotope()
            combo.blockSignals(True)

            combo.clear()
            combo.addIsotopes(isotopes)

            if current is not None:
                combo.setCurrentIsotope(current)

            if combo.currentIndex() == -1:
                combo.setCurrentIndex(0)

            combo.blockSignals(False)

        self.scatter_x.setIsotopes(isotopes)
        self.scatter_y.setIsotopes(isotopes)
        self.scatter_x.setText(str(isotopes[0]))
        self.scatter_y.setText(str(isotopes[1]))

        self.action_all_isotopes.setEnabled(len(isotopes) > 1)

    def reset(self):
        for combo in [self.combo_isotope]:
            combo.blockSignals(True)
            combo.clear()
            combo.blockSignals(False)


class SPCalViewToolBar(QtWidgets.QToolBar):
    viewChanged = QtCore.Signal(str)
    requestFilterDialog = QtCore.Signal()

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

        self.action_filter = create_action(
            "view-filter",
            "Filter Detections",
            "Filter detections based on element compositions.",
            self.requestFilterDialog,
        )

        action_group_views = QtGui.QActionGroup(self)
        for action in self.view_actions.values():
            action_group_views.addAction(action)
            self.addAction(action)

        self.addSeparator()
        self.addSeparator()
        self.addAction(self.action_filter)

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
        self.viewChanged.emit(view)
