from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.objects import DoubleOrPercentValidator
from spcal.gui.widgets import UnitsWidget
from spcal.siunits import (
    mass_units,
    molar_concentration_units,
    signal_units,
    size_units,
)


class HistogramOptionsDialog(QtWidgets.QDialog):
    fitChanged = QtCore.Signal(str)
    binWidthsChanged = QtCore.Signal(dict)
    drawFilteredChanged = QtCore.Signal(bool)

    def __init__(
        self,
        fit: str | None,
        bin_widths: dict[str, float | None],
        draw_filtered: bool,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Histogram Options")

        self.fit = fit
        self.bin_widths = bin_widths.copy()
        self.draw_filtered = draw_filtered

        self.radio_fit_off = QtWidgets.QRadioButton("Off")
        self.radio_fit_norm = QtWidgets.QRadioButton("Normal")
        self.radio_fit_log = QtWidgets.QRadioButton("Log normal")

        fit_group = QtWidgets.QButtonGroup()
        for button in [self.radio_fit_off, self.radio_fit_norm, self.radio_fit_log]:
            fit_group.addButton(button)

        if self.fit is None:
            self.radio_fit_off.setChecked(True)
        elif self.fit == "normal":
            self.radio_fit_norm.setChecked(True)
        elif self.fit == "log normal":
            self.radio_fit_log.setChecked(True)
        else:
            raise ValueError("HistogramOptionsDialog: unknown fit")

        sf = int(QtCore.QSettings().value("SigFigs", 4))
        color = self.palette().color(QtGui.QPalette.Base)

        self.width_signal = UnitsWidget(
            signal_units,
            base_value=self.bin_widths["signal"],
            color_invalid=color,
            validator=QtGui.QIntValidator(0, 999999999),
        )
        self.width_mass = UnitsWidget(
            mass_units,
            base_value=self.bin_widths["mass"],
            color_invalid=color,
            format=sf,
        )
        self.width_size = UnitsWidget(
            size_units,
            base_value=self.bin_widths["size"],
            color_invalid=color,
            format=sf,
        )
        self.width_conc = UnitsWidget(
            molar_concentration_units,
            base_value=self.bin_widths["cell_concentration"],
            color_invalid=color,
            format=sf,
        )

        for widget in [
            self.width_signal,
            self.width_mass,
            self.width_size,
            self.width_conc,
        ]:
            widget.setBestUnit()
            widget.lineedit.setPlaceholderText("auto")

        self.check_draw_filtered = QtWidgets.QCheckBox("Draw filtered detections.")
        self.check_draw_filtered.setToolTip("Draw filtered detections in grey.")
        self.check_draw_filtered.setChecked(draw_filtered)

        box_fit = QtWidgets.QGroupBox("Curve Fit")
        box_fit.setLayout(QtWidgets.QHBoxLayout())
        for button in [self.radio_fit_off, self.radio_fit_norm, self.radio_fit_log]:
            box_fit.layout().addWidget(button)

        box_widths = QtWidgets.QGroupBox("Bin Widths")
        box_widths.setLayout(QtWidgets.QFormLayout())
        box_widths.layout().addRow("Signal:", self.width_signal)
        box_widths.layout().addRow("Mass:", self.width_mass)
        box_widths.layout().addRow("Size:", self.width_size)
        box_widths.layout().addRow("Concentration:", self.width_conc)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.RestoreDefaults
            | QtWidgets.QDialogButtonBox.Apply
            | QtWidgets.QDialogButtonBox.Ok
            | QtWidgets.QDialogButtonBox.Cancel
        )
        self.button_box.clicked.connect(self.buttonBoxClicked)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(box_fit)
        layout.addWidget(box_widths)
        layout.addWidget(self.check_draw_filtered)
        layout.addWidget(self.button_box)

        self.setLayout(layout)

    def buttonBoxClicked(self, button: QtWidgets.QAbstractButton) -> None:
        sbutton = self.button_box.standardButton(button)
        if sbutton == QtWidgets.QDialogButtonBox.RestoreDefaults:
            self.reset()
            self.apply()
        elif sbutton == QtWidgets.QDialogButtonBox.Apply:
            self.apply()
        elif sbutton == QtWidgets.QDialogButtonBox.Ok:
            self.apply()
            self.accept()
        else:
            self.reject()

    def apply(self) -> None:
        if self.radio_fit_off.isChecked():
            fit = None
        elif self.radio_fit_norm.isChecked():
            fit = "normal"
        else:
            fit = "log normal"

        bin_widths = {
            "signal": self.width_signal.baseValue(),
            "mass": self.width_mass.baseValue(),
            "size": self.width_size.baseValue(),
            "cell_concentration": self.width_conc.baseValue(),
        }

        draw_filtered = self.check_draw_filtered.isChecked()

        # Check for changes
        if fit != self.fit:
            self.fit = fit
            self.fitChanged.emit(fit)
        if bin_widths != self.bin_widths:
            self.bin_widths = bin_widths
            self.binWidthsChanged.emit(bin_widths)
        if draw_filtered != self.draw_filtered:
            self.draw_filtered = draw_filtered
            self.drawFilteredChanged.emit(draw_filtered)

    def reset(self) -> None:
        self.radio_fit_log.setChecked(True)
        for widget in [
            self.width_signal,
            self.width_mass,
            self.width_size,
            self.width_conc,
        ]:
            widget.setBaseValue(None)
        self.check_draw_filtered.setChecked(False)


class CompositionsOptionsDialog(QtWidgets.QDialog):
    distanceChanged = QtCore.Signal(float)
    minimumSizeChanged = QtCore.Signal(str)
    modeChanged = QtCore.Signal(str)

    def __init__(
        self,
        distance: float = 0.03,
        minimum_size: str | float = "5%",
        mode: str = "pie",
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Composition Options")

        self.distance = distance
        self.minimum_size = minimum_size
        self.mode = mode

        self.lineedit_distance = QtWidgets.QLineEdit(str(distance * 100.0))
        self.lineedit_distance.setValidator(QtGui.QDoubleValidator(0.1, 99.9, 1))

        self.lineedit_size = QtWidgets.QLineEdit(str(minimum_size))
        self.lineedit_size.setValidator(DoubleOrPercentValidator(0.0, 1e99, 3, 0, 100))

        self.combo_mode = QtWidgets.QComboBox()
        self.combo_mode.addItems(["Pie", "Bar"])
        if mode == "bar":
            self.combo_mode.setCurrentIndex(1)

        layout_dist = QtWidgets.QHBoxLayout()
        layout_dist.addWidget(self.lineedit_distance, 1)
        layout_dist.addWidget(QtWidgets.QLabel("%"), 0)

        box = QtWidgets.QGroupBox("Clustering")
        box.setLayout(QtWidgets.QFormLayout())
        box.layout().addRow("Distance threshold:", layout_dist)
        box.layout().addRow("Minimum cluster size:", self.lineedit_size)
        box.layout().addRow("Display mode:", self.combo_mode)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.RestoreDefaults
            | QtWidgets.QDialogButtonBox.Apply
            | QtWidgets.QDialogButtonBox.Ok
            | QtWidgets.QDialogButtonBox.Cancel
        )
        self.button_box.clicked.connect(self.buttonBoxClicked)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(box)
        layout.addWidget(self.button_box)

        self.setLayout(layout)

    def buttonBoxClicked(self, button: QtWidgets.QAbstractButton) -> None:
        sbutton = self.button_box.standardButton(button)
        if sbutton == QtWidgets.QDialogButtonBox.RestoreDefaults:
            self.reset()
            self.apply()
        elif sbutton == QtWidgets.QDialogButtonBox.Apply:
            self.apply()
        elif sbutton == QtWidgets.QDialogButtonBox.Ok:
            self.apply()
            self.accept()
        else:
            self.reject()

    def apply(self) -> None:
        distance = float(self.lineedit_distance.text()) / 100.0
        size = self.lineedit_size.text().strip().replace(" ", "")
        mode = self.combo_mode.currentText().lower()

        # Check for changes
        if abs(self.distance - distance) > 0.001:
            self.distance = distance
            self.distanceChanged.emit(distance)
        if size != self.minimum_size:
            self.minimum_size == size
            self.minimumSizeChanged.emit(str(size))
        if mode != self.mode:
            self.modeChanged.emit(mode)

    def reset(self) -> None:
        self.lineedit_distance.setText("3.0")
        self.lineedit_size.setText("5%")
        self.combo_mode.setCurrentIndex(0)


class ScatterOptionsDialog(QtWidgets.QDialog):
    weightingChanged = QtCore.Signal(str)
    showFilteredChanged = QtCore.Signal(bool)

    def __init__(
        self,
        weighting: str = "equal",
        draw_filtered: bool = False,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Scatter Options")

        self.weighting = weighting
        self.draw_filtered = draw_filtered

        self.combo_weighting = QtWidgets.QComboBox()
        self.combo_weighting.addItems(["none", "1/x", "1/x²", "1/y", "1/y²"])

        box = QtWidgets.QGroupBox("")
        box.setLayout(QtWidgets.QFormLayout())
        box.layout().addRow("Weighting:", self.combo_weighting)

        self.check_draw_filtered = QtWidgets.QCheckBox("Draw filtered detections.")
        self.check_draw_filtered.setToolTip("Draw filtered detections in grey.")
        self.check_draw_filtered.setChecked(draw_filtered)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.RestoreDefaults
            | QtWidgets.QDialogButtonBox.Apply
            | QtWidgets.QDialogButtonBox.Ok
            | QtWidgets.QDialogButtonBox.Cancel
        )
        self.button_box.clicked.connect(self.buttonBoxClicked)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(box)
        layout.addWidget(self.check_draw_filtered)
        layout.addWidget(self.button_box)

        self.setLayout(layout)

    def buttonBoxClicked(self, button: QtWidgets.QAbstractButton) -> None:
        sbutton = self.button_box.standardButton(button)
        if sbutton == QtWidgets.QDialogButtonBox.RestoreDefaults:
            self.reset()
            self.apply()
        elif sbutton == QtWidgets.QDialogButtonBox.Apply:
            self.apply()
        elif sbutton == QtWidgets.QDialogButtonBox.Ok:
            self.apply()
            self.accept()
        else:
            self.reject()

    def apply(self) -> None:
        weighting = self.combo_weighting.currentText()
        draw_filtered = self.check_draw_filtered.isChecked()

        # Check for changes
        if weighting != self.weighting:
            self.weighting = weighting
            self.weightingChanged.emit(weighting)
        if draw_filtered != self.draw_filtered:
            self.showFilteredChanged.emit(draw_filtered)

    def reset(self) -> None:
        self.combo_weighting.setCurrentText("none")
        self.check_draw_filtered.setChecked(False)
