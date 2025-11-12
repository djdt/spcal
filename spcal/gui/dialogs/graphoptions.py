from PySide6 import QtCore, QtWidgets

from spcal.gui.objects import DoubleOrPercentValidator
from spcal.gui.widgets import UnitsWidget
from spcal.siunits import (
    mass_units,
    signal_units,
    size_units,
    volume_units,
)


class HistogramOptionsDialog(QtWidgets.QDialog):
    optionsChanged = QtCore.Signal(dict, float, bool)

    def __init__(
        self,
        # fit: str | None,
        bin_widths: dict[str, float | None],
        percentile: float,
        draw_filtered: bool,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Histogram Options")

        self.bin_widths = bin_widths.copy()
        self.percentile = percentile
        self.draw_filtered = draw_filtered

        sf = int(QtCore.QSettings().value("SigFigs", 4))  # type: ignore

        self.width_signal = UnitsWidget(
            signal_units,
            base_value=self.bin_widths.get("signal", None),
        )
        self.width_mass = UnitsWidget(
            mass_units, base_value=self.bin_widths.get("mass", None), sigfigs=sf
        )
        self.width_size = UnitsWidget(
            size_units,
            base_value=self.bin_widths.get("size", None),
            sigfigs=sf,
        )
        self.width_volume = UnitsWidget(
            volume_units,
            base_value=self.bin_widths.get("volume", None),
            sigfigs=sf,
        )

        for widget in [
            self.width_signal,
            self.width_mass,
            self.width_size,
            self.width_volume,
        ]:
            widget.setBestUnit()
            widget._value.lineEdit().setPlaceholderText("auto")

        self.spinbox_percentile = QtWidgets.QDoubleSpinBox()
        self.spinbox_percentile.setRange(50.0, 100.0)
        self.spinbox_percentile.setValue(percentile)
        self.spinbox_percentile.setDecimals(1)
        self.spinbox_percentile.setToolTip("Bin histrogram up to the this percentile.")

        self.check_draw_filtered = QtWidgets.QCheckBox("Draw filtered detections.")
        self.check_draw_filtered.setToolTip("Draw filtered detections in grey.")
        self.check_draw_filtered.setChecked(draw_filtered)

        box_widths = QtWidgets.QGroupBox("Bin Widths")
        box_widths_layout = QtWidgets.QFormLayout()
        box_widths_layout.addRow("Signal:", self.width_signal)
        box_widths_layout.addRow("Mass:", self.width_mass)
        box_widths_layout.addRow("Size:", self.width_size)
        box_widths_layout.addRow("Volume:", self.width_volume)
        box_widths.setLayout(box_widths_layout)

        box_max = QtWidgets.QGroupBox("Highest Bin")
        box_max_layout = QtWidgets.QFormLayout()
        box_max_layout.addRow("Percentile", self.spinbox_percentile)
        box_max.setLayout(box_max_layout)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.RestoreDefaults
            | QtWidgets.QDialogButtonBox.StandardButton.Apply
            | QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.clicked.connect(self.buttonBoxClicked)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(box_widths)
        layout.addWidget(box_max)
        layout.addWidget(self.check_draw_filtered)
        layout.addWidget(self.button_box)

        self.setLayout(layout)

    def buttonBoxClicked(self, button: QtWidgets.QAbstractButton) -> None:
        sbutton = self.button_box.standardButton(button)
        if sbutton == QtWidgets.QDialogButtonBox.StandardButton.RestoreDefaults:
            self.reset()
            self.apply()
        elif sbutton == QtWidgets.QDialogButtonBox.StandardButton.Apply:
            self.apply()
        elif sbutton == QtWidgets.QDialogButtonBox.StandardButton.Ok:
            self.apply()
            self.accept()
        else:
            self.reject()

    def apply(self) -> None:
        bin_widths = {
            "signal": self.width_signal.baseValue(),
            "mass": self.width_mass.baseValue(),
            "size": self.width_size.baseValue(),
            "volume": self.width_volume.baseValue(),
        }

        percentile = self.spinbox_percentile.value()

        draw_filtered = self.check_draw_filtered.isChecked()

        if (
            bin_widths != self.bin_widths
            or percentile != self.percentile
            or draw_filtered != self.draw_filtered
        ):
            self.bin_widths = bin_widths
            self.percentile = percentile
            self.draw_filtered = draw_filtered
            self.optionsChanged.emit(
                self.bin_widths, self.percentile, self.draw_filtered
            )

    def reset(self) -> None:
        for widget in [
            self.width_signal,
            self.width_mass,
            self.width_size,
            self.width_volume,
        ]:
            widget.setBaseValue(None)
        self.spinbox_percentile.setValue(95.0)
        self.check_draw_filtered.setChecked(False)


class CompositionsOptionsDialog(QtWidgets.QDialog):
    optionsChanged = QtCore.Signal(object, str)

    def __init__(
        self,
        # distance: float = 0.03,
        minimum_size: str | float = "5%",
        mode: str = "pie",
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Composition Options")

        # self.distance = distance
        self.minimum_size = minimum_size
        self.mode = mode

        # self.spinbox_distance = QtWidgets.QDoubleSpinBox()
        # self.spinbox_distance.setRange(0.1, 99.9)
        # self.spinbox_distance.setValue(distance * 100.0)
        # self.spinbox_distance.setDecimals(1)
        # self.spinbox_distance.setSuffix(" %")

        self.lineedit_size = QtWidgets.QLineEdit(str(minimum_size))
        self.lineedit_size.setValidator(DoubleOrPercentValidator(0.0, 1e99, 3, 0, 100))

        self.combo_mode = QtWidgets.QComboBox()
        self.combo_mode.addItems(["Pie", "Bar"])
        if mode == "bar":
            self.combo_mode.setCurrentIndex(1)

        box = QtWidgets.QGroupBox("Clustering")
        box_layout = QtWidgets.QFormLayout()
        # box_layout.addRow("Distance threshold:", self.spinbox_distance)
        box_layout.addRow("Minimum cluster size:", self.lineedit_size)
        box_layout.addRow("Display mode:", self.combo_mode)
        box.setLayout(box_layout)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.RestoreDefaults
            | QtWidgets.QDialogButtonBox.StandardButton.Apply
            | QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.clicked.connect(self.buttonBoxClicked)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(box)
        layout.addWidget(self.button_box)

        self.setLayout(layout)

    def buttonBoxClicked(self, button: QtWidgets.QAbstractButton) -> None:
        sbutton = self.button_box.standardButton(button)
        if sbutton == QtWidgets.QDialogButtonBox.StandardButton.RestoreDefaults:
            self.reset()
            self.apply()
        elif sbutton == QtWidgets.QDialogButtonBox.StandardButton.Apply:
            self.apply()
        elif sbutton == QtWidgets.QDialogButtonBox.StandardButton.Ok:
            self.apply()
            self.accept()
        else:
            self.reject()

    def apply(self) -> None:
        # distance = self.spinbox_distance.value() / 100.0
        size = self.lineedit_size.text().strip().replace(" ", "")
        mode = self.combo_mode.currentText().lower()

        # Check for changes
        if (
            # abs(self.distance - distance) > 0.001
            self.minimum_size != size
            or self.mode != mode
        ):
            # self.distance = distance
            self.minimum_size = size
            self.mode = mode
            self.optionsChanged.emit(self.minimum_size, self.mode)

    def reset(self) -> None:
        # self.spinbox_distance.setValue(3.0)
        self.lineedit_size.setText("5%")
        self.combo_mode.setCurrentIndex(0)


#
class ScatterOptionsDialog(QtWidgets.QDialog):
    pass


#     weightingChanged = QtCore.Signal(str)
#     drawFilteredChanged = QtCore.Signal(bool)
#
#     weightings = ["none", "1/x", "1/x²", "1/y", "1/y²"]
#
#     def __init__(
#         self,
#         weighting: str = "none",
#         draw_filtered: bool = False,
#         parent: QtWidgets.QWidget | None = None,
#     ):
#         super().__init__(parent)
#         self.setWindowTitle("Scatter Options")
#
#         if weighting not in weighting:
#             raise ValueError("invalid weighting string.")
#         self.weighting = weighting
#         self.draw_filtered = draw_filtered
#
#         self.combo_weighting = QtWidgets.QComboBox()
#         self.combo_weighting.addItems(ScatterOptionsDialog.weightings)
#
#         box = QtWidgets.QGroupBox("")
#         box.setLayout(QtWidgets.QFormLayout())
#         box.layout().addRow("Weighting:", self.combo_weighting)
#
#         self.check_draw_filtered = QtWidgets.QCheckBox("Draw filtered detections.")
#         self.check_draw_filtered.setToolTip("Draw filtered detections in grey.")
#         self.check_draw_filtered.setChecked(draw_filtered)
#
#         self.button_box = QtWidgets.QDialogButtonBox(
#             QtWidgets.QDialogButtonBox.RestoreDefaults
#             | QtWidgets.QDialogButtonBox.Apply
#             | QtWidgets.QDialogButtonBox.Ok
#             | QtWidgets.QDialogButtonBox.Cancel
#         )
#         self.button_box.clicked.connect(self.buttonBoxClicked)
#
#         layout = QtWidgets.QVBoxLayout()
#         layout.addWidget(box)
#         layout.addWidget(self.check_draw_filtered)
#         layout.addWidget(self.button_box)
#
#         self.setLayout(layout)
#
#     def buttonBoxClicked(self, button: QtWidgets.QAbstractButton) -> None:
#         sbutton = self.button_box.standardButton(button)
#         if sbutton == QtWidgets.QDialogButtonBox.RestoreDefaults:
#             self.reset()
#             self.apply()
#         elif sbutton == QtWidgets.QDialogButtonBox.Apply:
#             self.apply()
#         elif sbutton == QtWidgets.QDialogButtonBox.Ok:
#             self.apply()
#             self.accept()
#         else:
#             self.reject()
#
#     def apply(self) -> None:
#         weighting = self.combo_weighting.currentText()
#         draw_filtered = self.check_draw_filtered.isChecked()
#
#         # Check for changes
#         if weighting != self.weighting:
#             self.weighting = weighting
#             self.weightingChanged.emit(weighting)
#         if draw_filtered != self.draw_filtered:
#             self.draw_filtered = draw_filtered
#             self.drawFilteredChanged.emit(draw_filtered)
#
#     def reset(self) -> None:
#         self.combo_weighting.setCurrentText("none")
#         self.check_draw_filtered.setChecked(False)
