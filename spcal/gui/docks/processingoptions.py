from PySide6 import QtCore, QtWidgets

from spcal.gui.widgets.values import ValueWidget
from spcal.processing.options import SPCalProcessingOptions


class SPCalProcessingOptionsWidget(QtWidgets.QWidget):
    optionsChanged = QtCore.Signal()

    def __init__(
        self,
        options: SPCalProcessingOptions,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)

        self.accumulation_method = QtWidgets.QComboBox()
        self.accumulation_method.addItems(
            ["Signal Mean", "Half Detection Threshold", "Detection Threshold"]
        )
        self.accumulation_method.setCurrentText(options.accumulation_method.title())
        self.accumulation_method.currentTextChanged.connect(self.optionsChanged)
        self.accumulation_method.setToolTip(
            "Value at which to stop accumlating data (summing adjacent points)."
        )

        self.calibration_mode = QtWidgets.QComboBox()
        self.calibration_mode.addItems(["Efficiency", "Mass Response"])
        self.calibration_mode.setCurrentText(options.calibration_mode.title())
        self.calibration_mode.currentTextChanged.connect(self.optionsChanged)
        for i, tooltip in enumerate(
            [
                "Manually enter the transport efficiency.",
                "Calculate the efficiency using a reference particle.",
                "Use the mass response of a reference particle.",
            ]
        ):
            self.calibration_mode.setItemData(
                i, tooltip, QtCore.Qt.ItemDataRole.ToolTipRole
            )
        self.calibration_mode.setToolTip(
            self.calibration_mode.currentData(QtCore.Qt.ItemDataRole.ToolTipRole)
        )
        self.calibration_mode.currentIndexChanged.connect(
            lambda i: self.calibration_mode.setToolTip(
                self.calibration_mode.itemData(i, QtCore.Qt.ItemDataRole.ToolTipRole)
            )
        )

        self.points_required = QtWidgets.QSpinBox()
        self.points_required.setRange(1, 99)
        self.points_required.setValue(options.points_required)
        self.points_required.setToolTip(
            "Minimum number of points above detection threshold to be detected."
        )
        self.points_required.valueChanged.connect(self.optionsChanged)

        self.prominence_required = QtWidgets.QSpinBox()
        self.prominence_required.setRange(0, 100)
        self.prominence_required.setValue(int(options.prominence_required * 100))
        self.prominence_required.setSuffix(" %")
        self.prominence_required.setToolTip(
            "Fraction of highest peak required before splitting merged peaks."
        )
        self.prominence_required.valueChanged.connect(self.optionsChanged)

        self.cluster_distance = QtWidgets.QSpinBox()
        self.cluster_distance.setRange(0, 100)
        self.cluster_distance.setValue(int(options.cluster_distance * 100))
        self.cluster_distance.setSuffix(" %")
        self.cluster_distance.setToolTip(
            "Minimum distance between the two closest points in clusters before merging."
        )
        self.cluster_distance.valueChanged.connect(self.optionsChanged)

        form_layout = QtWidgets.QFormLayout()
        form_layout.addRow("Calibration mode:", self.calibration_mode)
        form_layout.addRow("Accumulation mode:", self.accumulation_method)
        form_layout.addRow("Points required:", self.points_required)
        form_layout.addRow("Prominence required:", self.prominence_required)
        form_layout.addRow("Cluster distance:", self.cluster_distance)

        self.setLayout(form_layout)

    def processingOptions(self) -> SPCalProcessingOptions:
        return SPCalProcessingOptions(
            self.calibration_mode.currentText().lower(),
            self.accumulation_method.currentText().lower(),
            self.points_required.value(),
            self.prominence_required.value() / 100.0,
            self.cluster_distance.value() / 100.0,
        )

    def setProcessingOptions(self, options: SPCalProcessingOptions):
        self.blockSignals(True)
        self.calibration_mode.setCurrentText(options.calibration_mode.title())
        self.accumulation_method.setCurrentText(options.accumulation_method.title())
        self.points_required.setValue(options.points_required)
        self.prominence_required.setValue(int(options.prominence_required * 100.0))
        self.cluster_distance.setValue(int(options.cluster_distance * 100.0))
        self.blockSignals(False)
        self.optionsChanged.emit()


class SPCalProcessingOptionsDock(QtWidgets.QDockWidget):
    optionsChanged = QtCore.Signal()
    efficiencyDialogRequested = QtCore.Signal()

    def __init__(
        self,
        options: SPCalProcessingOptions,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("spcal-processing-options-dock")
        self.setWindowTitle("Processing Options")

        self.options_widget = SPCalProcessingOptionsWidget(options)
        self.options_widget.optionsChanged.connect(self.optionsChanged)
        self.setWidget(self.options_widget)

    def processingOptions(self) -> SPCalProcessingOptions:
        return self.options_widget.processingOptions()

    def setProcessingOptions(self, options: SPCalProcessingOptions):
        self.options_widget.setProcessingOptions(options)

    def reset(self):
        self.blockSignals(True)
        self.options_widget.setProcessingOptions(SPCalProcessingOptions())
        self.blockSignals(False)
        self.optionsChanged.emit()

    def setSignificantFigures(self, num: int):
        for widget in self.options_widget.findChildren(ValueWidget):
            widget.setSigFigs(num)
