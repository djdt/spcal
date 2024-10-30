from PySide6 import QtCore, QtWidgets


class DefaultOptionsDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)

        self.setWindowTitle("Default Options")

        layout = QtWidgets.QVBoxLayout()

        settings = QtCore.QSettings()

        self.accumulation_method = QtWidgets.QComboBox()
        self.accumulation_method.addItem(
            ["Detection Threshold", "Half Detection Threshold", "Signal Mean"]
        )
        self.accumulation_method.setCurrentText(
            settings.value("Threshold/AccumulationMethod", "Signal Mean")
        )

        self.points_required = QtWidgets.QSpinBox()
        self.points_required.setRange(1, 99)
        self.points_required.setValue(
            int(settings.value("Threshold/PointsRequired", 1))
        )

        gbox_threshold = QtWidgets.QGroupBox("Thresholding")
        gbox_threshold.setLayout(QtWidgets.QFormLayout())
        gbox_threshold.layout().addRow("Accumulation method", self.accumulation_method)
        gbox_threshold.layout().addRow("Points required", self.points_requierd)

        # gbox_cpoisson = QtWidgets.QGroupBox("Compound Poisson")
        # gbox_gaussian = QtWidgets.QGroupBox("Compound Poisson")
        # gbox_poisson = QtWidgets.QGroupBox("Compound Poisson")
        layout.addWidget(gbox_threshold)

        self.setLayout(layout)
