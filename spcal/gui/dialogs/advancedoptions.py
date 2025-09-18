from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.widgets import ValueWidget


class AdvancedThresholdOptions(QtWidgets.QDialog):
    optionsSelected = QtCore.Signal(str, int, float)

    def __init__(
        self,
        accumulation_method: str,
        points_required: int,
        prominence_required: float,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Advanced Thresholding Options")

        # Controllable limit
        self.limit_accumulation = QtWidgets.QComboBox()
        self.limit_accumulation.addItems(
            ["Detection Threshold", "Half Detection Threshold", "Signal Mean"]
        )
        self.limit_accumulation.setCurrentText(accumulation_method)
        self.limit_accumulation.setItemData(
            0,
            "Sum contiguous regions above the detection threshold.",
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )
        self.limit_accumulation.setItemData(
            1,
            "Sum contiguous regions above the midpoint of the threshold and mean.",
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )
        self.limit_accumulation.setItemData(
            2,
            "Sum contiguous regions above the signal mean.",
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )
        self.limit_accumulation.setToolTip(
            self.limit_accumulation.currentData(QtCore.Qt.ItemDataRole.ToolTipRole)
        )
        self.limit_accumulation.currentIndexChanged.connect(
            lambda i: self.limit_accumulation.setToolTip(
                self.limit_accumulation.itemData(i, QtCore.Qt.ItemDataRole.ToolTipRole)
            )
        )

        self.points_req = QtWidgets.QSpinBox()
        self.points_req.setRange(1, 99)
        self.points_req.setValue(points_required)
        self.points_req.setToolTip(
            "Number of points > threshold to be considered a detection"
        )

        self.prominence_req = QtWidgets.QSpinBox()
        self.prominence_req.setRange(0, 100)
        self.prominence_req.setValue(int(prominence_required * 100))
        self.prominence_req.setSuffix(" %")
        self.prominence_req.setToolTip(
            "Minimum fraction of largest overlapping peak before splitting."
        )

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout = QtWidgets.QFormLayout()
        layout.addRow("Accumulation method:", self.limit_accumulation)
        layout.addRow("Points required:", self.points_req)
        layout.addRow("Prominence required:", self.prominence_req)
        layout.addRow(self.button_box)
        self.setLayout(layout)

    def accept(self) -> None:
        self.optionsSelected.emit(
            self.limit_accumulation.currentText(),
            self.points_req.value(),
            self.prominence_req.value() / 100.0,
        )
        super().accept()


class AdvancedPoissonOptions(QtWidgets.QWidget):
    def __init__(self, image: str, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Poisson Options")

        image = QtGui.QImage(image).scaledToWidth(
            400, QtCore.Qt.TransformationMode.SmoothTransformation
        )
        pixmap = QtGui.QPixmap(image)

        label = QtWidgets.QLabel()
        label.setPixmap(pixmap)
        label.setFixedSize(pixmap.size())

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(label, 0, QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setLayout(layout)

    def options(self) -> tuple[float, float]:
        raise NotImplementedError


class CurrieOptions(AdvancedPoissonOptions):
    def __init__(
        self, eta: float, epsilon: float, parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(":img/currie2008.png", parent)

        self.eta = ValueWidget(eta, validator=QtGui.QDoubleValidator(1.0, 2.0, 2))
        self.epsilon = ValueWidget(
            epsilon, validator=QtGui.QDoubleValidator(0.0, 1.0, 2)
        )

        layout = QtWidgets.QFormLayout()
        layout.addRow("η", self.eta)
        layout.addRow("ε", self.epsilon)

        self.layout().insertLayout(0, layout)

    def options(self) -> tuple[float, float]:
        return self.eta.value(), self.epsilon.value()


class MARLAPFormulaOptions(AdvancedPoissonOptions):
    def __init__(
        self,
        image: str,
        t_sample: float,
        t_blank: float,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(image, parent)

        self.t_sample = ValueWidget(
            t_sample, validator=QtGui.QDoubleValidator(0.0, 1.0, 2)
        )
        self.t_blank = ValueWidget(
            t_blank, validator=QtGui.QDoubleValidator(0.0, 1.0, 2)
        )

        layout = QtWidgets.QFormLayout()
        layout.addRow("t sample", self.t_sample)
        layout.addRow("t blank", self.t_blank)

        self.layout().insertLayout(0, layout)

    def options(self) -> tuple[float, float]:
        return self.t_sample.value(), self.t_blank.value()


class AdvancedPoissonDialog(QtWidgets.QDialog):
    optionsSelected = QtCore.Signal(str, float, float)

    def __init__(
        self,
        formula: str,
        eta: float,
        epsilon: float,
        t_sample: float,
        t_blank: float,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Advanced Options")

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        self.poisson_formula = QtWidgets.QComboBox()
        self.poisson_formula.addItems(["Currie", "Formula A", "Formula C", "Stapleton"])

        self.currie = CurrieOptions(eta, epsilon)
        self.formula_a = MARLAPFormulaOptions(":img/formula_a.png", t_sample, t_blank)
        self.formula_c = MARLAPFormulaOptions(":img/formula_c.png", t_sample, t_blank)
        self.stapleton = MARLAPFormulaOptions(":img/stapleton.png", t_sample, t_blank)

        self.poisson_stack = QtWidgets.QStackedWidget()
        self.poisson_stack.addWidget(self.currie)
        self.poisson_stack.addWidget(self.formula_a)
        self.poisson_stack.addWidget(self.formula_c)
        self.poisson_stack.addWidget(self.stapleton)
        self.poisson_formula.currentIndexChanged.connect(
            self.poisson_stack.setCurrentIndex
        )

        self.poisson_formula.setCurrentText(formula)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.poisson_formula)
        layout.addWidget(self.poisson_stack, 1)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

    def accept(self) -> None:
        options = self.poisson_stack.currentWidget().options()
        self.optionsSelected.emit(
            self.poisson_formula.currentText(), options[0], options[1]
        )
        super().accept()
