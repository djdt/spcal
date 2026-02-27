from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.widgets.values import ValueWidget


class AdvancedPoissonOptions(QtWidgets.QWidget):
    optionsChanged = QtCore.Signal()

    def __init__(self, image: str, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Poisson Options")

        _image = QtGui.QImage(image).scaledToWidth(
            400, QtCore.Qt.TransformationMode.SmoothTransformation
        )
        pixmap = QtGui.QPixmap(_image)

        label = QtWidgets.QLabel()
        label.setPixmap(pixmap)
        label.setFixedSize(pixmap.size())

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(label, 0, QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setLayout(layout)

    def options(self) -> tuple[float | None, float | None]:
        raise NotImplementedError


class CurrieOptions(AdvancedPoissonOptions):
    def __init__(
        self, eta: float, epsilon: float, parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(":img/currie2008.png", parent)

        self.eta = ValueWidget(eta, min=1.0, max=2.0, step=0.1, sigfigs=2)
        self.epsilon = ValueWidget(epsilon, min=0.0, max=1.0, step=0.1, sigfigs=2)
        self.eta.valueChanged.connect(self.optionsChanged)
        self.epsilon.valueChanged.connect(self.optionsChanged)

        layout = QtWidgets.QFormLayout()
        layout.addRow("η", self.eta)
        layout.addRow("ε", self.epsilon)

        base_layout = self.layout()
        assert isinstance(base_layout, QtWidgets.QVBoxLayout)
        base_layout.insertLayout(0, layout)

    def options(self) -> tuple[float | None, float | None]:
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

        self.t_sample = ValueWidget(t_sample, min=0.0, max=1.0, step=0.1, sigfigs=2)
        self.t_blank = ValueWidget(t_blank, min=0.0, max=1.0, step=0.1, sigfigs=2)
        self.t_sample.valueChanged.connect(self.optionsChanged)
        self.t_blank.valueChanged.connect(self.optionsChanged)

        layout = QtWidgets.QFormLayout()
        layout.addRow("t sample", self.t_sample)
        layout.addRow("t blank", self.t_blank)

        base_layout = self.layout()
        assert isinstance(base_layout, QtWidgets.QVBoxLayout)
        base_layout.insertLayout(0, layout)

    def options(self) -> tuple[float | None, float | None]:
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

        self.currie.optionsChanged.connect(self.completeChanged)
        self.formula_a.optionsChanged.connect(self.completeChanged)
        self.formula_c.optionsChanged.connect(self.completeChanged)
        self.stapleton.optionsChanged.connect(self.completeChanged)
        self.poisson_formula.currentIndexChanged.connect(self.completeChanged)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.poisson_formula)
        layout.addWidget(self.poisson_stack, 1)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

    def isComplete(self) -> bool:
        widget = self.poisson_stack.currentWidget()
        assert isinstance(widget, AdvancedPoissonOptions)
        options = widget.options()
        return options[0] is not None and options[1] is not None

    def completeChanged(self):
        button = self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        button.setEnabled(self.isComplete())

    def accept(self):
        widget = self.poisson_stack.currentWidget()
        assert isinstance(widget, AdvancedPoissonOptions)
        options = widget.options()

        self.optionsSelected.emit(
            self.poisson_formula.currentText(), options[0], options[1]
        )
        super().accept()
