from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.widgets import ValueWidget


class PoissonOptions(QtWidgets.QWidget):
    def __init__(self, image: str, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)

        pixmap = QtGui.QPixmap(image)

        label = QtWidgets.QLabel()
        label.setPixmap(pixmap)
        label.setFixedSize(pixmap.size())

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(label)
        self.setLayout(layout)


class CurrieOptions(PoissonOptions):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(":img/currie2008.png", parent)

        self.eta = ValueWidget(2.0, validator=QtGui.QDoubleValidator(1.0, 2.0, 2))
        self.epsilon = ValueWidget(0.5, validator=QtGui.QDoubleValidator(0.0, 1.0, 2))

        layout = QtWidgets.QFormLayout()
        layout.addRow("η", self.eta)
        layout.addRow("ε", self.epsilon)

        self.layout().insertLayout(0, layout)


class MARLAPFormulaOptions(PoissonOptions):
    def __init__(self, image: str, parent: QtWidgets.QWidget | None = None):
        super().__init__(image, parent)

        self.t_sample = ValueWidget(1.0, validator=QtGui.QDoubleValidator(0.0, 1.0, 2))
        self.t_blank = ValueWidget(1.0, validator=QtGui.QDoubleValidator(0.0, 1.0, 2))

        layout = QtWidgets.QFormLayout()
        layout.addRow("t sample", self.t_sample)
        layout.addRow("t blank", self.t_blank)

        self.layout().insertLayout(0, layout)


class AdvancedLimitOptions(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Options")

        self.poisson_formula = QtWidgets.QComboBox()
        self.poisson_formula.addItems(["Currie", "Formula A", "Formula C", "Stapleton"])

        self.poisson_stack = QtWidgets.QStackedWidget()

        layout = QtWidgets.QVBoxLayout()


if __name__ == "__main__":
    app = QtWidgets.QApplication()

    w = CurrieOptions()

    w.show()
    app.exec()
