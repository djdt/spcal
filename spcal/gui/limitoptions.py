from statistics import NormalDist

from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.widgets import ValueWidget


class PoissonOptions(QtWidgets.QWidget):
    def __init__(self, image: str, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)

        label = QtWidgets.QLabel()
        label.setPixmap(QtGui.QImage.fromData(image))

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(label)


class CurrieOptions(PoissonOptions):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(":img/currie.png", parent)

        self.eta = ValueWidget(2.0, validator=QtGui.QDoubleValidator(1.0, 2.0, 2))
        self.epsilon = ValueWidget(0.5, validator=QtGui.QDoubleValidator(0.0, 1.0, 2))


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
    w = QtWidgets.QStackedWidget()

    w.addWidget(GaussianOptions())

    w.show()
    app.exec()
