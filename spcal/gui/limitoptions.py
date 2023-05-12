from statistics import NormalDist

import h5py
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.io import get_open_spcal_path
from spcal.gui.widgets import ValueWidget
from spcal.io import nu, tofwerk


class LimitOptions(QtWidgets.QGroupBox):
    limitOptionsChanged = QtCore.Signal()

    def __init__(
        self, name: str, alpha: float = 0.001, parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(name, parent=parent)
        sf = int(QtCore.QSettings().value("sigfigs", 4))
        self.alpha = ValueWidget(
            alpha, validator=QtGui.QDoubleValidator(1e-16, 0.5, 9), format=sf
        )
        self.alpha.setToolTip("False positive error rate.")
        self.alpha.valueChanged.connect(self.limitOptionsChanged)

        layout = QtWidgets.QFormLayout()
        layout.addRow("α (Type I):", self.alpha)
        self.setLayout(layout)

    def state(self) -> dict:
        return {"alpha": self.alpha.value()}

    def setState(self, state: dict) -> None:
        self.blockSignals(True)
        if "alpha" in state:
            self.alpha.setValue(state["alpha"])
        self.blockSignals(False)
        self.limitOptionsChanged.emit()


class CompoundPoissonOptions(LimitOptions):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__("Compound Poisson", 1e-6, parent=parent)
        sf = int(QtCore.QSettings().value("sigfigs", 4))

        self.single_ion_dist: np.ndarray | None = None

        self.single_ion_average = ValueWidget(
            1.0, validator=QtGui.QDoubleValidator(1e-6, 1e99, 9), format=sf
        )
        self.single_ion_average.setToolTip(
            "The average single ion area, this is used "
            "as the mean and stddev to simulate a SIA distribution."
        )
        self.single_ion_average.valueChanged.connect(self.limitOptionsChanged)

        self.accumulations = ValueWidget(
            1, validator=QtGui.QIntValidator(1, 100), format=".0f"
        )
        self.accumulations.setToolTip(
            "The number of ion extraction events per acquisition."
        )
        self.accumulations.valueChanged.connect(self.limitOptionsChanged)

        self.button_single_ion = QtWidgets.QPushButton("Set SIA Distribution...")
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(
            self.button_single_ion, 1, QtCore.Qt.AlignmentFlag.AlignRight
        )
        self.button_single_ion.pressed.connect(self.loadSingleIonData)

        self.layout().addRow("Average SIA:", self.single_ion_average)
        self.layout().addRow("Accumulations:", self.accumulations)
        self.layout().addRow(button_layout)

    def loadSingleIonData(self) -> None:
        path = get_open_spcal_path(self, "Single Ion Data")
        if path is None:
            return

        if tofwerk.is_tofwerk_file(path):
            data = h5py.File(path)["SingleIon"]["Data"]
        elif nu.is_nu_directory(path):
            _, counts, _ = nu.read_nu_directory(path, cycle=1, raw=True)
            data = nu.single_ion_distribution(counts)
        else:
            with path.open("r") as fp:
                delimiter = "\t"
                skip_rows = 0
                for line in fp.readlines(1024):
                    try:
                        delimiter = next(d for d in ["\t", ";", ",", " "] if d in line)
                        float(line.split(delimiter)[-1])
                        break
                    except (ValueError, StopIteration):
                        pass
                    skip_rows += 1
                count = line.count(delimiter) + 1

            if count == 1:  # raw points from dist
                data = np.genfromtxt(  # type: ignore
                    path,
                    delimiter=delimiter,
                    skip_header=skip_rows,
                    dtype=np.float64,
                    usecols=(0),
                    invalid_raise=False,
                    loose=True,
                )
            elif count == 2:  # hist as bin, count
                bins, counts = np.genfromtxt(  # type: ignore
                    path,
                    delimiter=delimiter,
                    skip_header=skip_rows,
                    dtype=np.float64,
                    usecolss=(0, 1),
                    unpack=True,
                    invalid_raise=False,
                    loose=True,
                )
                data = np.stack((bins, counts), axis=1)
        self.setSingleIon(data)

    def getSingleIon(self) -> float | None | np.ndarray:
        return (
            self.single_ion_dist
            if self.single_ion_dist is not None
            else self.single_ion_average.value()
        )

    def setSingleIon(self, sia: float | np.ndarray) -> None:
        if isinstance(sia, float):
            self.single_ion_average.setValue(sia)
            self.single_ion_dist = None
            self.single_ion_average.setEnabled(True)
        else:
            self.single_ion_dist = sia
            self.single_ion_average.setEnabled(False)
            if sia.ndim == 2:  # hist of bins, counts
                mean = np.average(sia[:, 0], weights=sia[:, 1])
            else:
                mean = np.mean(sia)
            self.single_ion_average.setValue(mean)

    def state(self) -> dict:
        return {
            "alpha": self.alpha.value(),
            "single ion": self.getSingleIon(),
            "accumulations": int(self.accumulations.value() or 1),
        }

    def setState(self, state: dict) -> None:
        self.blockSignals(True)
        if "alpha" in state:
            self.alpha.setValue(state["alpha"])
        if "single ion" in state:
            self.setSingleIon(state["single ion"])
        if "accumulations" in state:
            self.accumulations.setValue(state["accumulations"])
        self.blockSignals(False)
        self.limitOptionsChanged.emit()


class GaussianOptions(LimitOptions):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__("Gaussian", 2.867e-7, parent=parent)  # 5.0 sigma
        self.alpha.valueChanged.connect(self.updateSigma)

        sf = int(QtCore.QSettings().value("sigfigs", 4))
        self.sigma = ValueWidget(
            5.0, validator=QtGui.QDoubleValidator(0.0, 8.0, 4), format=sf
        )
        self.sigma.setToolTip(
            "Type I error rate as number of standard deviations from mean."
        )
        self.sigma.valueChanged.connect(self.updateAlpha)

        self.layout().addRow("σ:", self.sigma)

    def updateAlpha(self) -> None:
        sigma = self.sigma.value()
        if sigma is None:
            return
        alpha = 1.0 - NormalDist().cdf(sigma)
        self.alpha.valueChanged.disconnect(self.updateSigma)
        self.alpha.setValue(float(f"{alpha:.4g}"))
        self.alpha.valueChanged.connect(self.updateSigma)

    def updateSigma(self) -> None:
        alpha = self.alpha.value()
        if alpha is None:
            alpha = 1e-6
        sigma = NormalDist().inv_cdf(1.0 - alpha)
        self.sigma.valueChanged.disconnect(self.updateAlpha)
        self.sigma.setValue(round(sigma, 4))
        self.sigma.valueChanged.connect(self.updateAlpha)


class PoissonOptions(LimitOptions):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__("Poisson", 0.001, parent=parent)
        self.button_advanced = QtWidgets.QPushButton("Advanced Options...")
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(
            self.button_advanced, 1, QtCore.Qt.AlignmentFlag.AlignRight
        )
        self.layout().addRow(button_layout)
        self.button_advanced.setEnabled(False)

        # class AdvancedPoissonOptions(QtWidgets.QWidget):
        #     def __init__(self, image: str, parent: QtWidgets.QWidget | None = None):
        #         super().__init__(parent)

        #         pixmap = QtGui.QPixmap(image)

        #         label = QtWidgets.QLabel()
        #         label.setPixmap(pixmap)
        #         label.setFixedSize(pixmap.size())

        #         layout = QtWidgets.QVBoxLayout()
        #         layout.addWidget(label)
        #         self.setLayout(layout)

        # class CurrieOptions(AdvancedPoissonOptions):
        #     def __init__(self, parent: QtWidgets.QWidget | None = None):
        #         super().__init__(":img/currie2008.png", parent)

        #         self.eta = ValueWidget(2.0, validator=QtGui.QDoubleValidator(1.0, 2.0, 2))
        #         self.epsilon = ValueWidget(0.5, validator=QtGui.QDoubleValidator(0.0, 1.0, 2))

        #         layout = QtWidgets.QFormLayout()
        #         layout.addRow("η", self.eta)
        #         layout.addRow("ε", self.epsilon)

        #         self.layout().insertLayout(0, layout)

        # class MARLAPFormulaOptions(AdvancedPoissonOptions):
        #     def __init__(self, image: str, parent: QtWidgets.QWidget | None = None):
        #         super().__init__(image, parent)

        #         self.t_sample = ValueWidget(1.0, validator=QtGui.QDoubleValidator(0.0, 1.0, 2))
        #         self.t_blank = ValueWidget(1.0, validator=QtGui.QDoubleValidator(0.0, 1.0, 2))

        #         layout = QtWidgets.QFormLayout()
        #         layout.addRow("t sample", self.t_sample)
        #         layout.addRow("t blank", self.t_blank)

        #         self.layout().insertLayout(0, layout)

        # class AdvancedLimitOptions(QtWidgets.QDialog):
        #     def __init__(self, parent: QtWidgets.QWidget | None = None):
        #         super().__init__(parent)
        #         self.setWindowTitle("Advanced Options")

        #         self.poisson_formula = QtWidgets.QComboBox()
        #         self.poisson_formula.addItems(["Currie", "Formula A", "Formula C", "Stapleton"])

        #         self.poisson_stack = QtWidgets.QStackedWidget()

        layout = QtWidgets.QVBoxLayout()
