import logging
from typing import Callable

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.util import Worker
from spcal.gui.widgets import ValueWidget
from spcal.nontarget import screen_element

logger = logging.getLogger(__name__)


class NonTargetScreeningDialog(QtWidgets.QDialog):
    ppmSelected = QtCore.Signal(float)
    dataSizeSelected = QtCore.Signal(int)
    screeningComplete = QtCore.Signal(np.ndarray)

    def __init__(
        self,
        get_data_function: Callable[[int], np.ndarray],
        screening_ppm: float = 100.0,
        minimum_data_size: int = 1_000_000,
        screening_compound_kws: dict | None = None,
        screening_gaussian_kws: dict | None = None,
        screening_poisson_kws: dict | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Non-target Screening")

        self.get_data_function = get_data_function
        self.screening_poisson_kws = screening_poisson_kws
        self.screening_gaussian_kws = screening_gaussian_kws
        self.screening_compound_kws = screening_compound_kws

        self.progress = QtWidgets.QProgressBar()
        self.aborted = False
        self.running = False
        self.threadpool = QtCore.QThreadPool()
        self.results: list[int] = []

        self.screening_ppm = ValueWidget(
            screening_ppm, validator=QtGui.QDoubleValidator(0, 1e6, 1), format=".1f"
        )
        self.screening_ppm.valueChanged.connect(self.completeChanged)
        self.screening_ppm.setToolTip(
            "The number of detection (as ppm) required to pass the screen."
        )

        self.data_size = ValueWidget(
            minimum_data_size,
            validator=QtGui.QIntValidator(1, 100_000_000),
            format=".0f",
        )
        self.data_size.valueChanged.connect(self.completeChanged)
        self.data_size.setToolTip(
            "The minimum number of events to use during screening."
        )

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout = QtWidgets.QFormLayout()
        layout.addRow("Screening ppm:", self.screening_ppm)
        layout.addRow("Minimum no. events:", self.data_size)
        layout.addRow(self.progress)
        layout.addRow(self.button_box)

        self.setLayout(layout)

    def completeChanged(self) -> None:
        button = self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        button.setEnabled(self.isComplete())

    def isComplete(self) -> bool:
        return (
            self.screening_ppm.hasAcceptableInput()
            and self.data_size.hasAcceptableInput()
        )

    def threadComplete(self) -> None:
        if self.aborted:
            return

        self.progress.setValue(self.progress.value() + 1)

        if self.progress.value() == self.progress.maximum() and self.running:
            self.finalise()

    def threadFailed(self, exception: Exception) -> None:
        if self.aborted:
            return
        self.abort()

        logger.exception(exception)
        msg = QtWidgets.QMessageBox(
            QtWidgets.QMessageBox.Warning,
            "Screening Failed",
            str(exception),
            parent=self,
        )
        msg.exec()

    def setControlsEnabled(self, enabled: bool) -> None:
        button = self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        button.setEnabled(enabled)
        self.screening_ppm.setEnabled(enabled)
        self.data_size.setEnabled(enabled)

    def abort(self) -> None:
        self.aborted = True
        self.threadpool.clear()
        self.threadpool.waitForDone()
        self.progress.reset()
        self.running = False

        self.setControlsEnabled(True)

    def accept(self) -> None:
        def idx_screen_element(
            idx: int, x: np.ndarray, ppm: float, limit_kws: dict
        ) -> int:
            result = screen_element(x, ppm, limit_kws=limit_kws, mode="detections")
            return idx if result else -1

        self.ppmSelected.emit(self.screening_ppm.value() or 1e6)
        self.dataSizeSelected.emit(int(self.data_size.value() or 0))

        self.setControlsEnabled(False)

        self.aborted = False
        self.running = True
        self.results.clear()

        self.progress.setMaximum(0)  # Set to draw a busy indicator
        self.progress.repaint()

        data = self.get_data_function(int(self.data_size.value() or 0))

        self.progress.setValue(0)
        self.progress.setMaximum(data.shape[1])

        limit_kws = {
            "poisson_kws": self.screening_poisson_kws,
            "gaussian_kws": self.screening_gaussian_kws,
            "compound_kws": self.screening_compound_kws,
        }

        for i in range(data.shape[1]):
            worker = Worker(
                idx_screen_element,
                i,
                data[:, i],
                self.screening_ppm.value() or 1e6,
                limit_kws=limit_kws,
            )
            worker.setAutoDelete(True)
            worker.signals.finished.connect(self.threadComplete)
            worker.signals.exception.connect(self.threadFailed)
            worker.signals.result.connect(self.results.append)
            self.threadpool.start(worker)

    def finalise(self) -> None:
        if not self.threadpool.waitForDone(1000):
            logger.warning("could not remove all threads at finalise")
        self.running = False

        idx = np.array(self.results, dtype=int)
        self.screeningComplete.emit(idx[idx >= 0])

        super().accept()

    def reject(self) -> None:
        if self.running:
            self.abort()
        else:
            super().reject()
