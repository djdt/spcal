import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import numpy.lib.recfunctions as rfn
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.widgets import PeriodicTableSelector, UnitsWidget
from spcal.io.nu import get_masses_from_nu_data, read_nu_integ_binary
from spcal.npdb import db
from spcal.siunits import time_units
from spcal.util import Worker

logger = logging.getLogger(__name__)

# Todo, info at the top


class NuImportDialog(QtWidgets.QDialog):
    dataImported = QtCore.Signal(np.ndarray, dict)

    def __init__(self, path: str | Path, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)

        self.file_path = Path(path)
        self.setWindowTitle(f"SPCal Nu Instruments Import: {self.file_path.name}")

        with self.file_path.joinpath("run.info").open("r") as fp:
            self.info = json.load(fp)
        with self.file_path.joinpath("integrated.index").open("r") as fp:
            self.index = json.load(fp)

        # read first integ
        data = read_nu_integ_binary(
            self.file_path.joinpath(f"{self.index[0]['FileNum']}.integ"),
            self.index[0]["FirstCycNum"],
            self.index[0]["FirstSegNum"],
            self.index[0]["FirstAcqNum"],
        )

        self.masses = get_masses_from_nu_data(
            data[0],
            self.info["MassCalCoefficients"],
            self.segment_delays,
        )[0]

        unit_masses = np.round(self.masses).astype(int)
        isotopes = db["isotopes"][np.isin(db["isotopes"]["Isotope"], unit_masses)]

        self.dwelltime = UnitsWidget(
            time_units,
            default_unit="ms",
            validator=QtGui.QDoubleValidator(0.0, 10.0, 10),
        )
        self.dwelltime.setBaseValue(
            self.info["SegmentInfo"][0]["AcquisitionPeriodNs"] * 1e-9
        )
        self.dwelltime.valueChanged.connect(self.completeChanged)

        self.table = PeriodicTableSelector(enabled_isotopes=isotopes)
        self.table.isotopesChanged.connect(self.completeChanged)

        self.progress = QtWidgets.QProgressBar()
        self.aborted = False

        self.threadpool = QtCore.QThreadPool()
        self.results: List[Tuple[int, np.ndarray]] = []

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        self.button_box.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(False)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        box_data = QtWidgets.QGroupBox("Data Options")
        box_data.setLayout(QtWidgets.QFormLayout())
        box_data.layout().addRow("Dwelltime:", self.dwelltime)

        box_table = QtWidgets.QGroupBox("Selected Isotopes")
        box_table.setLayout(QtWidgets.QVBoxLayout())
        box_table.layout().addWidget(self.table)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(box_data, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        layout.addWidget(box_table, 1)
        layout.addWidget(self.progress, 0)
        layout.addWidget(self.button_box, 0)
        self.setLayout(layout)

    @property
    def segment_delays(self) -> Dict[int, float]:
        return {
            s["Num"]: s["AcquisitionTriggerDelayNs"] for s in self.info["SegmentInfo"]
        }

    def advanceProgress(self) -> None:
        self.progress.setValue(self.progress.value() + 1)

    def isComplete(self) -> bool:
        isotopes = self.table.selectedIsotopes()
        return isotopes is not None and self.dwelltime.hasAcceptableInput()

    def completeChanged(self) -> None:
        complete = self.isComplete()
        self.button_box.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(complete)

    def importOptions(self) -> dict:
        return {
            "importer": "nu",
            "path": self.file_path,
            "dwelltime": self.dwelltime.baseValue(),
            "masses": self.masses,
            "selectedIsotopes": self.table.selectedIsotopes(),
            # "massCalCoefficients": self.info["MassCalCoefficients"],
            # "segmentDelays": self.segment_delays,
        }

    def setControlsEnabled(self, enabled: bool) -> None:
        button = self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        button.setEnabled(enabled)
        self.table.setEnabled(enabled)
        self.dwelltime.setEnabled(enabled)

    def abort(self) -> None:
        self.aborted = True
        self.threadpool.clear()
        self.threadpool.waitForDone()
        self.progress.reset()

        self.setControlsEnabled(True)

    def threadComplete(self) -> None:
        if self.aborted:
            return

        self.advanceProgress()
        if self.progress.value() == self.progress.maximum():
            self.finaliseImport()

    def threadFailed(self, exception: Exception) -> None:
        if self.aborted:
            return

        logger.exception(exception)
        self.abort()

        msg = QtWidgets.QMessageBox(
            QtWidgets.QMessageBox.Warning,
            "Import Failed",
            "Failed to read integ binary.",
            parent=self,
        )
        msg.exec()

    def accept(self) -> None:
        def read_signals(path: Path):
            data = read_nu_integ_binary(path)
            return data["result"]["signal"]

        self.setControlsEnabled(False)

        self.aborted = False
        self.progress.setMaximum(len(self.index))
        self.progress.setValue(1)
        self.results.clear()

        for idx in self.index:
            worker = Worker(
                read_signals,
                self.file_path.joinpath(f"{idx['FileNum']}.integ"),
            )
            worker.signals.finished.connect(self.threadComplete)
            worker.signals.exception.connect(self.threadFailed)
            worker.signals.result.connect(
                lambda r: self.results.append((idx["FileNum"], r))
            )
            self.threadpool.start(worker)

    def reject(self) -> None:
        if self.threadpool.activeThreadCount() > 0:
            self.abort()
        else:
            super().reject()

    def finaliseImport(self) -> None:
        self.threadpool.waitForDone()

        signals = np.concatenate(
            [result[1] for result in sorted(self.results, key=lambda r: r[0])]
        )
        isotopes = self.table.selectedIsotopes()
        assert isotopes is not None

        diffs = isotopes["Mass"].reshape(1, -1) - self.masses.reshape(-1, 1)
        idx = np.argmin(np.abs(diffs), axis=0)

        dtype = np.dtype(
            {
                "names": [f"{i['Symbol']}{i['Isotope']}" for i in isotopes],
                "formats": [np.float32 for _ in idx],
            }
        )
        data = rfn.unstructured_to_structured(signals[:, idx], dtype=dtype)
        self.dataImported.emit(data, self.importOptions())

        super().accept()