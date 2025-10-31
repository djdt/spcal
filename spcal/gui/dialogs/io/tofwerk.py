import logging
import re
from pathlib import Path

import h5py
import numpy as np
import numpy.lib.recfunctions as rfn
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.dialogs.io.base import ImportDialogBase
from spcal.gui.widgets import (
    CheckableComboBox,
    PeriodicTableSelector,
)
from spcal.io.tofwerk import calibrate_mass_to_index, factor_extraction_to_acquisition
from spcal.npdb import db

logger = logging.getLogger(__name__)


class TofwerkIntegrationThread(QtCore.QThread):
    integrationStarted = QtCore.Signal(int)
    sampleIntegrated = QtCore.Signal()
    integrationComplete = QtCore.Signal(np.ndarray)

    def __init__(
        self,
        h5: h5py._hl.files.File,  # type: ignore
        idx: np.ndarray,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent=parent)
        peak_table = h5["PeakData"]["PeakTable"]

        mode = h5["FullSpectra"].attrs["MassCalibMode"][0]
        ps = [
            h5["FullSpectra"].attrs["MassCalibration p1"][0],
            h5["FullSpectra"].attrs["MassCalibration p2"][0],
        ]
        if mode in [2, 5]:
            ps.append(h5["FullSpectra"].attrs["MassCalibration p3"][0])

        lower = calibrate_mass_to_index(
            peak_table["lower integration limit"][idx], mode, ps
        )
        upper = calibrate_mass_to_index(
            peak_table["upper integration limit"][idx], mode, ps
        )
        self.indicies = np.stack((lower, upper + 1), axis=1)
        self.scale_factor = float(
            (
                h5["FullSpectra"].attrs["SampleInterval"][0] * 1e9
            )  # mV * index -> mV * ns
            / h5["FullSpectra"].attrs["Single Ion Signal"][0]  # mV * ns -> ions
            / factor_extraction_to_acquisition(h5)  # ions -> ions/extraction
        )

        self.tof_data = h5["FullSpectra"]["TofData"]

    def run(self) -> None:
        data = np.empty(
            (*self.tof_data.shape[:-1], self.indicies.shape[0]),
            dtype=np.float32,
        )
        self.integrationStarted.emit(data.shape[0])
        for i, sample in enumerate(self.tof_data):
            if self.isInterruptionRequested():
                return
            data[i] = np.add.reduceat(sample, self.indicies.flat, axis=-1)[..., ::2]
            self.sampleIntegrated.emit()
        data *= self.scale_factor
        self.integrationComplete.emit(data)


class TofwerkImportDialog(ImportDialogBase):
    def __init__(self, path: str | Path, parent: QtWidgets.QWidget | None = None):
        super().__init__(path, "SPCal TOFWERK Import", parent)

        # Worker doesn't work as h5py locks
        self.import_thread: TofwerkIntegrationThread | None = None
        self.progress = QtWidgets.QProgressBar()

        # Get the masses from the file
        self.h5 = h5py.File(self.file_path, "r")
        self.peak_labels = self.h5["PeakData"]["PeakTable"]["label"].astype("U256")  # type: ignore , works
        self.selected_idx = np.array([])

        re_valid = re.compile("\\[(\\d+)([A-Z][a-z]?)\\]\\+")

        isotopes = []
        other_peaks = []
        for label in self.peak_labels:
            m = re_valid.match(label)
            if m is not None:
                isotopes.append(
                    db["isotopes"][
                        (db["isotopes"]["Isotope"] == int(m.group(1)))
                        & (db["isotopes"]["Symbol"] == m.group(2))
                    ]
                )
            else:
                other_peaks.append(label)

        self.table = PeriodicTableSelector(enabled_isotopes=np.array(isotopes))
        self.table.isotopesChanged.connect(self.completeChanged)

        self.combo_other_peaks = CheckableComboBox()
        self.combo_other_peaks.addItems(other_peaks)
        self.other_peaks_item = QtGui.QStandardItem("0 Selected")

        self.combo_other_peaks.model().itemChanged.connect(
            lambda: self.other_peaks_item.setText(
                f"{len(self.combo_other_peaks.checkedItems())} Selected"
            )
        )
        self.combo_other_peaks.model().insertRow(0, self.other_peaks_item)
        self.combo_other_peaks.setCurrentIndex(0)
        if len(other_peaks) == 0:
            self.combo_other_peaks.setEnabled(False)

        self.check_force_integrate = QtWidgets.QCheckBox("Force peak integration")
        self.check_force_integrate.setToolTip(
            "Reintegrate tofdata even if peakdata exists. Slow!"
        )

        self.box_options_layout.addRow(
            "Additional Peaks:",
            self.combo_other_peaks,
        )
        self.box_options_layout.addRow(
            self.check_force_integrate,
        )

        self.layout_body.addWidget(self.table, 1)
        self.layout_body.addWidget(self.progress, 0)

        events = int(
            self.h5.attrs["NbrWrites"][0]  # type: ignore , works
            * self.h5.attrs["NbrBufs"][0]  # type: ignore , works
            * self.h5.attrs["NbrSegments"][0]  # type: ignore , works
        )
        extraction_time = float(self.h5["TimingData"].attrs["TofPeriod"][0]) * 1e-9  # type: ignore , works
        extraction_time *= factor_extraction_to_acquisition(self.h5)

        # Set info and defaults
        config = self.h5.attrs["Configuration File"].decode()  # type: ignore , works
        self.box_info_layout.addRow(
            "Configuration:", QtWidgets.QLabel(config[config.rfind("\\") + 1 :])
        )
        self.box_info_layout.addRow("Number Events:", QtWidgets.QLabel(str(events)))
        self.box_info_layout.addRow(
            "Number Integrations:", QtWidgets.QLabel(str(len(self.peak_labels)))
        )
        self.box_info_layout.addRow(
            "Event time:", QtWidgets.QLabel(f"{extraction_time * 1e-3:.4f} ms")
        )
        self.table.setFocus()

    def isComplete(self) -> bool:
        isotopes = self.table.selectedIsotopes()
        return isotopes is not None

    #
    # def importOptions(self) -> dict:
    #     single_ion_area = float(self.h5["FullSpectra"].attrs["Single Ion Signal"][0])
    #     return {
    #         "importer": "tofwerk",
    #         "path": self.file_path,
    #         "dwelltime": self.dwelltime.baseValue(),
    #         "isotopes": self.table.selectedIsotopes(),
    #         "other peaks": self.combo_other_peaks.checkedItems(),
    #         "single ion area": single_ion_area,
    #         "accumulations": factor_extraction_to_acquisition(self.h5),
    #     }

    # def dataForScreening(self, size: int) -> np.ndarray:
    #     dim_size = np.sum(self.h5["PeakData"]["PeakData"].shape[1:3])
    #     data = self.h5["PeakData"]["PeakData"][: int(size / dim_size) + 1]
    #     data = np.reshape(data, (-1, data.shape[-1]))
    #     data *= factor_extraction_to_acquisition(self.h5)
    #     return data
    #
    # def screenData(self, idx: np.ndarray, ppm: np.ndarray) -> None:
    #     _isotopes, _ppm = [], []
    #     re_valid = re.compile("\\[(\\d+)([A-Z][a-z]?)\\]\\+")
    #     for label, val in zip(self.peak_labels[idx], ppm):
    #         m = re_valid.match(label)
    #         if m is not None:
    #             _isotopes.append(
    #                 db["isotopes"][
    #                     (db["isotopes"]["Isotope"] == int(m.group(1)))
    #                     & (db["isotopes"]["Symbol"] == m.group(2))
    #                 ]
    #             )
    #             _ppm.append(val)
    #
    #     isotopes = np.asarray(_isotopes, dtype=db["isotopes"].dtype).ravel()
    #     cidx = np.asarray(_ppm)[isotopes["Preferred"] > 0]  # before isotopes
    #     isotopes = isotopes[isotopes["Preferred"] > 0]
    #     cidx = (cidx / cidx.max() * (len(viridis_32) - 1)).astype(int)
    #
    #     self.table.setSelectedIsotopes(isotopes)
    #     self.table.setIsotopeColors(isotopes, np.asarray(viridis_32)[cidx])

    # def setImportOptions(
    #     self, options: dict, path: bool = False, dwelltime: bool = True
    # ) -> None:
    #     super().setImportOptions(options, path, dwelltime)
    #     self.table.setSelectedIsotopes(options["isotopes"])
    #     self.combo_other_peaks.setCheckedItems(options["other peaks"])
    #
    def setControlsEnabled(self, enabled: bool) -> None:
        button = self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        button.setEnabled(enabled)
        self.table.setEnabled(enabled)
        self.dwelltime.setEnabled(enabled)
        self.combo_other_peaks.setEnabled(enabled)

    def accept(self) -> None:
        isotopes = self.table.selectedIsotopes()
        assert isotopes is not None
        selected_labels = [f"[{i['Isotope']}{i['Symbol']}]+" for i in isotopes]
        selected_labels.extend(self.combo_other_peaks.checkedItems())
        self.selected_idx = np.flatnonzero(np.isin(self.peak_labels, selected_labels))

        if (
            "PeakData" not in self.h5["PeakData"]  # type: ignore , works
            or self.check_force_integrate.isChecked()
        ):
            logger.warning("PeakData does not exist, integrating...")
            self.progress.setValue(0)
            self.progress.setFormat("Integrating... %p%")
            self.setControlsEnabled(False)

            self.import_thread = TofwerkIntegrationThread(
                self.h5, self.selected_idx, parent=self
            )
            self.import_thread.integrationStarted.connect(self.progress.setMaximum)
            self.import_thread.sampleIntegrated.connect(
                lambda: self.progress.setValue(self.progress.value() + 1)
            )
            self.import_thread.integrationComplete.connect(self.finalise)
            self.import_thread.start()
            # Peaks do not exist, we must integrate ourselves.
        else:
            data = self.h5["PeakData"]["PeakData"][..., self.selected_idx]  # type: ignore , works
            self.finalise(data)

    def finalise(self, data: np.ndarray) -> None:
        data *= factor_extraction_to_acquisition(self.h5)
        data = rfn.unstructured_to_structured(
            data.reshape(-1, data.shape[-1]), names=self.peak_labels[self.selected_idx]
        )
        options = self.importOptions()
        self.dataImported.emit(data, options)

        logger.info(
            "TOFWERK instruments data loaded from "
            f"{self.file_path} ({data.size} events)."
        )
        super().accept()

    def reject(self) -> None:
        if self.import_thread is not None and self.import_thread.isRunning():
            self.import_thread.requestInterruption()
            self.progress.reset()
            self.setControlsEnabled(True)
        else:
            super().reject()
