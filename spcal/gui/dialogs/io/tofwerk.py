import logging
import re
from pathlib import Path

import h5py
from PySide6 import QtWidgets

from spcal.datafile import SPCalDataFile, SPCalTOFWERKDataFile
from spcal.gui.dialogs.io.base import ImportDialogBase
from spcal.gui.widgets import PeriodicTableSelector
from spcal.io.tofwerk import factor_extraction_to_acquisition
from spcal.isotope import ISOTOPE_TABLE

logger = logging.getLogger(__name__)


class TofwerkImportDialog(ImportDialogBase):
    def __init__(
        self,
        path: str | Path,
        existing_file: SPCalDataFile,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(path, "SPCal TOFWERK Import", parent)

        # Get the masses from the file
        self.h5 = h5py.File(self.file_path, "r")
        self.peak_labels = self.h5["PeakData"]["PeakTable"]["label"].astype("U256")  # type: ignore , works

        re_valid = re.compile("\\[(\\d+)([A-Z][a-z]?)\\]\\+")

        isotopes = []
        other_peaks = []
        for label in self.peak_labels:
            m = re_valid.match(label)
            if m is not None:
                isotopes.append(ISOTOPE_TABLE[(m.group(2), int(m.group(1)))])
            else:
                other_peaks.append(label)

        self.table = PeriodicTableSelector(enabled_isotopes=isotopes)
        self.table.isotopesChanged.connect(self.completeChanged)

        self.layout_body.addWidget(self.table, 1)

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
            "Event time:", QtWidgets.QLabel(f"{extraction_time * 1e3:.4f} ms")
        )
        self.table.setFocus()

    def isComplete(self) -> bool:
        isotopes = self.table.selectedIsotopes()
        return len(isotopes) > 0

    def setControlsEnabled(self, enabled: bool):
        button = self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        button.setEnabled(enabled)
        self.table.setEnabled(enabled)

    def accept(self):
        if (
            "PeakData" not in self.h5["PeakData"]  # type: ignore , works
        ):
            button = QtWidgets.QMessageBox.question(
                self,
                "Integrate Data?",
                "The provided .h5 file is missing PeakData. Integrating may take a long time, proceed?",
            )
            if button == QtWidgets.QMessageBox.StandardButton.No:
                return

        data_file = SPCalTOFWERKDataFile.load(self.file_path)

        logger.info(
            f"TOFWERK instruments data loaded from {self.file_path} ({data_file.num_events} events)."
        )

        data_file.selected_isotopes = self.table.selectedIsotopes()
        self.dataImported.emit(data_file)
        super().accept()
