import json
import logging
import re
from pathlib import Path

import h5py
import numpy as np
import numpy.lib.recfunctions as rfn
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.datafile import SPCalDataFile, SPCalNuDataFile, SPCalTextDataFile
from spcal.gui.graphs import viridis_32
from spcal.gui.modelviews import CheckableHeaderView
from spcal.gui.widgets import (
    CheckableComboBox,
    ElidedLabel,
    PeriodicTableSelector,
    UnitsWidget,
)
from spcal.io import nu
from spcal.io.text import (
    guess_text_parameters,
    iso_time_to_float_seconds,
)
from spcal.io.tofwerk import calibrate_mass_to_index, factor_extraction_to_acquisition
from spcal.npdb import db
from spcal.siunits import time_units

logger = logging.getLogger(__name__)


class _ImportDialogBase(QtWidgets.QDialog):
    dataImported = QtCore.Signal(SPCalDataFile, list)
    forbidden_names = ["Overlay"]

    def __init__(
        self,
        path: str | Path,
        title: str,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)

        self.file_path = Path(path)
        self.setWindowTitle(f"{title}: {self.file_path.name}")

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(
            False
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        box_info = QtWidgets.QGroupBox("Information")
        self.box_info_layout = QtWidgets.QFormLayout()
        self.box_info_layout.addRow(
            "File Path:", ElidedLabel(str(self.file_path.absolute()))
        )
        box_info.setLayout(self.box_info_layout)

        box_options = QtWidgets.QGroupBox("Import Options")
        self.box_options_layout = QtWidgets.QFormLayout()
        box_options.setLayout(self.box_options_layout)

        box_layout = QtWidgets.QHBoxLayout()
        box_layout.addWidget(box_info, 1)
        box_layout.addWidget(box_options, 1)

        self.layout_body = QtWidgets.QVBoxLayout()

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(box_layout)
        layout.addLayout(self.layout_body)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

    def completeChanged(self) -> None:
        complete = self.isComplete()
        self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(
            complete
        )

    def isComplete(self) -> bool:
        return True


class TextImportDialog(_ImportDialogBase):
    HEADER_COUNT = 20
    DELIMITERS = {",": ",", ";": ";", " ": "Space", "\t": "Tab"}

    def __init__(
        self,
        path: str | Path,
        existing_file: SPCalDataFile | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(path, "SPCal Text Import", parent)

        self.file_lines = self.file_path.open("r").readlines()
        # Guess the delimiter, skip rows and count from header

        delimiter, first_data_line, column_count = guess_text_parameters(
            self.file_lines[: self.HEADER_COUNT]
        )
        cps = any(
            "cps" in line.lower() for line in self.file_lines[: self.HEADER_COUNT]
        )
        event_time, override = None, None

        if delimiter == "":
            delimiter = ","

        if isinstance(existing_file, SPCalTextDataFile):
            delimiter = existing_file.delimter
            first_data_line = existing_file.skip_row
            cps = existing_file.cps
            event_time = existing_file.event_time
            override = existing_file.override_event_time

        self.table = QtWidgets.QTableWidget()
        self.table.itemChanged.connect(self.completeChanged)
        self.table.setMinimumSize(800, 400)
        self.table.setColumnCount(column_count)
        self.table.setRowCount(self.HEADER_COUNT)
        self.table.setFont(QtGui.QFont("Courier"))
        self.table_header = CheckableHeaderView(QtCore.Qt.Orientation.Horizontal)
        self.table.setHorizontalHeader(self.table_header)
        self.table_header.checkStateChanged.connect(self.updateTableUseColumns)

        self.box_info_layout.addRow(
            "Line Count:", QtWidgets.QLabel(str(len(self.file_lines)))
        )

        self.event_time = UnitsWidget(
            time_units,
            base_value=event_time,
            default_unit="ms",
            validator=QtGui.QDoubleValidator(0.0, 10.0, 10),
        )
        self.event_time.baseValueChanged.connect(self.completeChanged)
        self.event_time.setEnabled(False)

        self.override_event_time = QtWidgets.QCheckBox("Override")
        self.override_event_time.checkStateChanged.connect(
            self.overrideEventTimeChanged
        )
        self.override_event_time.setChecked(override is not None)

        self.combo_intensity_units = QtWidgets.QComboBox()
        self.combo_intensity_units.addItems(["Counts", "CPS"])
        if cps:
            self.combo_intensity_units.setCurrentText("CPS")

        self.combo_delimiter = QtWidgets.QComboBox()
        self.combo_delimiter.addItems(list(self.DELIMITERS.values()))
        self.combo_delimiter.setCurrentIndex(
            list(self.DELIMITERS.keys()).index(delimiter)
        )
        self.combo_delimiter.currentIndexChanged.connect(self.fillTable)

        self.spinbox_first_line = QtWidgets.QSpinBox()
        self.spinbox_first_line.setRange(1, self.HEADER_COUNT - 1)
        self.spinbox_first_line.setValue(first_data_line)
        self.spinbox_first_line.valueChanged.connect(self.updateTableUseColumns)

        layout_event_time = QtWidgets.QHBoxLayout()
        layout_event_time.addWidget(self.event_time, 1)
        layout_event_time.addWidget(self.override_event_time, 0)

        self.box_options_layout.addRow("Event time:", layout_event_time)
        self.box_options_layout.addRow("Intensity units:", self.combo_intensity_units)
        self.box_options_layout.addRow("Delimiter:", self.combo_delimiter)
        self.box_options_layout.addRow("Import from row:", self.spinbox_first_line)

        self.fillTable()

        self.guessIsotopesFromTable()

        self.layout_body.addWidget(self.table)

    def isComplete(self) -> bool:
        return not self.event_time.isEnabled() or self.event_time.hasAcceptableInput()

    def overrideEventTimeChanged(self) -> None:
        self.event_time.setEnabled(self.override_event_time.isChecked())

    def delimiter(self) -> str:
        delimiter = self.combo_delimiter.currentText()
        if delimiter == "Space":
            delimiter = " "
        elif delimiter == "Tab":
            delimiter = "\t"
        return delimiter

    def useColumns(self) -> list[int]:
        return [
            k
            for k, v in self.table_header._checked.items()
            if v == QtCore.Qt.CheckState.Checked
        ]

    def names(self) -> list[str]:
        names = []
        for c in range(self.table.columnCount()):
            item = self.table.item(self.spinbox_first_line.value() - 1, c)
            if item is not None:
                names.append(item.text().replace(" ", "_"))
        return names

    def selectedNames(self) -> list[str]:
        names = []
        for c in self.useColumns():
            item = self.table.item(self.spinbox_first_line.value() - 1, c)
            if item is not None:
                names.append(item.text().replace(" ", "_"))
        return names

    def fillTable(self) -> None:
        lines = [
            line.split(self.delimiter())
            for line in self.file_lines[: self.HEADER_COUNT]
        ]
        col_count = max(len(line) for line in lines)
        self.table.setColumnCount(col_count)

        for row, line in enumerate(lines):
            line.extend([""] * (col_count - len(line)))
            for col, text in enumerate(line):
                item = QtWidgets.QTableWidgetItem(text.strip().replace(" ", "_"))
                self.table.setItem(row, col, item)

        self.table.resizeColumnsToContents()
        self.updateTableUseColumns()

        if self.event_time.value() is None:
            try:
                val, unit = self.guessEventTimeFromTable()
                self.event_time.setUnit(unit)
                self.event_time.setValue(val)
            except StopIteration:
                pass

    def updateTableUseColumns(self) -> None:
        header_row = self.spinbox_first_line.value() - 1
        for row in range(self.table.rowCount()):
            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                if item is None:
                    continue
                if row != header_row:
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                else:
                    item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
                if row < header_row or col not in self.useColumns():
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEnabled)
                else:
                    item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsEnabled)

    def guessIsotopesFromTable(self) -> None:
        columns = []
        header_row = self.spinbox_first_line.value() - 1
        for col in range(self.table.columnCount()):
            text = self.table.item(header_row, col).text().lower()
            if not any(x in text for x in ["time", "index"]):
                columns.append(col)

        for col in columns:
            self.table_header.setCheckState(col, QtCore.Qt.CheckState.Checked)

    def guessEventTimeFromTable(self) -> tuple[float, str]:
        header_row = self.spinbox_first_line.value() - 1
        for col in range(self.table.columnCount()):
            text = self.table.item(header_row, col).text().lower()
            if "time" in text:
                m = re.search("[\\(\\[]([nmuµ]s)[\\]\\)]", text.lower())
                unit = "s"
                if m is not None:
                    if m.group(1) == "ms":
                        unit = "ms"
                    elif m.group(1) in ["us", "µs"]:
                        unit = "µs"
                    elif m.group(1) == "ns":
                        unit = "ns"

                time_texts = [
                    self.table.item(row, col).text().replace(",", ".")
                    for row in range(header_row + 1, self.table.rowCount())
                ]
                if "00:" in time_texts[0]:
                    times = [iso_time_to_float_seconds(tt) for tt in time_texts]
                else:
                    times = [float(tt) for tt in time_texts]

                return float(np.mean(np.diff(times))), unit
        raise StopIteration

    # def setImportOptions(
    #     self, options: dict, path: bool = False, dwelltime: bool = True
    # ) -> None:
    #     super().setImportOptions(options, path, dwelltime)
    #     delimiter = options["delimiter"]
    #     if delimiter == " ":
    #         delimiter = "Space"
    #     elif delimiter == "\t":
    #         delimiter = "Tab"
    #
    #     spinbox_first_line = self.spinbox_first_line.value()
    #     self.spinbox_first_line.setValue(options["first line"])
    #
    #     self.table_header._checked = {}
    #     for col in options["columns"]:
    #         if col < self.table.columnCount():
    #             self.table_header.setCheckState(col, QtCore.Qt.CheckState.Checked)
    #
    #     self.spinbox_first_line.setValue(spinbox_first_line)
    #     self.combo_delimiter.setCurrentText(delimiter)
    #     self.combo_intensity_units.setCurrentText("CPS" if options["cps"] else "Counts")
    #
    #     for oldname, name in options["names"].items():
    #         for col in range(self.table.columnCount()):
    #             item = self.table.item(self.spinbox_first_line.value() - 1, col)
    #             if item is not None and item.text() == oldname:
    #                 item.setText(name)

    def accept(self) -> None:
        data_file = SPCalTextDataFile.load(
            self.file_path,
            delimiter=self.delimiter(),
            skip_rows=self.spinbox_first_line.value(),
            cps=self.combo_intensity_units.currentText() == "CPS",
            rename_fields=self.names(),
            override_event_time=self.event_time.value()
            if self.override_event_time.isChecked()
            else None,
        )

        self.dataImported.emit(data_file, self.selectedNames())
        logger.info(
            f"Text data loaded from {self.file_path} ({data_file.num_events} events)."
        )
        super().accept()


class NuReadIntegsThread(QtCore.QThread):
    integRead = QtCore.Signal(int)

    def __init__(
        self,
        path: Path,
        index: list[dict],
        cycle: int | None = None,
        segment: int | None = None,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent)
        self.path = path
        self.index = index
        self.cyc_number = cycle
        self.seg_number = segment

        self.datas: list[np.ndarray] = []

    def run(self):
        for idx in self.index:
            if self.isInterruptionRequested():
                break
            binary_path = self.path.joinpath(f"{idx['FileNum']}.integ")
            if binary_path.exists():
                data = nu.read_integ_binary(
                    binary_path,
                    idx["FirstCycNum"],
                    idx["FirstSegNum"],
                    idx["FirstAcqNum"],
                )
                if self.cyc_number is not None:
                    data = data[data["cyc_number"] == self.cyc_number]
                if self.seg_number is not None:
                    data = data[data["seg_number"] == self.seg_number]
                self.datas.append(data)
            else:
                logger.warning(  # pragma: no cover, missing files
                    f"collect_data_from_index: missing data file {idx['FileNum']}.integ, skipping"
                )
            self.integRead.emit(idx)


class NuImportDialog(_ImportDialogBase):
    def __init__(
        self,
        path: str | Path,
        existing_file: SPCalDataFile | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(path, "SPCal Nu Instruments Import", parent)

        self.progress = QtWidgets.QProgressBar()
        self.import_thread: NuReadIntegsThread | None = None

        with self.file_path.joinpath("run.info").open("r") as fp:
            self.info = json.load(fp)
        with self.file_path.joinpath("integrated.index").open("r") as fp:
            self.index = json.load(fp)
        with self.file_path.joinpath("autob.index").open("r") as fp:
            self.autob_index = json.load(fp)

        # read first integ
        data: np.ndarray | None = None
        for idx in self.index:
            first_path = self.file_path.joinpath(f"{idx['FileNum']}.integ")
            if first_path.exists():
                data = nu.read_integ_binary(
                    first_path,
                    idx["FirstCycNum"],
                    idx["FirstSegNum"],
                    idx["FirstAcqNum"],
                )
                break
        if data is None:
            raise ValueError("NuImportDialog: no valid integ files found.")

        self.signals = data["result"]["signal"] / self.info["AverageSingleIonArea"]
        self.masses = nu.masses_from_integ(
            data[0],
            self.info["MassCalCoefficients"],
            self.segmentDelays(),
        )[0]

        unit_masses = np.round(self.masses).astype(int)
        isotopes = db["isotopes"][np.isin(db["isotopes"]["Isotope"], unit_masses)]

        self.table = PeriodicTableSelector(enabled_isotopes=isotopes)
        if isinstance(existing_file, SPCalNuDataFile):
            self.table.setSelectedIsotopes(existing_file.isotopes)
        self.table.isotopesChanged.connect(self.completeChanged)

        self.layout_body.addWidget(self.table, 1)
        self.layout_body.addWidget(self.progress, 0)

        # Set info and defaults
        method = self.info["MethodFile"]
        self.box_info_layout.addRow(
            "Method:", QtWidgets.QLabel(method[method.rfind("\\") + 1 :])
        )
        self.box_info_layout.addRow(
            "Events:",
            QtWidgets.QLabel(str(self.info["ActualRecordLength"])),
        )
        # event_time = UnitsWidget(
        #     default_unit="ms",
        #     units=time_units,
        #     base_value=nu.get_dwelltime_from_info(self.info),
        # )
        # event_time.setReadOnly(True)
        # self.box_info_layout.addRow(
        #     "Event time:",
        #     event_time,
        # )
        self.box_info_layout.addRow(
            "Event time:",
            QtWidgets.QLabel(f"{nu.eventtime_from_info(self.info) * 1000} ms"),
        )
        self.box_info_layout.addRow(
            "Integrations:",
            QtWidgets.QLabel(str(len(self.info["IntegrationRegions"]))),
        )

        self.cycle_number = QtWidgets.QSpinBox()
        self.cycle_number.setRange(0, self.info["CyclesWritten"])
        self.cycle_number.setValue(0)
        self.cycle_number.setSpecialValueText("All")

        self.segment_number = QtWidgets.QSpinBox()
        self.segment_number.setRange(0, len(self.info["SegmentInfo"]))
        self.segment_number.setValue(0)
        self.segment_number.setSpecialValueText("All")

        # self.file_number = QtWidgets.QSpinBox()
        # self.file_number.setRange(1, len(self.index))
        # self.file_number.setValue(len(self.index))

        self.checkbox_blanking = QtWidgets.QCheckBox("Apply auto-blanking.")
        self.checkbox_blanking.setChecked(True)

        self.box_options_layout.addRow("Cycle:", self.cycle_number)
        self.box_options_layout.addRow("Segment:", self.segment_number)
        # self.box_options.layout().addRow("Max file:", self.file_number)
        self.box_options_layout.addRow(self.checkbox_blanking)

        self.table.setFocus()

    # def dataForScreening(self, size: int) -> np.ndarray:
    #     options = self.importOptions()
    #     integ_size = self.signals.shape[0]
    #     _, data, _ = nu.read_nu_directory(
    #         options["path"],
    #         max_integ_files=int(size / integ_size) + 1,
    #         autoblank=False,
    #         cycle=options["cycle"],
    #         segment=options["segment"],
    #     )
    #     return data
    #
    # def screenData(self, idx: np.ndarray, ppm: np.ndarray) -> None:
    #     masses = self.masses[idx]
    #     unit_masses = np.round(masses).astype(int)
    #     isotopes = db["isotopes"][np.isin(db["isotopes"]["Isotope"], unit_masses)]
    #     isotopes = isotopes[isotopes["Preferred"] > 0]  # limit to best isotopes
    #     self.table.setSelectedIsotopes(isotopes)
    #
    #     idx = np.argsort(unit_masses)
    #     ppm, unit_masses = ppm[idx], unit_masses[idx]  # sort by mass
    #
    #     idx = np.searchsorted(unit_masses, isotopes["Isotope"], side="right") - 1
    #     cidx = (ppm[idx] / ppm[idx].max() * (len(viridis_32) - 1)).astype(int)
    #
    #     self.table.setIsotopeColors(isotopes, np.asarray(viridis_32)[cidx])

    def segmentDelays(self) -> dict[int, float]:
        return {
            s["Num"]: s["AcquisitionTriggerDelayNs"] for s in self.info["SegmentInfo"]
        }

    def isComplete(self) -> bool:
        return self.table.selectedIsotopes() is not None

    def setControlsEnabled(self, enabled: bool) -> None:
        button = self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        button.setEnabled(enabled)
        self.table.setEnabled(enabled)

    def updateProgress(self):
        self.progress.setValue(self.progress.value() + 1)

    def accept(self) -> None:
        self.setControlsEnabled(False)

        self.progress.setValue(0)
        self.progress.setMaximum(len(self.index))

        cycle = self.cycle_number.value()
        if cycle == 0:
            cycle = None
        segment = self.segment_number.value()
        if segment == 0:
            segment = None

        self.import_thread = NuReadIntegsThread(
            self.file_path, self.index, cycle, segment
        )
        self.import_thread.integRead.connect(self.updateProgress)
        self.import_thread.finished.connect(self.finalise)
        self.import_thread.start()

    def finalise(self) -> None:
        if self.import_thread is None or self.import_thread.isInterruptionRequested():
            self.progress.reset()
            self.setControlsEnabled(True)
            return

        # Get masses from data
        integs = self.import_thread.datas

        segment_delays = {
            s["Num"]: s["AcquisitionTriggerDelayNs"] for s in self.info["SegmentInfo"]
        }

        accumulations = self.info["NumAccumulations1"] * self.info["NumAccumulations2"]

        # Get masses from data
        masses = nu.masses_from_integ(
            integs[0], self.info["MassCalCoefficients"], segment_delays
        )[0]
        signals = nu.signals_from_integs(integs, accumulations)

        times = nu.times_from_integs(integs, self.info) * 1e-9

        # if not raw:
        signals /= self.info["AverageSingleIonArea"]

        # Blank out overrange regions
        if self.checkbox_blanking.isChecked():
            autobs = np.concatenate(
                nu.read_binaries_in_index(
                    self.file_path,
                    self.autob_index,
                    "autob",
                    nu.read_autob_binary,
                    cyc_number=self.import_thread.cyc_number,
                    seg_number=self.import_thread.seg_number,
                )
            )
            signals = nu.apply_autoblanking(
                autobs,
                signals,
                masses,
                accumulations,
                self.info["BlMassCalStartCoef"],
                self.info["BlMassCalEndCoef"],
            )

        selected_isotopes = self.table.selectedIsotopes()
        assert selected_isotopes is not None
        isotopes = [f"{i['Isotope']}{i['Symbol']}" for i in selected_isotopes]
        self.dataImported.emit(
            SPCalNuDataFile(self.file_path, signals, times, masses, self.info),
            isotopes,
        )
        super().accept()

    def reject(self) -> None:
        if self.import_thread is not None:
            self.import_thread.requestInterruption()
        else:
            super().reject()


class TofwerkIntegrationThread(QtCore.QThread):
    integrationStarted = QtCore.Signal(int)
    sampleIntegrated = QtCore.Signal()
    integrationComplete = QtCore.Signal(np.ndarray)

    def __init__(
        self,
        h5: h5py._hl.files.File,
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


class TofwerkImportDialog(_ImportDialogBase):
    def __init__(self, path: str | Path, parent: QtWidgets.QWidget | None = None):
        super().__init__(path, "SPCal TOFWERK Import", parent)

        # Worker doesn't work as h5py locks
        self.thread: TofwerkIntegrationThread | None = None
        self.progress = QtWidgets.QProgressBar()

        # Get the masses from the file
        self.h5 = h5py.File(self.file_path, "r")
        self.peak_labels = self.h5["PeakData"]["PeakTable"]["label"].astype("U256")
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

        self.box_options.layout().addRow(
            "Additional Peaks:",
            self.combo_other_peaks,
        )
        self.box_options.layout().addRow(
            self.check_force_integrate,
        )

        self.layout_body.addWidget(self.table, 1)
        self.layout_body.addWidget(self.progress, 0)

        events = int(
            self.h5.attrs["NbrWrites"][0]
            * self.h5.attrs["NbrBufs"][0]
            * self.h5.attrs["NbrSegments"][0]
        )
        extraction_time = float(self.h5["TimingData"].attrs["TofPeriod"][0]) * 1e-9

        # Set info and defaults
        config = self.h5.attrs["Configuration File"].decode()
        self.box_info.layout().addRow(
            "Configuration:", QtWidgets.QLabel(config[config.rfind("\\") + 1 :])
        )
        self.box_info.layout().addRow("Number Events:", QtWidgets.QLabel(str(events)))
        self.box_info.layout().addRow(
            "Number Integrations:", QtWidgets.QLabel(str(len(self.peak_labels)))
        )
        self.dwelltime.setBaseValue(
            np.around(
                extraction_time * factor_extraction_to_acquisition(self.h5), 9
            )  # nearest us
        )
        self.dwelltime.setBestUnit()
        self.table.setFocus()

    def isComplete(self) -> bool:
        isotopes = self.table.selectedIsotopes()
        return isotopes is not None and self.dwelltime.hasAcceptableInput()

    def importOptions(self) -> dict:
        single_ion_area = float(self.h5["FullSpectra"].attrs["Single Ion Signal"][0])
        return {
            "importer": "tofwerk",
            "path": self.file_path,
            "dwelltime": self.dwelltime.baseValue(),
            "isotopes": self.table.selectedIsotopes(),
            "other peaks": self.combo_other_peaks.checkedItems(),
            "single ion area": single_ion_area,
            "accumulations": factor_extraction_to_acquisition(self.h5),
        }

    def dataForScreening(self, size: int) -> np.ndarray:
        dim_size = np.sum(self.h5["PeakData"]["PeakData"].shape[1:3])
        data = self.h5["PeakData"]["PeakData"][: int(size / dim_size) + 1]
        data = np.reshape(data, (-1, data.shape[-1]))
        data *= factor_extraction_to_acquisition(self.h5)
        return data

    def screenData(self, idx: np.ndarray, ppm: np.ndarray) -> None:
        _isotopes, _ppm = [], []
        re_valid = re.compile("\\[(\\d+)([A-Z][a-z]?)\\]\\+")
        for label, val in zip(self.peak_labels[idx], ppm):
            m = re_valid.match(label)
            if m is not None:
                _isotopes.append(
                    db["isotopes"][
                        (db["isotopes"]["Isotope"] == int(m.group(1)))
                        & (db["isotopes"]["Symbol"] == m.group(2))
                    ]
                )
                _ppm.append(val)

        isotopes = np.asarray(_isotopes, dtype=db["isotopes"].dtype).ravel()
        cidx = np.asarray(_ppm)[isotopes["Preferred"] > 0]  # before isotopes
        isotopes = isotopes[isotopes["Preferred"] > 0]
        cidx = (cidx / cidx.max() * (len(viridis_32) - 1)).astype(int)

        self.table.setSelectedIsotopes(isotopes)
        self.table.setIsotopeColors(isotopes, np.asarray(viridis_32)[cidx])

    def setImportOptions(
        self, options: dict, path: bool = False, dwelltime: bool = True
    ) -> None:
        super().setImportOptions(options, path, dwelltime)
        self.table.setSelectedIsotopes(options["isotopes"])
        self.combo_other_peaks.setCheckedItems(options["other peaks"])

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
            "PeakData" not in self.h5["PeakData"]
            or self.check_force_integrate.isChecked()
        ):
            logger.warning("PeakData does not exist, integrating...")
            self.progress.setValue(0)
            self.progress.setFormat("Integrating... %p%")
            self.setControlsEnabled(False)

            self.thread = TofwerkIntegrationThread(
                self.h5, self.selected_idx, parent=self
            )
            self.thread.integrationStarted.connect(self.progress.setMaximum)
            self.thread.sampleIntegrated.connect(
                lambda: self.progress.setValue(self.progress.value() + 1)
            )
            self.thread.integrationComplete.connect(self.finalise)
            self.thread.start()
            # Peaks do not exist, we must integrate ourselves.
        else:
            data = self.h5["PeakData"]["PeakData"][..., self.selected_idx]
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
        if self.thread is not None and self.thread.isRunning():
            self.thread.requestInterruption()
            self.progress.reset()
            self.setControlsEnabled(True)
        else:
            super().reject()


if __name__ == "__main__":
    app = QtWidgets.QApplication()

    dlg = TextImportDialog(
        Path(
            "/home/tom/Documents/python/spcal/tests/data/text/tofwerk_export_au_bg.csv"
        )
    )
    dlg.open()
    app.exec()
