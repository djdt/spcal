import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import numpy.lib.recfunctions as rfn
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.util import Worker, create_action
from spcal.gui.widgets import (
    CheckableComboBox,
    ElidedLabel,
    PeriodicTableSelector,
    UnitsWidget,
    ValueWidget,
)
from spcal.io.nu import (
    get_dwelltime_from_info,
    get_masses_from_nu_data,
    read_nu_integ_binary,
    select_nu_signals,
)
from spcal.io.text import read_single_particle_file
from spcal.io.tofwerk import calibrate_mass_to_index, factor_extraction_to_acquisition
from spcal.nontarget import non_target_screen
from spcal.npdb import db
from spcal.siunits import time_units

logger = logging.getLogger(__name__)


class _ImportDialogBase(QtWidgets.QDialog):
    dataImported = QtCore.Signal(np.ndarray, dict)
    forbidden_names = ["Overlay"]

    def __init__(
        self, path: str | Path, title: str, parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(parent)

        self.action_nontarget_screen = create_action(
            "view-filter",
            "Screen",
            "Select isotopes using a non-targetted screening approach. "
            "Those with signals above the chosen ppm are selected.",
            self.screenData,
        )

        self.file_path = Path(path)
        self.setWindowTitle(f"{title}: {self.file_path.name}")

        self.dwelltime = UnitsWidget(
            time_units,
            default_unit="ms",
            validator=QtGui.QDoubleValidator(0.0, 10.0, 10),
        )
        self.dwelltime.baseValueChanged.connect(self.completeChanged)

        self.screening_ppm = ValueWidget(
            1000, validator=QtGui.QIntValidator(0, 1000000), format=".0f"
        )

        self.button_screen = QtWidgets.QToolButton()
        self.button_screen.setDefaultAction(self.action_nontarget_screen)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        self.button_box.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(False)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        self.box_info = QtWidgets.QGroupBox("Information")
        self.box_info.setLayout(QtWidgets.QFormLayout())

        self.box_info.layout().addRow(
            "File Path:", ElidedLabel(str(self.file_path.absolute()))
        )

        screen_layout = QtWidgets.QHBoxLayout()
        screen_layout.addWidget(self.screening_ppm, 1)
        screen_layout.addWidget(QtWidgets.QLabel("ppm"), 0)
        screen_layout.addWidget(
            self.button_screen, 0, QtCore.Qt.AlignmentFlag.AlignRight
        )

        self.box_options = QtWidgets.QGroupBox("Import Options")
        self.box_options.setLayout(QtWidgets.QFormLayout())
        self.box_options.layout().addRow("Dwelltime:", self.dwelltime)
        self.box_options.layout().addRow("Non-target screen:", screen_layout)

        box_layout = QtWidgets.QHBoxLayout()
        box_layout.addWidget(self.box_info, 1)
        box_layout.addWidget(self.box_options, 1)

        self.layout_body = QtWidgets.QVBoxLayout()

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(box_layout)
        layout.addLayout(self.layout_body)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

    def completeChanged(self) -> None:
        complete = self.isComplete()
        self.button_box.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(complete)

    def isComplete(self) -> bool:
        return True

    def importOptions(self) -> dict:
        raise NotImplementedError

    def setImportOptions(
        self, options: dict, path: bool = False, dwelltime: bool = True
    ) -> None:
        raise NotImplementedError

    def screenData(self) -> None:
        raise NotImplementedError


class TextImportDialog(_ImportDialogBase):
    dataImported = QtCore.Signal(np.ndarray, dict)

    def __init__(self, path: str | Path, parent: QtWidgets.QWidget | None = None):
        super().__init__(path, "SPCal Text Import", parent)

        header_row_count = 10

        self.file_header = [
            x for _, x in zip(range(header_row_count), self.file_path.open("r"))
        ]

        first_data_line = 0
        for line in self.file_header:
            try:
                float(line.split(",")[-1])
                break
            except ValueError:
                pass
            first_data_line += 1

        column_count = max([line.count(",") for line in self.file_header]) + 1

        with self.file_path.open("rb") as fp:
            line_count = 0
            buffer = bytearray(4086)
            while fp.readinto(buffer):
                line_count += buffer.count(b"\n")

        self.table = QtWidgets.QTableWidget()
        self.table.itemChanged.connect(self.completeChanged)
        self.table.setMinimumSize(800, 400)
        self.table.setColumnCount(column_count)
        self.table.setRowCount(header_row_count)
        self.table.setFont(QtGui.QFont("Courier"))

        self.box_info.layout().addRow("Line Count:", QtWidgets.QLabel(str(line_count)))

        self.combo_intensity_units = QtWidgets.QComboBox()
        self.combo_intensity_units.addItems(["Counts", "CPS"])
        if any("cps" in line.lower() for line in self.file_header):
            self.combo_intensity_units.setCurrentText("CPS")

        self.combo_delimiter = QtWidgets.QComboBox()
        self.combo_delimiter.addItems([",", ";", "Space", "Tab"])
        self.combo_delimiter.currentIndexChanged.connect(self.fillTable)

        self.spinbox_first_line = QtWidgets.QSpinBox()
        self.spinbox_first_line.setRange(1, header_row_count - 1)
        self.spinbox_first_line.setValue(first_data_line)
        self.spinbox_first_line.valueChanged.connect(self.updateTableIgnores)

        self.le_ignore_columns = QtWidgets.QLineEdit()
        self.le_ignore_columns.setValidator(
            QtGui.QRegularExpressionValidator(QtCore.QRegularExpression("[0-9;]+"))
        )
        self.le_ignore_columns.textChanged.connect(self.updateTableIgnores)

        self.box_options.layout().addRow("Intensity Units:", self.combo_intensity_units)
        self.box_options.layout().addRow("Delimiter:", self.combo_delimiter)
        self.box_options.layout().addRow("Import From Row:", self.spinbox_first_line)
        self.box_options.layout().addRow("Ignore Columns:", self.le_ignore_columns)

        self.fillTable()
        self.guessIgnoreColumnsFromTable()

        self.layout_body.addWidget(self.table)

    def completeChanged(self) -> None:
        complete = self.isComplete()
        self.button_screen.setEnabled(complete)
        super().completeChanged()

    def isComplete(self) -> bool:
        return self.dwelltime.hasAcceptableInput() and not any(
            x in self.forbidden_names for x in self.names()
        )

    def delimiter(self) -> str:
        delimiter = self.combo_delimiter.currentText()
        if delimiter == "Space":
            delimiter = " "
        elif delimiter == "Tab":
            delimiter = "\t"
        return delimiter

    def ignoreColumns(self) -> List[int]:
        return [int(i) - 1 for i in self.le_ignore_columns.text().split(";") if i != ""]

    def useColumns(self) -> List[int]:
        return [
            c for c in range(self.table.columnCount()) if c not in self.ignoreColumns()
        ]

    def names(self) -> List[str]:
        return [
            self.table.item(self.spinbox_first_line.value() - 1, c).text()
            for c in self.useColumns()
        ]

    def fillTable(self) -> None:
        lines = [line.split(self.delimiter()) for line in self.file_header]
        col_count = max(len(line) for line in lines)
        self.table.setColumnCount(col_count)

        for row, line in enumerate(lines):
            line.extend([""] * (col_count - len(line)))
            for col, text in enumerate(line):
                item = QtWidgets.QTableWidgetItem(text.strip())
                self.table.setItem(row, col, item)

        self.table.resizeColumnsToContents()
        self.updateTableIgnores()

        if self.dwelltime.value() is None:
            self.guessDwelltimeFromTable()
            self.dwelltime.setBestUnit()

    def updateTableIgnores(self) -> None:
        header_row = self.spinbox_first_line.value() - 1
        for row in range(self.table.rowCount()):
            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                if item is None:
                    continue
                if row != header_row:
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                else:
                    item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
                if row < header_row or col in self.ignoreColumns():
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEnabled)
                else:
                    item.setFlags(item.flags() | QtCore.Qt.ItemIsEnabled)

    def guessIgnoreColumnsFromTable(self) -> None:
        ignores = []
        header_row = self.spinbox_first_line.value() - 1
        for col in range(self.table.columnCount()):
            text = self.table.item(header_row, col).text().lower()
            if any(x in text for x in ["time", "index"]):
                ignores.append(col + 1)
        self.le_ignore_columns.setText(";".join(str(x) for x in ignores) + ";")

    def guessDwelltimeFromTable(self) -> None:
        header_row = self.spinbox_first_line.value() - 1
        for col in range(self.table.columnCount()):
            text = self.table.item(header_row, col).text().lower()
            if "time" in text:
                try:
                    times = [
                        float(self.table.item(row, col).text().replace(",", "."))
                        for row in range(header_row + 1, self.table.rowCount())
                    ]
                except ValueError:
                    continue
                if "ms" in text:
                    factor = 1e-3
                elif "us" in text or "μs" in text:
                    factor = 1e-6
                else:  # assume that value is in seconds
                    factor = 1.0
                self.dwelltime.setBaseValue(
                    np.round(np.mean(np.diff(times)) * factor, 6)  # type: ignore
                )
                break

    def importOptions(self) -> dict:
        return {
            "importer": "text",
            "path": self.file_path,
            "dwelltime": self.dwelltime.baseValue(),
            "delimiter": self.delimiter(),
            "ignores": self.ignoreColumns(),
            "columns": self.useColumns(),
            "first line": self.spinbox_first_line.value(),
            "names": self.names(),
            "cps": self.combo_intensity_units.currentText() == "CPS",
        }

    def setImportOptions(
        self, options: dict, path: bool = False, dwelltime: bool = True
    ) -> None:
        if path:
            self.file_path.setText(str(options["path"]))
        if dwelltime:
            self.dwelltime.setBaseValue(options["dwelltime"])
            self.dwelltime.setBestUnit()
        delimiter = options["delimiter"]
        if delimiter == " ":
            delimiter = "Space"
        elif delimiter == "\t":
            delimiter = "Tab"
        self.combo_delimiter.setCurrentText(delimiter)
        self.le_ignore_columns.setText(";".join(str(i + 1) for i in options["ignores"]))
        self.spinbox_first_line.setValue(options["first line"])
        for name, c in zip(options["names"], self.useColumns()):
            item = self.table.item(self.spinbox_first_line.value() - 1, c)
            if item is not None:
                item.setText(name)
        self.combo_intensity_units.setCurrentText("CPS" if options["cps"] else "Counts")

    def screenData(self) -> None:
        ppm = self.screening_ppm.value()
        if ppm is None:
            return

        options = self.importOptions()
        data, _ = read_single_particle_file(
            options["path"],
            delimiter=options["delimiter"],
            columns=options["columns"],
            first_line=options["first line"],
            convert_cps=options["dwelltime"] if options["cps"] else None,
            max_rows=100000,
        )
        data = rfn.structured_to_unstructured(data)
        idx = non_target_screen(data, ppm)  # Todo: get alphas
        mask = np.ones(data.shape[1], dtype=bool)
        mask[idx] = 0
        ignores = options["ignores"] + list(np.array(options["columns"])[mask])
        self.le_ignore_columns.setText(";".join(str(i + 1) for i in ignores) + ";")

    def accept(self) -> None:
        options = self.importOptions()

        data, old_names = read_single_particle_file(
            options["path"],
            delimiter=options["delimiter"],
            columns=options["columns"],
            first_line=options["first line"],
            new_names=options["names"],
            convert_cps=options["dwelltime"] if options["cps"] else None,
        )
        # Save original names
        assert data.dtype.names is not None
        options["old names"] = old_names

        self.dataImported.emit(data, options)
        logger.info(f"Text data loaded from {self.file_path} ({data.size} events).")
        super().accept()


class NuImportDialog(_ImportDialogBase):
    def __init__(self, path: str | Path, parent: QtWidgets.QWidget | None = None):
        super().__init__(path, "SPCal Nu Instruments Import", parent)

        self.progress = QtWidgets.QProgressBar()
        self.aborted = False
        self.running = False

        self.threadpool = QtCore.QThreadPool()
        self.results: List[Tuple[int, np.ndarray]] = []

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

        self.signals = data["result"]["signal"] / self.info["AverageSingleIonArea"]
        self.masses = get_masses_from_nu_data(
            data[0],
            self.info["MassCalCoefficients"],
            self.segmentDelays(),
        )[0]

        unit_masses = np.round(self.masses).astype(int)
        isotopes = db["isotopes"][np.isin(db["isotopes"]["Isotope"], unit_masses)]

        self.table = PeriodicTableSelector(enabled_isotopes=isotopes)
        self.table.isotopesChanged.connect(self.completeChanged)

        self.layout_body.addWidget(self.table, 1)
        self.layout_body.addWidget(self.progress, 0)

        # Set info and defaults
        method = self.info["MethodFile"]
        self.box_info.layout().addRow(
            "Method:", QtWidgets.QLabel(method[method.rfind("\\") + 1 :])
        )
        self.box_info.layout().addRow(
            "Number Events:",
            QtWidgets.QLabel(str(self.info["ActualRecordLength"])),
        )
        self.box_info.layout().addRow(
            "Number Integrations:",
            QtWidgets.QLabel(str(len(self.info["IntegrationRegions"]))),
        )

        self.dwelltime.setBaseValue(get_dwelltime_from_info(self.info))
        self.dwelltime.setBestUnit()

        self.table.setFocus()

    def screenData(self) -> None:
        ppm = self.screening_ppm.value()
        if ppm is None:
            return

        idx = non_target_screen(self.signals, ppm)  # Todo: get alphas
        masses = self.masses[idx]
        unit_masses = np.round(masses).astype(int)
        isotopes = db["isotopes"][np.isin(db["isotopes"]["Isotope"], unit_masses)]
        isotopes = isotopes[isotopes["Preferred"] > 0]  # limit to best isotopes
        self.table.setSelectedIsotopes(isotopes)

    def segmentDelays(self) -> Dict[int, float]:
        return {
            s["Num"]: s["AcquisitionTriggerDelayNs"] for s in self.info["SegmentInfo"]
        }

    def isComplete(self) -> bool:
        isotopes = self.table.selectedIsotopes()
        return isotopes is not None and self.dwelltime.hasAcceptableInput()

    def importOptions(self) -> dict:
        return {
            "importer": "nu",
            "path": self.file_path,
            "dwelltime": self.dwelltime.baseValue(),
            "isotopes": self.table.selectedIsotopes(),
        }

    def setImportOptions(
        self, options: dict, path: bool = False, dwelltime: bool = True
    ) -> None:
        if path:
            self.file_path.setText(str(options["path"]))
        if dwelltime:
            self.dwelltime.setBaseValue(options["dwelltime"])
            self.dwelltime.setBestUnit()
        self.table.setSelectedIsotopes(options["isotopes"])

    def setControlsEnabled(self, enabled: bool) -> None:
        button = self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        button.setEnabled(enabled)
        self.table.setEnabled(enabled)
        self.dwelltime.setEnabled(enabled)

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
            "Import Failed",
            "Failed to read integ binary.",
            parent=self,
        )
        msg.exec()

    def abort(self) -> None:
        self.aborted = True
        self.threadpool.clear()
        self.threadpool.waitForDone()
        self.progress.reset()
        self.running = False

        self.setControlsEnabled(True)

    def accept(self) -> None:
        def read_signals(path: Path):
            data = read_nu_integ_binary(path)
            return data["result"]["signal"]

        self.setControlsEnabled(False)

        self.aborted = False
        self.running = True
        self.progress.setValue(0)
        self.progress.setMaximum(len(self.index))
        self.results.clear()

        for idx in self.index:
            path = self.file_path.joinpath(f"{idx['FileNum']}.integ")
            if not path.exists():
                logger.warning(
                    f"NuImportDialog: missing integ file {idx['FileNum']}, skipping"
                )
                self.progress.setValue(self.progress.value() + 1)
                continue
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

    def finalise(self) -> None:
        self.threadpool.waitForDone(1000)
        self.running = False

        options = self.importOptions()
        try:
            signals = np.concatenate(
                [result[1] for result in sorted(self.results, key=lambda r: r[0])]
            )
            signals = signals / self.info["AverageSingleIonArea"]

            isotopes = self.table.selectedIsotopes()
            assert isotopes is not None
            selected_masses = {
                f"{i['Symbol']}{i['Isotope']}": i["Mass"] for i in isotopes
            }
            data = select_nu_signals(self.masses, signals, selected_masses)
        except Exception as e:
            msg = QtWidgets.QMessageBox(
                QtWidgets.QMessageBox.Warning,
                "Import Failed",
                str(e),
                parent=self,
            )
            msg.exec()
            self.abort()
            return

        self.dataImported.emit(data, options)
        logger.info(
            f"Nu instruments data loaded from {self.file_path} ({data.size} events)."
        )
        super().accept()

    def reject(self) -> None:
        if self.running:
            self.abort()
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

        mode = h5["FullSpectra"].attrs["MassCalibMode"]
        ps = [
            h5["FullSpectra"].attrs["MassCalibration p1"],
            h5["FullSpectra"].attrs["MassCalibration p2"],
        ]
        if mode in [2, 5]:
            ps.append(h5["FullSpectra"].attrs["MassCalibration p3"])

        lower = calibrate_mass_to_index(
            peak_table["lower integration limit"][idx], mode, ps
        )
        upper = calibrate_mass_to_index(
            peak_table["upper integration limit"][idx], mode, ps
        )
        self.indicies = np.stack((lower, upper + 1), axis=1)
        self.scale_factor = float(
            (h5["FullSpectra"].attrs["SampleInterval"] * 1e9)  # mV * index -> mV * ns
            / h5["FullSpectra"].attrs["Single Ion Signal"]  # mV * ns -> ions
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
            self.h5.attrs["NbrWrites"]
            * self.h5.attrs["NbrBufs"]
            * self.h5.attrs["NbrSegments"]
        )
        extraction_time = float(self.h5["TimingData"].attrs["TofPeriod"]) * 1e-9

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
        return {
            "importer": "tofwerk",
            "path": self.file_path,
            "dwelltime": self.dwelltime.baseValue(),
            "isotopes": self.table.selectedIsotopes(),
            "other peaks": self.combo_other_peaks.checkedItems(),
        }

    def screenData(self) -> None:
        ppm = self.screening_ppm.value()
        if ppm is None:
            return

        data = self.h5["PeakData"]["PeakData"][:10]
        data = np.reshape(data, (-1, data.shape[-1]))
        data *= factor_extraction_to_acquisition(self.h5)
        idx = non_target_screen(data, ppm)  # Todo alpha

        _isotopes = []
        re_valid = re.compile("\\[(\\d+)([A-Z][a-z]?)\\]\\+")
        for label in self.peak_labels[idx]:
            m = re_valid.match(label)
            if m is not None:
                _isotopes.append(
                    db["isotopes"][
                        (db["isotopes"]["Isotope"] == int(m.group(1)))
                        & (db["isotopes"]["Symbol"] == m.group(2))
                    ]
                )
        isotopes = np.array(_isotopes)
        isotopes = isotopes[isotopes["Preferred"] > 0]
        self.table.setSelectedIsotopes(isotopes)

    def setImportOptions(
        self, options: dict, path: bool = False, dwelltime: bool = True
    ) -> None:
        if path:
            self.file_path.setText(str(options["path"]))
        if dwelltime:
            self.dwelltime.setBaseValue(options["dwelltime"])
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
        self.selected_idx = np.flatnonzero(np.in1d(self.peak_labels, selected_labels))

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
