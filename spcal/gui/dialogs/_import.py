import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.util import Worker
from spcal.gui.widgets import ElidedLabel, PeriodicTableSelector, UnitsWidget
from spcal.io.nu import get_masses_from_nu_data, read_nu_integ_binary, select_nu_signals
from spcal.io.text import import_single_particle_file
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

        self.file_path = Path(path)
        self.setWindowTitle(f"{title}: {self.file_path.name}")

        self.dwelltime = UnitsWidget(
            time_units,
            default_unit="ms",
            validator=QtGui.QDoubleValidator(0.0, 10.0, 10),
        )
        self.dwelltime.valueChanged.connect(self.completeChanged)

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

        self.box_options = QtWidgets.QGroupBox("Import Options")
        self.box_options.setLayout(QtWidgets.QFormLayout())
        self.box_options.layout().addRow("Dwelltime:", self.dwelltime)

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


class ImportDialog(_ImportDialogBase):
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
        self.le_ignore_columns.setText("1;")
        self.le_ignore_columns.setValidator(
            QtGui.QRegularExpressionValidator(QtCore.QRegularExpression("[0-9;]+"))
        )
        self.le_ignore_columns.textChanged.connect(self.updateTableIgnores)

        self.box_options.layout().addRow("Intensity Units:", self.combo_intensity_units)
        self.box_options.layout().addRow("Delimiter:", self.combo_delimiter)
        self.box_options.layout().addRow("Import From Row:", self.spinbox_first_line)
        self.box_options.layout().addRow("Ignore Columns:", self.le_ignore_columns)

        self.fillTable()

        self.layout_body.addWidget(self.table)

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
            self.readDwelltimeFromTable()

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

    def readDwelltimeFromTable(self) -> None:
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
                    np.round(np.mean(np.diff(times)), 6) * factor  # type: ignore
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

    def accept(self) -> None:
        options = self.importOptions()

        data, old_names = import_single_particle_file(
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
        logger.info(f"Text data loaded from {self.file_path}.")
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

        self.dwelltime.setBaseValue(
            self.info["SegmentInfo"][0]["AcquisitionPeriodNs"] * 1e-9
        )

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

    def setControlsEnabled(self, enabled: bool) -> None:
        button = self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        button.setEnabled(enabled)
        self.table.setEnabled(enabled)
        self.dwelltime.setEnabled(enabled)

    def threadComplete(self) -> None:
        if self.aborted:
            return

        self.progress.setValue(self.progress.value() + 1)
        if self.threadpool.activeThreadCount() == 0 and self.running:
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
        logger.info(f"Nu instruments data loaded from {self.file_path}.")
        super().accept()

    def reject(self) -> None:
        if self.running:
            self.abort()
        else:
            super().reject()
