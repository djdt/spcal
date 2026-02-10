import json
import re
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.datafile import SPCalNuDataFile, SPCalTOFWERKDataFile
from spcal.gui.batch import METHOD_PAGE_ID
from spcal.gui.dialogs.io.text import TextImportDialog
from spcal.gui.widgets.periodictable import PeriodicTableSelector
from spcal.gui.widgets.units import UnitsWidget
from spcal.io.text import guess_text_parameters, iso_time_to_float_seconds
from spcal.isotope import REGEX_ISOTOPE, SPCalIsotope
from spcal.siunits import time_units
from spcal.gui.modelviews.isotope import IsotopeNameDelegate, IsotopeNameValidator


class BatchNuWizardPage(QtWidgets.QWizardPage):
    def __init__(
        self,
        isotopes: list[SPCalIsotope],
        max_mass_diff: float,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setTitle("SPCal Batch Processing")
        self.setSubTitle("Nu Import Options")

        self.cycle_number = QtWidgets.QSpinBox()
        self.cycle_number.setValue(0)
        self.cycle_number.setSpecialValueText("All")

        self.segment_number = QtWidgets.QSpinBox()
        self.segment_number.setValue(0)
        self.segment_number.setSpecialValueText("All")

        self.max_mass_diff = QtWidgets.QDoubleSpinBox()
        self.max_mass_diff.setRange(0.0, 1.0)
        self.max_mass_diff.setValue(max_mass_diff)

        self.check_chunked = QtWidgets.QCheckBox("Split files")
        self.check_chunked.checkStateChanged.connect(self.onChunkChecked)

        self.chunk_size = QtWidgets.QSpinBox()
        self.chunk_size.setRange(1, 10000)
        self.chunk_size.setValue(1000)
        self.chunk_size.setSingleStep(100)
        self.chunk_size.setEnabled(False)

        # todo: option to remove blanked regions?
        # self.combo_blanking = QtWidgets.QComboBox()
        # self.combo_blanking.addItems(["Off", "Blank", "Remove"])
        self.check_blanking = QtWidgets.QCheckBox("Apply auto-blanking.")
        self.check_blanking.setChecked(True)

        self.table = PeriodicTableSelector()
        self.table.isotopesChanged.connect(self.completeChanged)
        self.table.setSelectedIsotopes(
            [iso for iso in isotopes if isinstance(iso, SPCalIsotope)]
        )

        layout_chunk = QtWidgets.QHBoxLayout()
        layout_chunk.addWidget(self.chunk_size)
        layout_chunk.addWidget(self.check_chunked)

        options_box = QtWidgets.QGroupBox("Options")
        options_box_layout = QtWidgets.QFormLayout()
        options_box_layout.addRow("Cycle:", self.cycle_number)
        options_box_layout.addRow("Segment:", self.segment_number)
        options_box_layout.addRow("Chunk size:", layout_chunk)
        options_box_layout.addRow(self.check_blanking)
        options_box.setLayout(options_box_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(options_box, 0)
        layout.addWidget(self.table)
        self.setLayout(layout)

        self.registerField("nu.chunk", self.check_chunked)
        self.registerField("nu.chunk.size", self.chunk_size)
        self.registerField("nu.cycle_number", self.cycle_number)
        self.registerField("nu.segment_number", self.segment_number)
        self.registerField("nu.max_mass_diff", self.max_mass_diff, "value")
        self.registerField("nu.autoblank", self.check_blanking)

        self.registerField("nu.isotopes", self.table, "selectedIsotopesProp")

    def initializePage(self):
        paths: list[Path] = self.field("paths")

        df = SPCalNuDataFile.load(paths[0], last_integ_file=1)
        isotopes = set(df.isotopes)
        min_cycles = df.info["CyclesWritten"]
        min_segments = len(df.info["SegmentInfo"])

        for path in paths[1:]:
            df = SPCalNuDataFile.load(
                path, max_mass_diff=self.max_mass_diff.value(), last_integ_file=1
            )
            min_cycles = min(min_cycles, df.info["CyclesWritten"])
            min_segments = min(min_cycles, len(df.info["SegmentInfo"]))
            isotopes = isotopes.intersection(df.isotopes)

        self.table.setEnabledIsotopes(list(isotopes))
        self.cycle_number.setRange(0, min_cycles)
        self.segment_number.setRange(0, min_segments)

    def onChunkChecked(self, state: QtCore.Qt.CheckState):
        self.chunk_size.setEnabled(state == QtCore.Qt.CheckState.Checked)

    def nextId(self):
        return METHOD_PAGE_ID

    def isComplete(self) -> bool:
        return len(self.table.selectedIsotopes()) > 0

    def validatePage(self):
        if self.check_chunked.isChecked() or self.cycle_number.value() > 0:
            return True

        paths: list[Path] = self.field("paths")
        for path in paths:
            with path.joinpath("integrated.index").open("r") as fp:
                nintegs = len(json.load(fp))
            if nintegs > 1000:
                button = QtWidgets.QMessageBox.warning(
                    self,
                    "Large Files",
                    "Some files have more than 1000 integ files, processing in chunks is reccomended.",
                    QtWidgets.QMessageBox.StandardButton.Ignore
                    | QtWidgets.QMessageBox.StandardButton.Cancel,
                )
                if button == QtWidgets.QMessageBox.StandardButton.Ignore:
                    return True
                else:
                    return False
        return True


class BatchTextWizardPage(QtWidgets.QWizardPage):
    def __init__(
        self,
        isotopes: list[SPCalIsotope],
        delimiter: str,
        skip_rows: int,
        cps: bool,
        event_time: float | None,
        override_event_time: bool,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setTitle("SPCal Batch Processing")
        self.setSubTitle("Text Import Options")

        if delimiter == "":
            delimiter = ","

        self.event_time = UnitsWidget(
            time_units, base_value=event_time, default_unit="ms"
        )
        self.event_time.baseValueChanged.connect(self.completeChanged)
        self.event_time.setEnabled(False)

        self.override_event_time = QtWidgets.QCheckBox("Override")
        self.override_event_time.setChecked(override_event_time)
        self.override_event_time.checkStateChanged.connect(self.onOverrideChecked)

        self.combo_intensity_units = QtWidgets.QComboBox()
        self.combo_intensity_units.addItems(["Counts", "CPS"])
        if cps:
            self.combo_intensity_units.setCurrentText("CPS")

        self.combo_delimiter = QtWidgets.QComboBox()
        self.combo_delimiter.addItems(list(TextImportDialog.DELIMITERS.values()))
        self.combo_delimiter.setCurrentIndex(
            list(TextImportDialog.DELIMITERS.keys()).index(delimiter)
        )
        self.combo_delimiter.currentTextChanged.connect(self.updateIsotopes)

        self.first_line = QtWidgets.QSpinBox()
        self.first_line.setRange(1, TextImportDialog.HEADER_LINE_COUNT - 1)
        self.first_line.setValue(skip_rows)
        self.first_line.valueChanged.connect(self.updateIsotopes)

        self.instrument_type = QtWidgets.QComboBox()
        self.instrument_type.addItems(["Quadrupole", "TOF"])

        self.table_isotopes = QtWidgets.QTableWidget()
        self.table_isotopes.setColumnCount(2)
        self.table_isotopes.setHorizontalHeaderLabels(["Column Name", "Isotope"])
        self.table_isotopes.setItemDelegateForColumn(1, IsotopeNameDelegate())
        self.table_isotopes.model().dataChanged.connect(self.completeChanged)

        layout_event_time = QtWidgets.QHBoxLayout()
        layout_event_time.addWidget(self.event_time, 1)
        layout_event_time.addWidget(
            self.override_event_time, 0, QtCore.Qt.AlignmentFlag.AlignRight
        )

        options_box = QtWidgets.QGroupBox("Options")
        options_box_layout = QtWidgets.QFormLayout()
        options_box_layout.addRow("Event time:", layout_event_time)
        options_box_layout.addRow("Intensity units:", self.combo_intensity_units)
        options_box_layout.addRow("Delimiter:", self.combo_delimiter)
        options_box_layout.addRow("Import from row:", self.first_line)
        options_box_layout.addRow("Instrument type:", self.instrument_type)
        options_box.setLayout(options_box_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(options_box, 0)
        layout.addWidget(self.table_isotopes)
        self.setLayout(layout)

        self.registerField("text.delimiter", self.combo_delimiter, "currentText")
        self.registerField("text.first_line", self.first_line)
        self.registerField("text.cps", self.combo_intensity_units, "currentText")
        self.registerField("text.event_time", self.event_time, "baseValueProp")
        self.registerField("text.event_time.override", self.override_event_time)
        self.registerField("text.instrument_type", self.instrument_type, "currentText")

        self.registerField("text.isotopes", self, "isotopesProp")
        self.registerField("text.isotopes.table", self, "isotopesTableProp")

    def delimiter(self) -> str:
        delimiter = self.combo_delimiter.currentText()
        if delimiter == "Space":
            delimiter = " "
        elif delimiter == "Tab":
            delimiter = "\t"
        return delimiter

    def guessEventTime(self, path: Path) -> float | None:
        header_row = self.first_line.value() - 1
        re_time = re.compile("[\\(\\[]([nmuµ]s)[\\]\\)]")

        header = path.open("r").readlines(
            (header_row + 10) * TextImportDialog.HEADER_LINE_SIZE
        )
        col_names = header[header_row].split(self.delimiter())
        for col, name in enumerate(col_names):
            if "time" in name.lower():
                m = re_time.search(name.lower())
                unit = "s"
                if m is not None:
                    if m.group(1) == "ms":
                        unit = "ms"
                    elif m.group(1) in ["us", "µs"]:
                        unit = "µs"
                    elif m.group(1) == "ns":
                        unit = "ns"

                time_texts = [
                    line.split(self.delimiter())[col]
                    for line in header[header_row + 1 :]
                ]
                if len(time_texts) == 0:
                    return None
                elif "00:" in time_texts[0]:
                    times = [iso_time_to_float_seconds(tt) for tt in time_texts]
                else:
                    times = [float(tt) for tt in time_texts]
                return float(np.mean(np.diff(times))) * time_units[unit]

    def initializePage(self):
        paths: list[Path] = self.field("paths")

        size = TextImportDialog.HEADER_LINE_SIZE
        consistent_parameters = True
        first_header = (
            paths[0].open("r").readlines(TextImportDialog.HEADER_LINE_COUNT * size)
        )
        delimiter, skip_rows, columns = guess_text_parameters(first_header)

        for path in paths[1:]:
            header = path.open("r").readlines((skip_rows + 1) * size)
            _delimiter, _skip_rows, _columns = guess_text_parameters(header)
            if _delimiter != delimiter or _skip_rows != skip_rows:
                consistent_parameters = False
                break

        if consistent_parameters:
            self.combo_delimiter.setCurrentText(TextImportDialog.DELIMITERS[delimiter])
            self.first_line.setValue(skip_rows)

            event_time = self.guessEventTime(paths[0])
            self.event_time.setBaseValue(event_time)
            self.event_time.setBestUnit()

        self.updateIsotopes()

    def updateIsotopes(self):
        paths: list[Path] = self.field("paths")

        row = self.first_line.value() - 1
        delimiter: str = self.delimiter()
        size = TextImportDialog.HEADER_LINE_SIZE
        header = paths[0].open("r").readlines((row + 1) * size)

        shared_names = set(header[row].split(delimiter))

        for path in paths[1:]:
            header = path.open("r").readlines((row + 1) * size)
            shared_names = shared_names.intersection(header[row].split(delimiter))

        self.table_isotopes.clear()
        self.table_isotopes.setRowCount(len(shared_names))

        isotope_count = 0

        for row, name in enumerate(shared_names):
            name = name.strip()
            item = QtWidgets.QTableWidgetItem()
            item.setText(name.replace(" ", "_"))
            item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            item.setCheckState(QtCore.Qt.CheckState.Unchecked)

            self.table_isotopes.setItem(row, 0, item)
            iso_item = QtWidgets.QTableWidgetItem()
            m = REGEX_ISOTOPE.search(name)
            if m is not None and m.group(1) is not None and m.group(2) is not None:
                iso_item.setText(m.group(1) + m.group(2))
                item.setCheckState(QtCore.Qt.CheckState.Checked)
                background = QtGui.QPalette.ColorRole.Base
                isotope_count += 1
            elif m is not None and m.group(3) is not None and m.group(4) is not None:
                iso_item.setText(m.group(4) + m.group(3))
                item.setCheckState(QtCore.Qt.CheckState.Checked)
                background = QtGui.QPalette.ColorRole.Base
                isotope_count += 1
            else:
                background = QtGui.QPalette.ColorRole.AlternateBase
            iso_item.setBackground(self.palette().color(background))

            self.table_isotopes.setItem(row, 1, iso_item)

        self.instrument_type.setEnabled(isotope_count > 1)
        if isotope_count == 1:
            self.instrument_type.setCurrentText("Quadrupole")
        else:
            self.instrument_type.setCurrentText("TOF")

    def selectedIsotopes(self) -> list[SPCalIsotope]:
        selected = []
        for i in range(self.table_isotopes.rowCount()):
            item = self.table_isotopes.item(i, 0)
            if item is not None and item.checkState() == QtCore.Qt.CheckState.Checked:
                item = self.table_isotopes.item(i, 1)
                if item is not None and item.text() != "":
                    selected.append(SPCalIsotope.fromString(item.text()))
        return selected

    def onOverrideChecked(self, checked: QtCore.Qt.CheckState):
        self.event_time.setEnabled(checked == QtCore.Qt.CheckState.Checked)
        self.completeChanged.emit()

    def isotopesTable(self) -> dict[SPCalIsotope, str]:
        table = {}
        for r in range(self.table_isotopes.rowCount()):
            name_item = self.table_isotopes.item(r, 0)
            iso_item = self.table_isotopes.item(r, 1)
            if name_item is None or iso_item is None:
                continue
            try:
                isotope = SPCalIsotope.fromString(iso_item.text())
                table[isotope] = name_item.text()
            except NameError:
                continue
        return table

    def nextId(self):
        return METHOD_PAGE_ID

    def isComplete(self) -> bool:
        if (
            not self.override_event_time.isChecked()
            and self.event_time.baseValue() is None
        ):
            return False
        if len(self.selectedIsotopes()) == 0:
            return False
        return True

    def validatePage(self):
        if self.override_event_time.isChecked():
            return True

        paths: list[Path] = self.field("paths")
        event_time = self.guessEventTime(paths[0])
        for path in paths[1:]:
            _event_time = self.guessEventTime(path)
            if event_time != _event_time:
                button = QtWidgets.QMessageBox.warning(
                    self,
                    "Inconsistent Event Time",
                    "The event time is different in some files, use the event time override.",
                    QtWidgets.QMessageBox.StandardButton.Ignore
                    | QtWidgets.QMessageBox.StandardButton.Cancel,
                )
                if button == QtWidgets.QMessageBox.StandardButton.Ignore:
                    return True
                else:
                    return False
        return True

    isotopesProp = QtCore.Property(list, selectedIsotopes)
    isotopesTableProp = QtCore.Property(object, isotopesTable)


class BatchTOFWERKWizardPage(QtWidgets.QWizardPage):
    def __init__(
        self, isotopes: list[SPCalIsotope], parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(parent)
        self.setTitle("SPCal Batch Processing")
        self.setSubTitle("TOFWERK Options")

        self.table = PeriodicTableSelector()
        self.table.isotopesChanged.connect(self.completeChanged)
        self.table.setSelectedIsotopes(
            [iso for iso in isotopes if isinstance(iso, SPCalIsotope)]
        )

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.table)
        self.setLayout(layout)

        self.registerField("tofwerk.isotopes", self.table, "selectedIsotopesProp")

    def initializePage(self):
        paths: list[Path] = self.field("paths")

        df = SPCalTOFWERKDataFile.load(paths[0], max_size=1)
        isotopes = set(df.isotopes)

        for path in paths[1:]:
            df = SPCalTOFWERKDataFile.load(path, max_size=1)
            isotopes = isotopes.intersection(df.isotopes)

        self.table.setEnabledIsotopes(list(isotopes))

    def nextId(self):
        return METHOD_PAGE_ID

    def isComplete(self) -> bool:
        return len(self.table.selectedIsotopes()) > 0
