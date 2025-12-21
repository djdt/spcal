import logging
import re
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.datafile import SPCalDataFile, SPCalTextDataFile
from spcal.gui.dialogs.io.base import ImportDialogBase
from spcal.gui.modelviews import IsotopeRole
from spcal.gui.modelviews.headers import CheckableHeaderView
from spcal.gui.widgets import UnitsWidget
from spcal.io.text import guess_text_parameters, iso_time_to_float_seconds
from spcal.isotope import (
    ISOTOPE_TABLE,
    RECOMMENDED_ISOTOPES,
    SPCalIsotope,
    REGEX_ISOTOPE,
)
from spcal.siunits import time_units

logger = logging.getLogger(__name__)


class IsotopeValidator(QtGui.QValidator):
    def validate(self, input: str, pos: int) -> tuple[QtGui.QValidator.State, str, int]:
        match = REGEX_ISOTOPE.fullmatch(input)
        if match is None:
            return QtGui.QValidator.State.Intermediate, input, pos
        symbol = match.group(2)
        if match.group(1) is not None:
            isotope = int(match.group(1))
        elif match.group(3) is not None:
            isotope = int(match.group(3))
        else:
            return QtGui.QValidator.State.Intermediate, input, pos

        if (symbol, isotope) in ISOTOPE_TABLE:
            return QtGui.QValidator.State.Acceptable, input, pos
        else:
            return QtGui.QValidator.State.Intermediate, input, pos

    def fixup(self, input: str) -> str:
        match = REGEX_ISOTOPE.match(input)
        if match is None:
            return input
        if (
            len(match.group(2)) == 2
            and match.group(1) is None
            and match.group(3) is None
        ):
            if match.group(2) in RECOMMENDED_ISOTOPES:
                return input + str(RECOMMENDED_ISOTOPES[match.group(2)])
        elif match.group(1) is not None:
            return match.group(1) + match.group(2)
        elif match.group(3) is not None:
            return match.group(2) + match.group(3)

        return input


class IsotopeNameDelegate(QtWidgets.QItemDelegate):
    ISOTOPE_COMPLETER_STRINGS = list(
        f"{symbol}{isotope}" for symbol, isotope in ISOTOPE_TABLE.keys()
    ) + list(f"{isotope}{symbol}" for symbol, isotope in ISOTOPE_TABLE.keys())

    def createEditor(
        self,
        parent: QtWidgets.QWidget,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ) -> QtWidgets.QWidget:
        editor = QtWidgets.QLineEdit(
            index.data(QtCore.Qt.ItemDataRole.EditRole), parent=parent
        )
        editor.setValidator(IsotopeValidator())
        editor.setCompleter(
            QtWidgets.QCompleter(IsotopeNameDelegate.ISOTOPE_COMPLETER_STRINGS)
        )
        return editor

    def setEditorData(
        self,
        editor: QtWidgets.QWidget,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ):
        assert isinstance(editor, QtWidgets.QLineEdit)
        editor.setText(index.data(QtCore.Qt.ItemDataRole.EditRole))

    def setModelData(
        self,
        editor: QtWidgets.QWidget,
        model: QtCore.QAbstractItemModel,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ):
        assert isinstance(editor, QtWidgets.QLineEdit)
        model.setData(index, editor.text(), QtCore.Qt.ItemDataRole.EditRole)
        try:
            isotope = SPCalIsotope.fromString(editor.text())
            model.setData(index, isotope, IsotopeRole)
            model.setData(
                index,
                QtGui.QPalette.ColorRole.Text,
                QtCore.Qt.ItemDataRole.ForegroundRole,
            )
        except NameError:
            model.setData(
                index,
                QtGui.QPalette.ColorRole.Accent,
                QtCore.Qt.ItemDataRole.ForegroundRole,
            )


class TextImportDialog(ImportDialogBase):
    HEADER_LINE_COUNT = 20
    HEADER_LINE_SIZE = 512
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
            self.file_lines[: self.HEADER_LINE_COUNT]
        )
        cps = any(
            "cps" in line.lower() for line in self.file_lines[: self.HEADER_LINE_COUNT]
        )
        event_time, override = None, None

        if delimiter == "":
            delimiter = ","

        if isinstance(existing_file, SPCalTextDataFile):
            delimiter = existing_file.delimiter
            first_data_line = existing_file.skip_row
            cps = existing_file.cps
            event_time = existing_file.event_time
            override = existing_file.override_event_time

        self.table = QtWidgets.QTableWidget()
        self.table.verticalHeader().setMaximumSectionSize(self.logicalDpiX() * 2)
        self.table.itemChanged.connect(self.completeChanged)
        self.table.setMinimumSize(800, 400)
        self.table.setColumnCount(column_count)
        self.table.setRowCount(self.HEADER_LINE_COUNT)
        self.table.setFont(QtGui.QFont("Courier"))

        self.table.setItemDelegate(IsotopeNameDelegate())

        self.table_header = CheckableHeaderView(QtCore.Qt.Orientation.Horizontal)
        self.table.setHorizontalHeader(self.table_header)
        self.table_header.checkStateChanged.connect(self.updateTableUseColumns)

        self.box_info_layout.addRow(
            "Line Count:", QtWidgets.QLabel(str(len(self.file_lines)))
        )

        self.event_time = UnitsWidget(
            time_units, base_value=event_time, default_unit="ms"
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
        self.spinbox_first_line.setRange(1, self.HEADER_LINE_COUNT - 1)
        self.spinbox_first_line.setValue(first_data_line)
        self.spinbox_first_line.valueChanged.connect(self.updateTableUseColumns)

        layout_event_time = QtWidgets.QHBoxLayout()
        layout_event_time.addWidget(self.event_time, 1)
        layout_event_time.addWidget(
            self.override_event_time, 0, QtCore.Qt.AlignmentFlag.AlignRight
        )

        self.box_options_layout.addRow("Event time:", layout_event_time)
        self.box_options_layout.addRow("Intensity units:", self.combo_intensity_units)
        self.box_options_layout.addRow("Delimiter:", self.combo_delimiter)
        self.box_options_layout.addRow("Import from row:", self.spinbox_first_line)

        self.fillTable()

        self.guessIsotopesFromTable()

        self.layout_body.addWidget(self.table)

    def isComplete(self) -> bool:
        if self.event_time.isEnabled() and not self.event_time.hasAcceptableInput():
            return False

        try:
            self.selectedIsotopes()
        except NameError:
            return False
        return True

    def overrideEventTimeChanged(self):
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
                names.append(item.text())
        return names

    def selectedNames(self) -> list[str]:
        names = []
        for c in self.useColumns():
            item = self.table.item(self.spinbox_first_line.value() - 1, c)
            if item is not None:
                names.append(item.text())
        return names

    def isotopeTable(self) -> dict[SPCalIsotope, str]:
        table = {}
        for c in range(self.table.columnCount()):
            item = self.table.item(self.spinbox_first_line.value() - 1, c)
            if item is not None:
                try:
                    table[SPCalIsotope.fromString(item.text())] = item.data(
                        QtCore.Qt.ItemDataRole.UserRole
                    )
                except NameError:
                    pass
        return table

    def selectedIsotopes(self) -> list[SPCalIsotope]:
        return [SPCalIsotope.fromString(name) for name in self.selectedNames()]

    def fillTable(self):
        lines = [
            line.split(self.delimiter())
            for line in self.file_lines[: self.HEADER_LINE_COUNT]
        ]
        col_count = max(len(line) for line in lines)
        self.table.setColumnCount(col_count)

        for row, line in enumerate(lines):
            line.extend([""] * (col_count - len(line)))
            for col, text in enumerate(line):
                item = QtWidgets.QTableWidgetItem(text.strip())
                item.setData(QtCore.Qt.ItemDataRole.UserRole, text.strip())
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

    def updateTableUseColumns(self):
        header_row = self.spinbox_first_line.value() - 1
        for row in range(self.table.rowCount()):
            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                if item is None:
                    continue

                color_group = QtGui.QPalette.ColorGroup.Active
                color_role = QtGui.QPalette.ColorRole.Text
                if row != header_row:
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                else:
                    item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
                    if not REGEX_ISOTOPE.fullmatch(item.text()):
                        color_group = QtGui.QPalette.ColorGroup.Active
                        color_role = QtGui.QPalette.ColorRole.Accent

                if row < header_row or col not in self.useColumns():
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEnabled)
                    color_group = QtGui.QPalette.ColorGroup.Disabled
                else:
                    item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsEnabled)

                item.setForeground(self.palette().color(color_group, color_role))

    def guessIsotopesFromTable(self):
        columns = []
        header_row = self.spinbox_first_line.value() - 1
        for col in range(self.table.columnCount()):
            item = self.table.item(header_row, col)
            if item is None:
                raise ValueError(f"missing item at {header_row}, {col}")
            text = item.text().lower()
            if not any(x in text for x in ["time", "index"]):
                columns.append(col)

        for col in columns:
            self.table_header.setCheckState(col, QtCore.Qt.CheckState.Checked)

    def guessEventTimeFromTable(self) -> tuple[float, str]:
        header_row = self.spinbox_first_line.value() - 1
        for col in range(self.table.columnCount()):
            item = self.table.item(header_row, col)
            if item is None:
                raise ValueError(f"missing item at {header_row}, {col}")
            if "time" in item.text().lower():
                m = re.search("[\\(\\[]([nmuµ]s)[\\]\\)]", item.text().lower())
                unit = "s"
                if m is not None:
                    if m.group(1) == "ms":
                        unit = "ms"
                    elif m.group(1) in ["us", "µs"]:
                        unit = "µs"
                    elif m.group(1) == "ns":
                        unit = "ns"

                time_items = [
                    self.table.item(row, col)
                    for row in range(header_row + 1, self.table.rowCount())
                ]
                time_texts = [
                    ti.text().replace(",", ".") for ti in time_items if ti is not None
                ]
                if len(time_texts) == 0:
                    raise StopIteration
                elif "00:" in time_texts[0]:
                    times = [iso_time_to_float_seconds(tt) for tt in time_texts]
                else:
                    times = [float(tt) for tt in time_texts]
                return float(np.mean(np.diff(times))), unit

        raise StopIteration

    def accept(self):
        data_file = SPCalTextDataFile.load(
            self.file_path,
            isotope_table=self.isotopeTable(),
            delimiter=self.delimiter(),
            skip_rows=self.spinbox_first_line.value(),
            cps=self.combo_intensity_units.currentText() == "CPS",
            override_event_time=self.event_time.value()
            if self.override_event_time.isChecked()
            else None,
        )

        data_file.selected_isotopes = self.selectedIsotopes()
        self.dataImported.emit(data_file)
        logger.info(
            f"Text data loaded from {self.file_path} ({data_file.num_events} events)."
        )
        super().accept()
