import json
import datetime
import logging
from pathlib import Path
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.datafile import (
    SPCalDataFile,
    SPCalNuDataFile,
    SPCalTOFWERKDataFile,
    SPCalTextDataFile,
)
from spcal.gui.dialogs.io.text import TextImportDialog
from spcal.gui.docks.instrumentoptions import SPCalInstrumentOptionsWidget
from spcal.gui.docks.isotopeoptions import IsotopeOptionTable
from spcal.gui.docks.limitoptions import SPCalLimitOptionsWidget
from spcal.gui.io import (
    NU_FILE_FILTER,
    TEXT_FILE_FILTER,
    TOFWERK_FILE_FILTER,
    get_open_spcal_paths,
    most_recent_spcal_path,
)
from spcal.gui.widgets.units import UnitsWidget
from spcal.io.nu import eventtime_from_info, is_nu_directory, is_nu_run_info_file
from spcal.io.text import guess_text_parameters, is_text_file
from spcal.io.tofwerk import is_tofwerk_file
from spcal.isotope import SPCalIsotope, SPCalIsotopeBase
from spcal.gui.widgets.periodictable import PeriodicTableSelector
from spcal.processing.method import SPCalProcessingMethod
from spcal.processing.options import SPCalIsotopeOptions
from spcal.siunits import mass_units, size_units, volume_units, time_units

from spcal.gui.batch.workers import NuBatchWorker, TextBatchWorker

logger = logging.getLogger(__name__)


class BatchFileListDelegate(QtWidgets.QStyledItemDelegate):
    def editorEvent(
        self,
        event: QtCore.QEvent,
        model: QtCore.QAbstractItemModel,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ) -> bool:
        if event.type() == QtCore.QEvent.Type.MouseButtonPress:
            assert isinstance(event, QtGui.QMouseEvent)
            self.initStyleOption(option, index)
            style = (
                option.widget.style()  # type: ignore
                if option.widget is not None  # type: ignore
                else QtWidgets.QApplication.style()
            )

            rect = style.subElementRect(
                QtWidgets.QStyle.SubElement.SE_ItemViewItemDecoration,
                option,
                option.widget,  # type: ignore
            )
            if rect.contains(event.position().toPoint()):
                model.removeRow(index.row())
                return True
        return super().editorEvent(event, model, option, index)


class BatchFilesWizardPage(QtWidgets.QWizardPage):
    FORMAT_FILTERS = {
        "Text": TEXT_FILE_FILTER,
        "Nu": NU_FILE_FILTER,
        "TOFWERK": TOFWERK_FILE_FILTER,
    }
    FORMAT_FUNCTIONS = {
        "Text": is_text_file,
        "Nu": is_nu_directory,
        "TOFWERK": is_tofwerk_file,
    }
    MAX_SEARCH_DEPTH = 5

    pathsChanged = QtCore.Signal()

    def __init__(
        self,
        existing_file: SPCalDataFile | None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setTitle("SPCal Batch Processing")
        self.setSubTitle("Select Data Files")

        label = QtWidgets.QLabel("Select files for batch processing.")

        self.radio_text = QtWidgets.QRadioButton("Text Exports")
        self.radio_nu = QtWidgets.QRadioButton("Nu Instruments")
        self.radio_tofwerk = QtWidgets.QRadioButton("TOFWERK HDF5")
        self.radio_text.toggled.connect(self.formatChanged)
        self.radio_nu.toggled.connect(self.formatChanged)
        self.radio_tofwerk.toggled.connect(self.formatChanged)

        self.files = QtWidgets.QListWidget()
        self.files.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.files.setTextElideMode(QtCore.Qt.TextElideMode.ElideLeft)
        self.files.setItemDelegate(BatchFileListDelegate())
        self.files.model().rowsInserted.connect(self.completeChanged)
        self.files.model().rowsRemoved.connect(self.completeChanged)
        self.files.model().rowsInserted.connect(self.pathsChanged)
        self.files.model().rowsRemoved.connect(self.pathsChanged)

        self.button_open = QtWidgets.QPushButton("Add File(s)")
        self.button_open.pressed.connect(self.dialogOpen)
        self.button_open.setToolTip("Add files of the selected format.")

        self.button_open_all = QtWidgets.QPushButton("Add Directory")
        self.button_open_all.pressed.connect(self.dialogOpenAll)
        self.button_open_all.setToolTip(
            "Recursively search directory for single particle files of the selected format."
        )

        self.button_clear = QtWidgets.QPushButton("Clear")
        self.button_clear.pressed.connect(self.files.clear)
        self.button_clear.pressed.connect(self.completeChanged)

        # init
        if isinstance(existing_file, SPCalNuDataFile):
            self.radio_nu.setChecked(True)
        elif isinstance(existing_file, SPCalTOFWERKDataFile):
            self.radio_tofwerk.setChecked(True)
        else:
            self.radio_text.setChecked(True)

        radio_box = QtWidgets.QGroupBox("Data Format")
        radio_box_layout = QtWidgets.QVBoxLayout()
        radio_box_layout.addWidget(self.radio_text)
        radio_box_layout.addWidget(self.radio_nu)
        radio_box_layout.addWidget(self.radio_tofwerk)
        radio_box.setLayout(radio_box_layout)

        file_button_layout = QtWidgets.QHBoxLayout()
        file_button_layout.addWidget(self.button_clear)
        file_button_layout.addStretch(1)
        file_button_layout.addWidget(self.button_open)
        file_button_layout.addWidget(self.button_open_all)
        file_button_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

        files_box = QtWidgets.QGroupBox("Data Files")
        files_box_layout = QtWidgets.QVBoxLayout()
        files_box_layout.addWidget(self.files, 1)
        files_box_layout.addLayout(file_button_layout, 0)
        files_box.setLayout(files_box_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(radio_box, 0)
        layout.addWidget(files_box, 1)
        self.setLayout(layout)

        self.registerField("paths", self, "pathsProp")

    def addFile(self, path: Path):
        item = QtWidgets.QListWidgetItem()
        item.setData(QtCore.Qt.ItemDataRole.UserRole, path)
        item.setText(str(path))
        item.setIcon(QtGui.QIcon.fromTheme("list-remove"))
        self.files.addItem(item)

    def paths(self) -> list[Path]:
        return [Path(self.files.item(i).text()) for i in range(self.files.count())]

    def setPaths(self, paths: list[Path]):
        self.files.clear()
        for path in paths:
            self.addFile(path)

    def formatChanged(self):
        format = self.selectedFormat()
        file_fn = BatchFilesWizardPage.FORMAT_FUNCTIONS[format]
        palette = self.palette()
        for i in range(self.files.count()):
            item = self.files.item(i)
            if file_fn(item.data(QtCore.Qt.ItemDataRole.UserRole)):
                item.setForeground(palette.color(QtGui.QPalette.ColorRole.Text))
            else:
                item.setForeground(QtCore.Qt.GlobalColor.red)

        self.completeChanged.emit()

    def selectedFormat(self) -> str:
        if self.radio_text.isChecked():
            return "Text"
        elif self.radio_nu.isChecked():
            return "Nu"
        elif self.radio_tofwerk.isChecked():
            return "TOFWERK"
        else:
            raise ValueError("unknown format")

    def dialogOpen(self):
        paths = get_open_spcal_paths(
            self,
            selected_filter=BatchFilesWizardPage.FORMAT_FILTERS[self.selectedFormat()],
        )
        for path in paths:
            self.addFile(path)

    def dialogOpenAll(self):
        recent = most_recent_spcal_path()
        if recent is not None:
            dir = str(recent.parent)
        else:
            dir = ""

        format = self.selectedFormat()

        root = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            f"Search Directory ({format})",
            dir,
            QtWidgets.QFileDialog.Option.ReadOnly,
        )
        if root == "":
            return

        root = Path(root)

        file_fn = BatchFilesWizardPage.FORMAT_FUNCTIONS[format]

        for path, dirs, filenames in root.walk():
            depth = len(path.relative_to(root).parents)
            if depth > BatchFilesWizardPage.MAX_SEARCH_DEPTH:
                dirs.clear()
            for filename in filenames:
                filepath = path.joinpath(filename)
                if is_nu_run_info_file(filepath):
                    filepath = filepath.parent
                if file_fn(filepath):
                    self.addFile(filepath)

    def isComplete(self) -> bool:
        if self.files.count() == 0:
            return False
        format = self.selectedFormat()
        file_fn = BatchFilesWizardPage.FORMAT_FUNCTIONS[format]
        for path in self.paths():
            if not file_fn(path):
                return False
        return True

    def validatePage(self) -> bool:
        format = self.selectedFormat()

        if format == "Text":
            return True
        elif format == "Nu":
            paths = self.paths()
            with open(paths[0].joinpath("run.info"), "r") as fp:
                info = json.load(fp)
            event_time = eventtime_from_info(info)
            for path in paths[1:]:
                with open(path.joinpath("run.info"), "r") as fp:
                    info = json.load(fp)
                if eventtime_from_info(info) != event_time:
                    button = QtWidgets.QMessageBox.warning(
                        self,
                        "Different Event Times",
                        "The event time is not consistent across all selected files.",
                        QtWidgets.QMessageBox.StandardButton.Ignore
                        | QtWidgets.QMessageBox.StandardButton.Cancel,
                    )
                    if button == QtWidgets.QMessageBox.StandardButton.Ignore:
                        return True
                    else:
                        return False
        return True

    def nextId(self) -> int:
        format = self.selectedFormat()
        if format == "Text":
            return SPCalBatchProcessingWizard.TEXT_PAGE_ID
        elif format == "Nu":
            return SPCalBatchProcessingWizard.NU_PAGE_ID
        elif format == "TOFWERK":
            return SPCalBatchProcessingWizard.TOFWERK_PAGE_ID
        else:
            raise ValueError("unknown format")

    pathsProp = QtCore.Property(list, paths, setPaths, notify=pathsChanged)  # type: ignore


class BatchNuWizardPage(QtWidgets.QWizardPage):
    def __init__(
        self,
        isotopes: list[SPCalIsotopeBase],
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
        return SPCalBatchProcessingWizard.METHOD_PAGE_ID

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
        isotopes: list[SPCalIsotopeBase],
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
        # self.override_event_time.checkStateChanged.connect(
        #     self.overrideEventTimeChanged
        # )
        self.override_event_time.setChecked(override_event_time)
        self.override_event_time.checkStateChanged.connect(self.completeChanged)

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

        self.list_isotopes = QtWidgets.QListWidget()
        self.list_isotopes.itemPressed.connect(self.completeChanged)

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
        options_box.setLayout(options_box_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(options_box, 0)
        layout.addWidget(self.list_isotopes)
        self.setLayout(layout)

        self.registerField("text.delimiter", self.combo_delimiter)
        self.registerField("text.first_line", self.first_line)
        self.registerField("text.units", self.combo_intensity_units)
        self.registerField("text.event_time", self.event_time, "baseValueProp")
        self.registerField("text.event_time.override", self.override_event_time)

        self.registerField("text.isotopes", self, "isotopesProp")

    def delimiter(self) -> str:
        delimiter = self.combo_delimiter.currentText()
        if delimiter == "Space":
            delimiter = " "
        elif delimiter == "Tab":
            delimiter = "\t"
        return delimiter

    # @QtCore.Property(bool)
    # def cps(self) -> bool:
    #     return self.combo_intensity_units.currentText() == "CPS"
    #
    # @QtCore.Property(int)
    # def skip_rows(self) -> int:
    #     return self.first_line.value() - 1

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

        # header_row = self.spinbox_first_line.value() - 1
        # for col in range(self.table.columnCount()):
        #     item = self.table.item(header_row, col)
        #     if item is None:
        #         raise ValueError(f"missing item at {header_row}, {col}")
        #     if "time" in item.text().lower():
        #         m = re.search("[\\(\\[]([nmuµ]s)[\\]\\)]", item.text().lower())
        #         unit = "s"
        #         if m is not None:
        #             if m.group(1) == "ms":
        #                 unit = "ms"
        #             elif m.group(1) in ["us", "µs"]:
        #                 unit = "µs"
        #             elif m.group(1) == "ns":
        #                 unit = "ns"
        #
        #         time_items = [
        #             self.table.item(row, col)
        #             for row in range(header_row + 1, self.table.rowCount())
        #         ]
        #         time_texts = [
        #             ti.text().replace(",", ".") for ti in time_items if ti is not None
        #         ]
        #         if len(time_texts) == 0:
        #             raise StopIteration
        #         elif "00:" in time_texts[0]:
        #             times = [iso_time_to_float_seconds(tt) for tt in time_texts]
        #         else:
        #             times = [float(tt) for tt in time_texts]
        #         return float(np.mean(np.diff(times))), unit
        #
        self.updateIsotopes()

    def updateIsotopes(self):
        paths: list[Path] = self.field("paths")

        row = self.first_line.value() - 1
        delimiter: str = self.delimiter()
        size = TextImportDialog.HEADER_LINE_SIZE
        header = paths[0].open("r").readlines((row + 1) * size)

        selected_isotopes = self.selectedIsotopes()

        shared_isotopes = set(
            SPCalIsotope.fromString(x) for x in header[row].split(delimiter)
        )

        for path in paths[1:]:
            header = path.open("r").readlines((row + 1) * size)
            isotopes = set(
                SPCalIsotope.fromString(x) for x in header[row].split(delimiter)
            )
            shared_isotopes = shared_isotopes.intersection(isotopes)

        self.list_isotopes.clear()
        for isotope in sorted(shared_isotopes, key=lambda iso: iso.isotope):
            item = QtWidgets.QListWidgetItem(str(isotope))
            item.setData(QtCore.Qt.ItemDataRole.UserRole, isotope)
            item.setCheckState(
                QtCore.Qt.CheckState.Checked
                if isotope in selected_isotopes
                else QtCore.Qt.CheckState.Unchecked
            )
            self.list_isotopes.addItem(item)

    def selectedIsotopes(self) -> list[SPCalIsotope]:
        selected = []
        for i in range(self.list_isotopes.count()):
            item = self.list_isotopes.item(i)
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                selected.append(item.data(QtCore.Qt.ItemDataRole.UserRole))
        return selected

    def nextId(self):
        return SPCalBatchProcessingWizard.METHOD_PAGE_ID

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


class BatchTOFWERKWizardPage(QtWidgets.QWizardPage):
    def __init__(
        self, isotopes: list[SPCalIsotopeBase], parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(parent)
        self.setTitle("SPCal Batch Processing")
        self.setSubTitle("TOFWERK Options")
        self.table = PeriodicTableSelector()


class BatchMethodWizardPage(QtWidgets.QWizardPage):
    def __init__(
        self,
        method: SPCalProcessingMethod,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setTitle("SPCal Batch Processing")
        self.setSubTitle("Processing Method")

        self.instrument_options = SPCalInstrumentOptionsWidget(
            method.instrument_options, method.calibration_mode
        )
        self.instrument_options.button_efficiency.hide()

        self.limit_options = SPCalLimitOptionsWidget(
            method.limit_options,
            method.accumulation_method,
            method.points_required,
            method.prominence_required,
        )

        self.isotope_table = IsotopeOptionTable()
        self.isotope_table.isotope_model.isotope_options = method.isotope_options

        top_layout = QtWidgets.QHBoxLayout()
        top_layout.addWidget(self.instrument_options)
        top_layout.addWidget(self.limit_options)
        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(top_layout)
        layout.addWidget(self.isotope_table)
        self.setLayout(layout)

        self.registerField("method", self, "methodProp")

    def initializePage(self):
        if self.wizard().hasVisitedPage(SPCalBatchProcessingWizard.TEXT_PAGE_ID):
            isotopes = self.field("text.isotopes")
        elif self.wizard().hasVisitedPage(SPCalBatchProcessingWizard.NU_PAGE_ID):
            isotopes = self.field("nu.isotopes")
        elif self.wizard().hasVisitedPage(SPCalBatchProcessingWizard.TOFWERK_PAGE_ID):
            isotopes = self.field("tofwerk.isotopes")
        else:
            raise ValueError("has not visited any format pages")

        current = self.isotope_table.isotope_model.isotope_options

        self.isotope_table.isotope_model.beginResetModel()
        self.isotope_table.isotope_model.isotope_options = {
            isotope: current.get(isotope, SPCalIsotopeOptions(None, None, None))
            for isotope in isotopes
        }
        self.isotope_table.isotope_model.endResetModel()

    def getMethod(self) -> SPCalProcessingMethod:
        return SPCalProcessingMethod(
            instrument_options=self.instrument_options.instrumentOptions(),
            limit_options=self.limit_options.limitOptions(),
            isotope_options=self.isotope_table.isotope_model.isotope_options,
            accumulation_method=self.limit_options.limit_accumulation,
            points_required=self.limit_options.points_required,
            prominence_required=self.limit_options.prominence_required,
            calibration_mode=self.instrument_options.calibration_mode.currentText().lower(),
        )

    methodProp = QtCore.Property(object, getMethod)


class BatchRunWizardPage(QtWidgets.QWizardPage):
    INVALID_CHARS = '<>:"/\\|?*'

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setTitle("SPCal Batch Processing")
        self.setSubTitle("Run Batch Processing")

        filename_regexp = QtCore.QRegularExpression(
            f"(%DataFile%)?[^{BatchRunWizardPage.INVALID_CHARS}]+(%DataFile%)?"
        )

        self.output_files = QtWidgets.QListWidget()
        self.output_files.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.output_files.setTextElideMode(QtCore.Qt.TextElideMode.ElideLeft)

        self.output_name = QtWidgets.QLineEdit("%DataFile%_spcal_results.csv")
        self.output_name.setMinimumWidth(300)
        self.output_name.setValidator(
            QtGui.QRegularExpressionValidator(filename_regexp)
        )
        self.output_name.textChanged.connect(self.updateOutputNames)
        self.output_name.textChanged.connect(self.completeChanged)

        self.output_dir = QtWidgets.QLineEdit()
        self.output_dir.textChanged.connect(self.updateOutputNames)
        self.output_dir.textChanged.connect(self.completeChanged)

        self.button_dir = QtWidgets.QPushButton("Select")
        self.button_dir.pressed.connect(self.dialogOutputDirectory)

        # options

        self.check_export_options = QtWidgets.QCheckBox(
            "Instrument, limit and isotope options"
        )
        self.check_export_options.setChecked(True)
        self.check_export_clusters = QtWidgets.QCheckBox("Clustering results")
        self.check_export_arrays = QtWidgets.QCheckBox("Particle data arrays")
        self.check_export_arrays.setChecked(True)
        self.check_export_summary = QtWidgets.QCheckBox("Batch summary")
        self.summary_filename = QtWidgets.QLineEdit(
            f"{datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%dT%H_%M_%S_scpal_batch.csv')}"
        )
        self.check_export_summary.toggled.connect(self.completeChanged)
        self.check_export_summary.toggled.connect(self.summary_filename.setEnabled)
        self.summary_filename.setEnabled(False)
        self.summary_filename.textChanged.connect(self.completeChanged)

        self.check_export_images = QtWidgets.QCheckBox("Images")
        self.button_image_options = QtWidgets.QPushButton("Options...")
        self.check_export_images.setEnabled(False)

        # units

        self.mass_units = QtWidgets.QComboBox()
        self.mass_units.addItems(list(mass_units.keys()))
        self.mass_units.setCurrentText("fg")

        self.size_units = QtWidgets.QComboBox()
        self.size_units.addItems(list(size_units.keys()))
        self.size_units.setCurrentText("nm")

        self.volume_units = QtWidgets.QComboBox()
        self.volume_units.addItems(list(volume_units.keys()))
        self.volume_units.setCurrentText("nm³")

        self.status = QtWidgets.QLabel("")

        layout_dir = QtWidgets.QHBoxLayout()
        layout_dir.addWidget(self.output_dir, 1)
        layout_dir.addWidget(self.button_dir, 0, QtCore.Qt.AlignmentFlag.AlignRight)

        box_output = QtWidgets.QGroupBox("Export Paths")
        box_output_layout = QtWidgets.QFormLayout()
        box_output_layout.addRow("Filename:", self.output_name)
        box_output_layout.addRow("Directory:", layout_dir)
        box_output_layout.addRow(self.output_files)
        box_output.setLayout(box_output_layout)

        summary_option_layout = QtWidgets.QHBoxLayout()
        summary_option_layout.setDirection(QtWidgets.QHBoxLayout.Direction.RightToLeft)
        summary_option_layout.addWidget(
            self.summary_filename,
            1,
        )
        summary_option_layout.addWidget(self.check_export_summary, 0)

        image_option_layout = QtWidgets.QHBoxLayout()
        image_option_layout.addWidget(self.check_export_images, 1)
        image_option_layout.addWidget(
            self.button_image_options, 0, QtCore.Qt.AlignmentFlag.AlignRight
        )

        box_options = QtWidgets.QGroupBox("Export Options")
        box_options_layout = QtWidgets.QFormLayout()
        box_options_layout.addRow(self.check_export_options)
        box_options_layout.addRow(self.check_export_arrays)
        box_options_layout.addRow(self.check_export_clusters)
        box_options_layout.addRow(image_option_layout)
        box_options_layout.addRow(summary_option_layout)
        box_options.setLayout(box_options_layout)

        box_units = QtWidgets.QGroupBox("Export Units")
        box_units_layout = QtWidgets.QFormLayout()
        box_units_layout.addRow("Mass", self.mass_units)
        box_units_layout.addRow("Size", self.size_units)
        box_units_layout.addRow("Volume", self.volume_units)
        box_units.setLayout(box_units_layout)

        layout = QtWidgets.QGridLayout()
        layout.addWidget(box_output, 0, 0, 2, 1)
        layout.addWidget(box_options, 0, 1)
        layout.addWidget(box_units, 1, 1)
        layout.addWidget(self.status, 2, 0, 1, 2)
        self.setLayout(layout)

        self.registerField("export.options", self.check_export_options)
        self.registerField("export.arrays", self.check_export_arrays)
        self.registerField("export.clusters", self.check_export_clusters)
        self.registerField("export.images", self.check_export_images)
        self.registerField("export.summary", self.check_export_summary)
        self.registerField("export.summary.name", self.summary_filename)
        self.registerField("export.units", self, "unitsProp")

    def units(self) -> dict[str, tuple[str, float]]:
        mass = self.mass_units.currentText()
        size = self.size_units.currentText()
        volume = self.volume_units.currentText()
        return {
            "signal": ("cts", 1.0),
            "mass": (mass, mass_units[mass]),
            "size": (size, size_units[size]),
            "volume": (volume, volume_units[volume]),
        }

    def isComplete(self) -> bool:
        if self.check_export_summary.isChecked() and (
            not self.summary_filename.hasAcceptableInput()
            or self.summary_filename.text() == ""
        ):
            return False

        outdir = self.output_dir.text()
        if outdir == "" or (Path(outdir).exists() and not Path(outdir).is_dir()):
            return False

        return self.output_name.hasAcceptableInput()

    def initializePage(self):
        paths: list[Path] = self.field("paths")
        self.output_dir.setText(str(paths[0].parent))
        self.output_files.clear()

        for path in paths:
            self.addFile(path)

        self.updateOutputNames()

    def validatePage(self) -> bool:
        existing = [path for _, path in self.pathPairs() if path.exists()]
        if len(existing) == 0:
            return True
        else:
            button = QtWidgets.QMessageBox.question(
                self,
                "Overwrite Files?",
                f"Are you sure you want to overwrite {len(existing)} files?",
            )
            if button == QtWidgets.QMessageBox.StandardButton.Yes:
                return True
        return False

    def updateOutputNames(self):
        name = self.output_name.text()

        for i in range(self.output_files.count()):
            item = self.output_files.item(i)
            path: Path = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if path.is_dir():
                outname = name.replace("%DataFile%", path.name)
            else:
                outname = name.replace("%DataFile%", path.stem)

            item.setText(outname)
            outpath = Path(self.output_dir.text()).joinpath(outname)

            if outpath.exists():
                item.setIcon(QtGui.QIcon.fromTheme("data-warning"))
            else:
                item.setIcon(QtGui.QIcon.fromTheme("document-new"))

    def dialogOutputDirectory(self):
        dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Output Directory", dir=self.output_dir.text()
        )
        if dir != "":
            self.output_dir.setText(dir)

    def addFile(self, path: Path, chunk: tuple[int, int | None] | None = None):
        item = QtWidgets.QListWidgetItem()
        item.setText(str(path))
        item.setData(QtCore.Qt.ItemDataRole.UserRole, path)
        item.setData(QtCore.Qt.ItemDataRole.UserRole + 1, chunk)
        self.output_files.addItem(item)

    def pathPairs(self) -> list[tuple[Path, Path]]:
        pairs = []
        for i in range(self.output_files.count()):
            item = self.output_files.item(i)
            pairs.append(
                (
                    item.data(QtCore.Qt.ItemDataRole.UserRole),
                    Path(self.output_dir.text()).joinpath(item.text()),
                )
            )
        return pairs

    @QtCore.Slot()
    def updateProgress(self, index: int, progress: float):
        item = self.output_files.item(index)

        if progress >= 1.0:
            icon = QtGui.QIcon.fromTheme("task-process-4")
        elif progress >= 0.75:
            icon = QtGui.QIcon.fromTheme("task-process-3")
        elif progress >= 0.5:
            icon = QtGui.QIcon.fromTheme("task-process-2")
        elif progress >= 0.25:
            icon = QtGui.QIcon.fromTheme("task-process-1")
        else:
            icon = QtGui.QIcon.fromTheme("task-process-0")
        item.setIcon(icon)

    def setStatus(self, text: str):
        self.status.setText(text)

    def updateStatus(self, index: int, elapsed: float):
        nitems = self.output_files.count()
        if index == 0:
            text = "Processing..."
        else:
            time_per_task = elapsed / index
            remaining_time = time_per_task * (nitems - index)
            if remaining_time < 60.0:
                text = "Processing... < 1 minute remaining"
            else:
                text = (
                    f"Processing... {int(remaining_time / 60.0) + 1} minutes remaining"
                )
        self.status.setText(text)

    def resetProgress(self):
        self.updateOutputNames()

    unitsProp = QtCore.Property(dict, units)


class SPCalBatchProcessingWizard(QtWidgets.QWizard):
    FILE_PAGE_ID = 0
    TEXT_PAGE_ID = 1
    NU_PAGE_ID = 2
    TOFWERK_PAGE_ID = 3
    METHOD_PAGE_ID = 4
    RUN_PAGE_ID = 5

    def __init__(
        self,
        existing_file: SPCalDataFile | None,
        method: SPCalProcessingMethod,
        selected_isotopes: list[SPCalIsotopeBase],
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("SPCal Batch Processing")
        self.resize(1024, 800)
        self.setButtonText(QtWidgets.QWizard.WizardButton.FinishButton, "Start Batch")
        self.setButtonText(QtWidgets.QWizard.WizardButton.CancelButton, "Close")

        self.process_thread = QtCore.QThread()
        self.process_timer = QtCore.QElapsedTimer()

        self.setPage(
            SPCalBatchProcessingWizard.FILE_PAGE_ID, BatchFilesWizardPage(existing_file)
        )
        delimiter = ","
        skip_rows = 1
        cps = False
        event_time = None
        override_event_time = False
        if isinstance(existing_file, SPCalTextDataFile):
            delimiter = existing_file.delimter
            skip_rows = existing_file.skip_row
            cps = existing_file.cps
            event_time = existing_file.override_event_time
            override_event_time = existing_file.override_event_time is not None

        self.setPage(
            SPCalBatchProcessingWizard.TEXT_PAGE_ID,
            BatchTextWizardPage(
                selected_isotopes,
                delimiter,
                skip_rows,
                cps,
                event_time,
                override_event_time,
            ),
        )
        max_mass_diff = (
            existing_file.max_mass_diff
            if isinstance(existing_file, SPCalNuDataFile)
            else 0.05
        )
        self.setPage(
            SPCalBatchProcessingWizard.NU_PAGE_ID,
            BatchNuWizardPage(selected_isotopes, max_mass_diff),
        )
        self.setPage(
            SPCalBatchProcessingWizard.TOFWERK_PAGE_ID,
            BatchTOFWERKWizardPage(selected_isotopes),
        )
        self.setPage(
            SPCalBatchProcessingWizard.METHOD_PAGE_ID, BatchMethodWizardPage(method)
        )
        self.run_page = BatchRunWizardPage()
        self.setPage(SPCalBatchProcessingWizard.RUN_PAGE_ID, self.run_page)

    def accept(self):
        if not self.validateCurrentPage():
            return

        self.button(QtWidgets.QWizard.WizardButton.FinishButton).setEnabled(False)
        self.setButtonText(QtWidgets.QWizard.WizardButton.CancelButton, "Cancel")

        paths = self.run_page.pathPairs()
        method: SPCalProcessingMethod = self.field("method")

        if not paths[0][1].parent.is_dir():
            paths[0][1].parent.mkdir()

        if self.field("export.summary"):
            summary_path = paths[0][1].parent.joinpath(
                self.field("export.summary.name")
            )
        else:
            summary_path = None

        export_options = {
            "results": True,
            "options": self.field("export.options"),
            "arrays": self.field("export.arrays"),
            "clusters": self.field("export.clusters"),
            "summary": summary_path,
            "units": self.field("export.units"),
        }
        print(export_options)

        if self.hasVisitedPage(SPCalBatchProcessingWizard.TEXT_PAGE_ID):
            isotopes: list[SPCalIsotope] = self.field("text.isotopes")
            self.worker = TextBatchWorker(
                paths,
                method,
                isotopes,
                export_options,
                delimiter=self.field("text.delimiter"),
                skip_rows=self.field("text.skip_rows"),
                cps=self.field("text.cps"),
                override_event_time=self.field("text.override_event_time"),
                instrument_type=self.field("text.instrument_type"),
            )
        elif self.hasVisitedPage(SPCalBatchProcessingWizard.NU_PAGE_ID):
            isotopes: list[SPCalIsotope] = self.field("nu.isotopes")

            chunk_size = self.field("nu.chunk.size") if self.field("nu.chunk") else 0
            cyc_number = self.field("nu.cycle_number")
            if cyc_number == 0:
                cyc_number = None
            seg_number = self.field("nu.segment_number")
            if seg_number == 0:
                seg_number = None

            self.worker = NuBatchWorker(
                paths,
                method,
                isotopes,
                export_options,
                chunk_size=chunk_size,
                max_mass_diff=self.field("nu.max_mass_diff"),
                cyc_number=cyc_number,
                seg_number=seg_number,
                autoblank=self.field("nu.autoblank"),
            )

        elif self.hasVisitedPage(SPCalBatchProcessingWizard.TOFWERK_PAGE_ID):
            isotopes: list[SPCalIsotope] = self.field("tofwerk.isotopes")
            self.worker = TextBatchWorker(paths, method, isotopes)
        else:
            raise ValueError("no format page visited")

        self.worker.moveToThread(self.process_thread)

        self.worker.started.connect(self.startProgress)
        self.worker.progress.connect(self.updateProgress)
        self.worker.finished.connect(self.stopProgress)

        self.process_thread.started.connect(self.worker.process)
        self.process_thread.finished.connect(self.worker.deleteLater)

        self.process_thread.start()

    def startProgress(self, number_items: int):
        self.process_timer.start()
        logger.info(f"Batch processing started for {number_items} data files.")

    def updateProgress(self, index: int, partial: float):
        self.run_page.updateProgress(index, partial)
        if partial == 0.0:
            self.run_page.updateStatus(index, self.process_timer.elapsed() / 1000.0)

    def stopProgress(self):
        self.stopThread()
        self.setButtonText(QtWidgets.QWizard.WizardButton.CancelButton, "Close")
        self.run_page.setStatus("Processing Complete!")
        logger.info("Batch processing complete.")

    def reject(self):
        if self.process_thread.isRunning():
            self.process_thread.requestInterruption()
            self.stopThread()
            self.button(QtWidgets.QWizard.WizardButton.FinishButton).setEnabled(True)
            self.setButtonText(QtWidgets.QWizard.WizardButton.CancelButton, "Close")
            self.run_page.resetProgress()
            self.run_page.setStatus("Processing Canceled!")
            logger.warning("Batch processing canceled.")
        else:
            self.stopThread()
            self.process_thread.deleteLater()
            super().reject()

    def stopThread(self):
        self.process_thread.quit()
        self.process_thread.wait()


if __name__ == "__main__":
    from spcal.processing import SPCalProcessingMethod

    app = QtWidgets.QApplication()
    method = SPCalProcessingMethod()
    wiz = SPCalBatchProcessingWizard(None, method, [])
    wiz.open()
    app.exec()
