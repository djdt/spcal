from pathlib import Path
from typing import Callable, Generator
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.datafile import SPCalDataFile, SPCalNuDataFile, SPCalTOFWERKDataFile
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
from spcal.io.nu import is_nu_directory, is_nu_run_info_file
from spcal.io.text import is_text_file
from spcal.io.tofwerk import is_tofwerk_file
from spcal.isotope import SPCalIsotope, SPCalIsotopeBase
from spcal.gui.widgets.periodictable import PeriodicTableSelector
from spcal.processing.method import SPCalProcessingMethod
from spcal.processing.options import SPCalIsotopeOptions


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
        self, existing_file: SPCalDataFile, parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(parent)
        self.setTitle("SPCal Batch Processing")
        self.setSubTitle("Select Data Files")

        label = QtWidgets.QLabel("Select files for batch processing.")

        self.radio_text = QtWidgets.QRadioButton("Text Exports")
        self.radio_nu = QtWidgets.QRadioButton("Nu Instruments")
        self.radio_tofwerk = QtWidgets.QRadioButton("TOFWERK HDF5")

        self.files = QtWidgets.QListWidget()
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

        self.registerField("paths", self, "paths_prop")

    def addFile(self, path: Path):
        item = QtWidgets.QListWidgetItem()
        item.setText(str(path))
        item.setIcon(QtGui.QIcon.fromTheme("list-remove"))
        self.files.addItem(item)

    def paths(self) -> list[Path]:
        return [Path(self.files.item(i).text()) for i in range(self.files.count())]

    def setPaths(self, paths: list[Path]):
        self.files.clear()
        for path in paths:
            self.addFile(path)

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
        return self.files.count() > 0

    def validatePage(self) -> bool:
        format = self.selectedFormat()
        file_fn = BatchFilesWizardPage.FORMAT_FUNCTIONS[format]
        for path in self.paths():
            if not file_fn(path):
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

    paths_prop = QtCore.Property("QVariant", paths, setPaths, notify=pathsChanged)  # type: ignore


class BatchTextWizardPage(QtWidgets.QWizardPage):
    def __init__(
        self, isotopes: list[SPCalIsotopeBase], parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(parent)
        self.setTitle("SPCal Batch Processing")
        self.setSubTitle("Text Options")


class BatchNuWizardPage(QtWidgets.QWizardPage):
    def __init__(
        self,
        isotopes: list[SPCalIsotopeBase],
        max_mass_diff: float,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setTitle("SPCal Batch Processing")
        self.setSubTitle("Nu Options")

        self.cycle_number = QtWidgets.QSpinBox()
        self.cycle_number.setValue(0)
        self.cycle_number.setSpecialValueText("All")

        self.segment_number = QtWidgets.QSpinBox()
        self.segment_number.setValue(0)
        self.segment_number.setSpecialValueText("All")

        self.max_mass_diff = QtWidgets.QDoubleSpinBox()
        self.max_mass_diff.setRange(0.0, 1.0)
        self.max_mass_diff.setValue(max_mass_diff)

        self.check_chunked = QtWidgets.QCheckBox("Split files.")
        self.check_chunked.checkStateChanged.connect(self.onChunkChecked)

        self.chunk_size = QtWidgets.QSpinBox()
        self.chunk_size.setRange(1, 10000)
        self.chunk_size.setValue(1000)
        self.chunk_size.setSingleStep(100)
        self.chunk_size.setEnabled(False)

        self.table = PeriodicTableSelector()
        self.table.setSelectedIsotopes(
            [iso for iso in isotopes if isinstance(iso, SPCalIsotope)]
        )

        layout_chunk = QtWidgets.QHBoxLayout()
        layout_chunk.addWidget(self.chunk_size)
        layout_chunk.addWidget(self.check_chunked)

        options_box = QtWidgets.QGroupBox("Options")
        options_box_layout = QtWidgets.QFormLayout()
        # options_box_layout.addRow("Max diff m/z:", self.max_mass_diff)
        options_box_layout.addRow("Cycle:", self.cycle_number)
        options_box_layout.addRow("Segment:", self.segment_number)
        options_box_layout.addRow("Chunk size:", layout_chunk)
        options_box.setLayout(options_box_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(options_box, 0)
        layout.addWidget(self.table)
        self.setLayout(layout)

        self.registerField("nu.cycle_number", self.cycle_number)
        self.registerField("nu.segment_number", self.segment_number)
        self.registerField("nu.max_mass_diff", self.max_mass_diff)
        self.registerField("nu.isotopes", self.table, "selected_isotopes_prop")

    def initializePage(self):
        paths: list[Path] = self.field("paths")
        paths = [path.parent if is_nu_run_info_file(path) else path for path in paths]

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
        self.method = method

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

        top_layout = QtWidgets.QHBoxLayout()
        top_layout.addWidget(self.instrument_options)
        top_layout.addWidget(self.limit_options)
        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(top_layout)
        layout.addWidget(self.isotope_table)
        self.setLayout(layout)

    def initializePage(self):
        if self.wizard().hasVisitedPage(SPCalBatchProcessingWizard.TEXT_PAGE_ID):
            isotopes = self.field("text.isotopes")
        elif self.wizard().hasVisitedPage(SPCalBatchProcessingWizard.NU_PAGE_ID):
            isotopes = self.field("nu.isotopes")
        elif self.wizard().hasVisitedPage(SPCalBatchProcessingWizard.TOFWERK_PAGE_ID):
            isotopes = self.field("tofwerk.isotopes")
        else:
            raise ValueError("has not visited any format pages")

        self.isotope_table.isotope_model.beginResetModel()
        self.isotope_table.isotope_model.isotope_options = {
            isotope: self.method.isotope_options.get(
                isotope, SPCalIsotopeOptions(None, None, None)
            )
            for isotope in isotopes
        }
        self.isotope_table.isotope_model.endResetModel()


class BatchRunWizardPage(QtWidgets.QWizardPage):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setTitle("SPCal Batch Processing")
        self.setSubTitle("Run Batch Processing")


class SPCalBatchProcessingWizard(QtWidgets.QWizard):
    TEXT_PAGE_ID = 1
    NU_PAGE_ID = 2
    TOFWERK_PAGE_ID = 3
    METHOD_PAGE_ID = 4

    def __init__(
        self,
        existing_file: SPCalDataFile,
        method: SPCalProcessingMethod,
        selected_isotopes: list[SPCalIsotopeBase],
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Batch Processing")

        self.setPage(0, BatchFilesWizardPage(existing_file))
        self.setPage(
            SPCalBatchProcessingWizard.TEXT_PAGE_ID,
            BatchTextWizardPage(selected_isotopes),
        )
        max_mass_diff = (
            existing_file.max_mass_diff
            if isinstance(existing_file, SPCalNuDataFile)
            else 0.1
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


if __name__ == "__main__":
    from spcal.processing import SPCalProcessingMethod

    app = QtWidgets.QApplication()
    df = SPCalNuDataFile.load(
        Path("/home/tom/Downloads/14-38-58 UPW + 80nm Au 90nm UCNP many particles/")
    )
    method = SPCalProcessingMethod()
    wiz = SPCalBatchProcessingWizard(df, method, [])
    # wiz.page_files.addFile(
    #     Path("/home/tom/Downloads/NT032/14-37-30 1 ppb att/run.info")
    # )
    wiz.show()
    app.exec()
