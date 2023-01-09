import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.detection import accumulate_detections, combine_detections
from spcal.gui.inputs import ReferenceWidget, SampleWidget
from spcal.gui.options import OptionsWidget
from spcal.gui.util import Worker
from spcal.io.nu import read_nu_directory, select_nu_signals
from spcal.io.text import export_single_particle_results, import_single_particle_file
from spcal.limit import SPCalLimit
from spcal.result import SPCalResult
from spcal.siunits import mass_units, molar_concentration_units, size_units

logger = logging.getLogger(__name__)

# Todo: filters?


def process_data(
    path: Path,
    data: np.ndarray,
    method: str,
    inputs: Dict[str, Dict[str, float | None]],
    limit_method: str,
    limit_params: Dict[str, float],
    limit_window_size: int = 0,
) -> Dict[str, SPCalResult]:

    # === Calculate Limits ===
    limits: Dict[str, SPCalLimit] = {}
    d, l, r = {}, {}, {}
    assert data.dtype.names is not None
    for name in data.dtype.names:
        if limit_method == "Manual Input":
            limits[name] = SPCalLimit(
                np.mean(data[name]),
                limit_params["manual"],
                limit_params["manual"],
                name="Manual Input",
                params={},
            )
        else:
            limits[name] = SPCalLimit.fromMethodString(
                limit_method,
                data[name],
                sigma=limit_params["sigma"],
                alpha=limit_params["alpha"],
                beta=limit_params["beta"],
                window_size=limit_window_size,
            )

        # === Create detections ===
        d[name], l[name], r[name] = accumulate_detections(
            data[name],
            limits[name].limit_of_criticality,
            limits[name].limit_of_detection,
        )

    detections, labels, regions = combine_detections(d, l, r)

    results = {
        name: SPCalResult(path, data[name], detections[name], labels, limits[name])
        for name in detections.dtype.names
    }

    # === Calculate results ===
    for name, result in results.items():
        assert inputs[name]["dwelltime"] is not None
        inputs[name]["time"] = result.events * inputs[name]["dwelltime"]  # type: ignore

        # No None inputs
        result.inputs.update({k: v for k, v in inputs[name].items() if v is not None})

        try:
            if method in ["Manual Input", "Reference Particle"]:
                result.fromNebulisationEfficiency()
            elif method == "Mass Response":
                result.fromMassResponse()
        except ValueError:
            pass

    return results


def process_text_file(
    path: Path,
    outpath: Path,
    import_options: dict,
    trim: Tuple[int, int],
    process_kws: dict,
    output_kws: dict,
) -> None:
    data, old_names = import_single_particle_file(
        path,
        delimiter=import_options["delimiter"],
        columns=import_options["columns"],
        first_line=import_options["first line"],
        new_names=import_options["names"],
        convert_cps=import_options["dwelltime"] if import_options["cps"] else None,
    )
    if old_names != import_options["old names"] or data.dtype.names is None:
        raise ValueError("different elements from sample")

    data = data[trim[0] : data.size - trim[1]]
    if data.size == 0:
        raise ValueError("data size zero")

    results = process_data(path, data, **process_kws)

    # === Export to file ===
    export_single_particle_results(outpath, results, **output_kws)


def process_nu_file(
    path: Path,
    outpath: Path,
    import_options: dict,
    trim: Tuple[int, int],
    process_kws: dict,
    output_kws: dict,
) -> None:
    masses, signals, info = read_nu_directory(path)

    selected_masses = {
        f"{i['Symbol']}{i['Isotope']}": i["Mass"]
        for i in import_options["selectedIsotopes"]
    }
    data = select_nu_signals(masses, signals, selected_masses=selected_masses)

    data = data[trim[0] : data.size - trim[1]]
    if data.size == 0:
        raise ValueError("data size zero")

    results = process_data(path, data, **process_kws)

    # === Export to file ===
    export_single_particle_results(outpath, results, **output_kws)


class BatchProcessDialog(QtWidgets.QDialog):
    fileprocessed = QtCore.Signal()
    processingStarted = QtCore.Signal()
    processingFinshed = QtCore.Signal()

    def __init__(
        self,
        files: List[str],
        sample: SampleWidget,
        reference: ReferenceWidget,
        options: OptionsWidget,
        best_units: Dict[str, Tuple[str, float]],
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Batch Process")
        self.setMinimumSize(640, 640)
        self.setAcceptDrops(True)

        self.sample = sample
        self.reference = reference
        self.options = options

        self.button_files = QtWidgets.QPushButton("Open Files")
        self.button_files.pressed.connect(self.dialogLoadFiles)
        self.button_output = QtWidgets.QPushButton("Open Directory")
        self.button_output.pressed.connect(self.dialogOpenOutputDir)

        self.button_process = QtWidgets.QPushButton("Start Batch")
        self.button_process.setEnabled(len(files) > 0)
        self.button_process.pressed.connect(self.buttonProcess)

        self.trim_left = QtWidgets.QCheckBox("Use sample left trim.")
        self.trim_left.setChecked(True)
        self.trim_right = QtWidgets.QCheckBox("Use sample right trim.")
        self.trim_right.setChecked(True)

        self.progress = QtWidgets.QProgressBar()
        self.threadpool = QtCore.QThreadPool()
        self.threadpool.setMaxThreadCount(1)
        self.aborted = False
        self.running = False

        self.files = QtWidgets.QListWidget()
        self.files.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.files.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.files.addItems(files)
        self.files.setTextElideMode(QtCore.Qt.ElideLeft)
        self.files.model().rowsInserted.connect(self.completeChanged)
        self.files.model().rowsRemoved.connect(self.completeChanged)

        self.inputs = QtWidgets.QGroupBox("Batch Options")
        self.inputs.setLayout(QtWidgets.QFormLayout())

        self.output_dir = QtWidgets.QLineEdit("")
        self.output_dir.setPlaceholderText("Same as input")
        self.output_dir.setToolTip("Leave blank to use the input directory.")
        self.output_dir.textChanged.connect(self.completeChanged)

        self.output_name = QtWidgets.QLineEdit("%_results.csv")
        self.output_name.setToolTip("Use '%' to represent the input file name.")
        self.output_name.textChanged.connect(self.completeChanged)

        self.inputs.layout().addRow("Output Name:", self.output_name)
        self.inputs.layout().addRow("Output Directory:", self.output_dir)
        self.inputs.layout().addWidget(self.button_output)
        self.inputs.layout().addRow(self.trim_left)
        self.inputs.layout().addRow(self.trim_right)

        self.mass_units = QtWidgets.QComboBox()
        self.mass_units.addItems(mass_units.keys())
        self.mass_units.setCurrentText(best_units["mass"][0])
        self.size_units = QtWidgets.QComboBox()
        self.size_units.addItems(size_units.keys())
        self.size_units.setCurrentText(best_units["size"][0])
        self.conc_units = QtWidgets.QComboBox()
        self.conc_units.addItems(molar_concentration_units.keys())
        self.conc_units.setCurrentText(best_units["cell_concentration"][0])

        units = QtWidgets.QGroupBox("Output Units")
        units.setLayout(QtWidgets.QFormLayout())
        units.layout().addRow("Mass units", self.mass_units)
        units.layout().addRow("Size units", self.size_units)
        units.layout().addRow("Conc. units", self.conc_units)

        self.check_export_inputs = QtWidgets.QCheckBox("Export options and inputs.")
        self.check_export_inputs.setChecked(True)
        self.check_export_arrays = QtWidgets.QCheckBox(
            "Export detected particle arrays."
        )
        self.check_export_arrays.setChecked(True)
        self.check_export_compositions = QtWidgets.QCheckBox(
            "Export peak compositions."
        )

        switches = QtWidgets.QGroupBox("Output Options")
        switches.setLayout(QtWidgets.QVBoxLayout())
        switches.layout().addWidget(self.check_export_inputs)
        switches.layout().addWidget(self.check_export_arrays)
        switches.layout().addWidget(self.check_export_compositions)

        self.import_options = QtWidgets.QGroupBox("Import Options")
        self.import_options.setLayout(QtWidgets.QFormLayout())

        le_delimiter = QtWidgets.QLineEdit(self.sample.import_options["delimiter"])
        le_delimiter.setReadOnly(True)
        sb_first_line = QtWidgets.QSpinBox()
        sb_first_line.setValue(self.sample.import_options["first line"])
        sb_first_line.setReadOnly(True)
        te_columns = QtWidgets.QTextEdit()
        te_columns.setPlainText(
            "\n".join(
                f"{c} :: {n}"
                for c, n in zip(
                    self.sample.import_options["columns"],
                    self.sample.import_options["names"],
                )
            )
        )
        te_columns.setReadOnly(True)
        le_units = QtWidgets.QLineEdit(
            "CPS" if self.sample.import_options["cps"] else "Counts"
        )
        le_units.setReadOnly(True)

        self.import_options.layout().addRow("Delimiter:", le_delimiter)

        self.import_options.layout().addRow("Import from row:", sb_first_line)
        self.import_options.layout().addRow("Use Columns:", te_columns)
        self.import_options.layout().addRow("Intensity Units:", le_units)

        layout_list = QtWidgets.QVBoxLayout()
        layout_list.addWidget(self.button_files, 0, QtCore.Qt.AlignLeft)
        layout_list.addWidget(self.files, 1)

        layout_right = QtWidgets.QVBoxLayout()
        layout_right.addWidget(self.inputs, 0)
        layout_right.addWidget(self.import_options, 0)
        layout_right.addWidget(units, 0)
        layout_right.addWidget(switches, 0)

        layout_horz = QtWidgets.QHBoxLayout()
        layout_horz.addLayout(layout_list)
        layout_horz.addLayout(layout_right, 0)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(layout_horz)
        layout.addWidget(self.progress, 0)
        layout.addWidget(self.button_process, 0, QtCore.Qt.AlignRight)

        self.setLayout(layout)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:  # pragma: no cover
            super().dragEnterEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                self.files.addItem(url.toLocalFile())
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() in [QtCore.Qt.Key_Backspace, QtCore.Qt.Key_Delete]:
            items = self.files.selectedIndexes()
            for item in reversed(sorted(items)):
                self.files.model().removeRow(item.row())
        else:
            super().keyPressEvent(event)

    def completeChanged(self) -> None:
        complete = self.isComplete()
        self.button_process.setEnabled(complete)

    def isComplete(self) -> bool:
        if self.files.count() == 0:
            return False
        if "%" not in self.output_name.text():
            return False
        if any(x in self.output_name.text() for x in "<>:/\\|?*"):
            return False
        if self.output_dir.text() != "" and not Path(self.output_dir.text()).is_dir():
            return False

        return True

    def dialogLoadFiles(self) -> None:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Batch Process Files",
            "",
            (
                "NP Data Files (*.csv *.info);;CSV Documents(*.csv *.txt *.text);;"
                "Nu Instruments(*.info);;All files(*)"
            ),
        )

        if len(files) > 0:
            self.files.addItems(files)

    def dialogOpenOutputDir(self) -> None:
        dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Output Directory", "", QtWidgets.QFileDialog.ShowDirsOnly
        )
        if dir != "":
            self.output_dir.setText(dir)

    def outputsForFiles(self, files: List[Path]) -> List[Path]:
        outdir = self.output_dir.text()

        outputs = []
        for file in files:
            outname = self.output_name.text().replace("%", file.stem)
            if outdir == "":
                outdir = file.parent
            outputs.append(Path(outdir, outname))

        return outputs

    def buttonProcess(self) -> None:
        if self.running:
            self.abort()
        else:
            self.button_process.setText("Cancel Batch")
            self.start()

    def abort(self) -> None:
        self.aborted = True
        self.threadpool.clear()
        self.threadpool.waitForDone()

        self.progress.reset()
        self.button_process.setText("Start Batch")
        self.running = False

    def start(self) -> None:
        infiles = [Path(self.files.item(i).text()) for i in range(self.files.count())]
        outfiles = self.outputsForFiles(infiles)

        self.completed_files: List[str] = []
        self.failed_files: List[Tuple[str, str]] = []

        self.aborted = False
        self.running = True
        self.progress.setMaximum(len(infiles))
        self.progress.setValue(1)

        method = self.options.efficiency_method.currentText()

        inputs: Dict[str, Dict[str, float | None]] = {}

        assert self.sample.responses.dtype.names is not None
        trim = self.sample.trimRegion(self.sample.responses.dtype.names[0])
        # trims converted to left, -right
        trim = trim[0], self.sample.responses.size - trim[1]
        if not self.trim_left.isChecked():
            trim = 0, trim[1]
        if not self.trim_right.isChecked():
            trim = trim[0], 0

        for name in self.sample.responses.dtype.names:
            inputs[name] = {
                "dwelltime": self.options.dwelltime.baseValue(),
                "uptake": self.options.uptake.baseValue(),
                "cell_diameter": self.options.celldiameter.baseValue(),
                "molar_mass": self.sample.io[name].molarmass.baseValue(),
                "density": self.sample.io[name].density.baseValue(),
                "response": self.sample.io[name].response.baseValue(),
            }
            try:
                inputs[name]["mass_fraction"] = float(
                    self.sample.io[name].massfraction.text()
                )
            except ValueError:
                pass
            try:
                if method == "Manual Input":
                    inputs[name]["efficiency"] = float(self.options.efficiency.text())
                elif method == "Reference Particle":
                    inputs[name]["efficiency"] = self.reference.getEfficiency(name)
                elif method == "Mass Response":
                    inputs[name]["mass_response"] = self.reference.io[
                        name
                    ].massresponse.baseValue()
            except ValueError:
                pass

        limit_params = {
            "sigma": float(self.options.sigma.text()),
            "alpha": float(self.options.error_rate_alpha.text()),
            "beta": float(self.options.error_rate_beta.text()),
            "manual": float(self.options.manual.text() or 0.0),
        }
        units = {
            "mass": (
                self.mass_units.currentText(),
                mass_units[self.mass_units.currentText()],
            ),
            "size": (
                self.size_units.currentText(),
                size_units[self.size_units.currentText()],
            ),
            "conc": (
                self.conc_units.currentText(),
                molar_concentration_units[self.conc_units.currentText()],
            ),
        }
        fn = (
            process_text_file
            if self.sample.import_options["importer"] == "text"
            else process_nu_file
        )
        for path, outpath in zip(infiles, outfiles):
            worker = Worker(
                fn,
                path,
                outpath,
                import_options=self.sample.import_options,
                trim=trim,
                process_kws={
                    "method": method,
                    "inputs": inputs,
                    "limit_method": self.options.method.currentText(),
                    "limit_params": limit_params,
                    "limit_window_size": int(self.options.window_size.text())
                    if self.options.window_size.isEnabled()
                    else 0,
                },
                output_kws={
                    "units_for_results": units,
                    "output_inputs": self.check_export_inputs.isChecked(),
                    "output_compositions": self.check_export_compositions.isChecked(),
                    "output_arrays": self.check_export_arrays.isChecked(),
                },
            )
            worker.signals.finished.connect(self.workerComplete)
            worker.signals.exception.connect(self.workerFailed)
            self.threadpool.start(worker)

        self.processingStarted.emit()

    def finalise(self) -> None:
        self.threadpool.waitForDone()
        self.running = False

        self.progress.reset()
        self.button_process.setText("Start Batch")

        self.processingFinshed.emit()

    def workerComplete(self) -> None:
        if self.aborted:
            return

        self.progress.setValue(self.progress.value() + 1)
        if self.threadpool.activeThreadCount() == 0 and self.running:
            self.finalise()

    def workerFailed(self, exception: Exception) -> None:
        if self.aborted:
            return

        self.abort()

        logger.exception(exception)

        msg = QtWidgets.QMessageBox(
            QtWidgets.QMessageBox.Warning,
            "Batch Process Failed",
            str(exception),
            parent=self,
        )
        msg.exec()
