import logging
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

import spcal
from spcal.calc import SPCalLimit, SPCalResult
from spcal.gui.inputs import ReferenceWidget, SampleWidget
from spcal.gui.options import OptionsWidget
from spcal.io import export_nanoparticle_results

logger = logging.getLogger(__name__)


# Todo: warn if files have different elements
# Todo: update fro SPCalLimits / Results


def process_file_detections(
    file: Path,
    import_options: dict,
    trims: Dict[str, Tuple[int, int]],
    limit_method: str,
    limit_params: Dict[str, float],
    limit_manual: float,
    limit_window: int = 0,
) -> dict:
    responses = np.genfromtxt(
        file,
        delimiter=import_options["delimiter"],
        usecols=import_options["columns"],
        names=import_options["headers"],
        skip_header=import_options["first line"],
        converters={0: lambda s: float(s.replace(",", "."))},
        invalid_raise=False,
    )
    if responses.size == 0 or responses.dtype.names is None:
        raise ValueError(f"Unabled to import file '{file.name}'.")

    if import_options["cps"]:
        dwell = import_options["dwelltime"]
        for name in responses.dtype.names:
            responses[name] *= dwell  # type: ignore

    results: Dict[str, SPCalResult] = {}
    for name in responses.dtype.names:
        response = responses[name][trims[name][0] : responses.size - trims[name][1]]  # type: ignore
        if limit_method == "Manual Input":
            limits = SPCalLimit(
                np.mean(response),
                limit_manual,
                limit_manual,
                name="Manual Input",
                params={},
            )
        elif limit_method == "Automatic":
            limits = SPCalLimit.fromBest(
                response,
                sigma=limit_params["sigma"],
                alpha=limit_params["alpha"],
                beta=limit_params["beta"],
                window_size=limit_window,
            )
        elif limit_method == "Highest":
            limits = SPCalLimit.fromHighest(
                response,
                sigma=limit_params["sigma"],
                alpha=limit_params["alpha"],
                beta=limit_params["beta"],
                window_size=limit_window,
            )
        elif limit_method.startswith("Guassian"):
            limits = SPCalLimit.fromGaussian(
                response,
                sigma=limit_params["sigma"],
                window_size=limit_window,
                use_median="median" in limit_method.lower(),
            )
        else:
            limits = SPCalLimit.fromPoisson(
                response,
                alpha=limit_params["alpha"],
                beta=limit_params["beta"],
                window_size=limit_window,
                use_median="median" in limit_method.lower(),
            )

        detections, labels, _ = spcal.accumulate_detections(
            response, limits.limit_of_criticality, limits.limit_of_detection
        )

        results[name] = SPCalResult(file, responses, detections, labels, limits)

    return results


class ProcessThread(QtCore.QThread):
    processComplete = QtCore.Signal(str)
    processFailed = QtCore.Signal(str)

    def __init__(
        self,
        infiles: List[Path],
        outfiles: List[Path],
        import_options: dict,
        inputs: Dict[str, Dict[str, float | None]],
        method: str,
        trims: Dict[str, Tuple[int, int]],
        limit_method: str = "Automatic",
        limit_params: Dict[str, float] | None = None,
        limit_manual: float = 0.0,
        limit_window: int = 0,
        cps_dwelltime: float | None = None,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent)

        self.infiles = infiles
        self.outfiles = outfiles
        self.import_options = import_options

        self.method = method
        self.inputs = {k: v for k, v in inputs.items() if v is not None}

        self.trims = trims

        self.limit_method = limit_method
        self.limit_params = {"sigma": 3.0, "alpha": 0.05, "beta": 0.05}
        if limit_params is not None:
            self.limit_params.update(limit_params)
        self.limit_manual = limit_manual
        self.limit_window = limit_window
        self.cps_dwelltime = cps_dwelltime

    def run(self) -> None:
        for infile, outfile in zip(self.infiles, self.outfiles):
            if self.isInterruptionRequested():
                break
            try:
                results = process_file_detections(
                    infile,
                    self.import_options,
                    self.trims,
                    self.limit_method,
                    limit_params=self.limit_params,
                    limit_manual=self.limit_manual,
                    limit_window=self.limit_window,
                )
            except ValueError as e:
                logger.exception(e)
                self.processFailed.emit(infile.name)
                continue

            try:
                for name, result in results.items():
                    self.inputs[name]["time"] = (
                        result.response * self.inputs[name]["dwelltime"]
                    )

                    # No None inputs
                    result.inputs.update(
                        {k: v for k, v in self.inputs[name].items() if v is not None}
                    )

                    try:
                        if self.method in ["Manual Input", "Reference Particle"]:
                            result.fromNebulisationEfficiency()
                        elif self.method == "Mass Response":
                            result.fromMassResponse()
                    except ValueError:
                        pass

            except ValueError as e:
                logger.exception(e)
                self.processFailed.emit(infile.name)
                continue

            try:
                export_nanoparticle_results(outfile, results)
            except ValueError as e:
                logger.exception(e)
                self.processFailed.emit(infile.name)
                continue

            self.processComplete.emit(infile.name)


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
        self.button_output.pressed.connect(self.dialogOpenOuputDir)

        self.button_process = QtWidgets.QPushButton("Start Batch")
        self.button_process.setEnabled(len(files) > 0)
        self.button_process.pressed.connect(self.buttonProcess)

        self.combo_trim = QtWidgets.QComboBox()
        self.combo_trim.addItems(["None", "As Sample", "Average", "Maximum"])
        self.combo_trim.setCurrentText("As Sample")
        for i, tooltip in enumerate(
            [
                "Ignore sample trim and do not trim any data.",
                "Use per element trim from currently loaded sample",
                "Use average of all sample element trims.",
                "Use maximum of all sample element trims.",
            ]
        ):
            self.combo_trim.setItemData(i, tooltip, QtCore.Qt.ToolTipRole)

        self.trim_left = QtWidgets.QCheckBox("Use sample left trims.")
        self.trim_left.setChecked(True)
        self.trim_right = QtWidgets.QCheckBox("Use sample right trims.")
        self.trim_right.setChecked(True)

        self.progress = QtWidgets.QProgressBar()
        self.thread: QtCore.QThread | None = None

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

        self.output_name = QtWidgets.QLineEdit("%_result.csv")
        self.output_name.setToolTip("Use '%' to represent the input file name.")
        self.output_name.textChanged.connect(self.completeChanged)

        self.inputs.layout().addRow("Output Name:", self.output_name)
        self.inputs.layout().addRow("Output Directory:", self.output_dir)
        self.inputs.layout().addWidget(self.button_output)
        self.inputs.layout().addRow("Trim:", self.combo_trim)
        self.inputs.layout().addRow(self.trim_left)
        self.inputs.layout().addRow(self.trim_right)

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
                    self.sample.import_options["headers"],
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
            "CSV Documents(*.csv *.txt *.text);;All files(*)",
        )

        if len(files) > 0:
            self.files.addItems(files)

    def dialogOpenOuputDir(self) -> None:
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
        if self.thread is None:
            self.button_process.setText("Cancel Batch")
            self.startProcess()
        elif self.thread.isRunning():
            self.thread.requestInterruption()

    def advanceProgress(self) -> None:
        self.progress.setValue(self.progress.value() + 1)

    def processComplete(self, file: str) -> None:
        self.completed_files.append(file)
        self.advanceProgress()

    def processFailed(self, file: str) -> None:
        self.failed_files.append(file)
        self.advanceProgress()

    def startProcess(self) -> None:
        infiles = [Path(self.files.item(i).text()) for i in range(self.files.count())]
        outfiles = self.outputsForFiles(infiles)

        self.completed_files: List[str] = []
        self.failed_files: List[str] = []

        self.progress.setMaximum(len(infiles))
        self.progress.setValue(1)

        method = self.options.efficiency_method.currentText()

        trims = {}
        inputs: Dict[str, Dict[str, float | None]] = {}
        for name in self.sample.detections:
            # trims converted to left, -right
            if self.combo_trim.currentText() == "None":
                trims[name] = 0, 0
            elif self.combo_trim.currentText() == "As Sample":
                trim = self.sample.trimRegion(name)
                trims[name] = trim[0], self.sample.responses.size - trim[1]

            if not self.trim_left.isChecked():
                trims[name] = 0, trims[name][1]
            if not self.trim_right.isChecked():
                trims[name] = trims[name][0], 0

            inputs[name] = {
                "dwelltime": self.options.dwelltime.baseValue(),
                "uptake": self.options.uptake.baseValue(),
                "cell_diameter": self.options.celldiameter.baseValue(),
                "molar_mass": self.sample.io[name].molarmass.baseValue(),
                "density": self.sample.io[name].density.baseValue(),
                "response": self.sample.io[name].response.baseValue(),
            }
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

        self.thread = ProcessThread(
            infiles,
            outfiles,
            import_options=self.sample.import_options,
            inputs=inputs,
            method=method,
            trims=trims,
            limit_method=self.options.method.currentText(),
            limit_params={
                "sigma": float(self.options.sigma.text()),
                "alpha": float(self.options.error_rate_alpha.text()),
                "beta": float(self.options.error_rate_beta.text()),
            },
            limit_manual=float(self.options.manual.text() or 0.0),
            limit_window=(
                int(self.options.window_size.text())
                if self.options.window_size.isEnabled()
                else 0
            ),
            parent=self,
        )

        self.thread.processComplete.connect(self.processComplete)
        self.thread.processFailed.connect(self.processFailed)
        self.thread.finished.connect(self.finishProcess)
        self.thread.start()

    def finishProcess(self) -> None:
        self.button_process.setText("Start Batch")
        self.progress.setValue(0)
        self.thread = None

        if len(self.failed_files) > 0:
            msg = QtWidgets.QMessageBox(
                QtWidgets.QMessageBox.Warning,
                "Import Failed",
                f"Failed to process {len(self.failed_files)} files!",
                parent=self,
            )
            newline = "\n"
            msg.setDetailedText(f"\n{newline.join(f for f in self.failed_files)}")
            msg.exec_()
