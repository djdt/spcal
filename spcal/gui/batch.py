import logging
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

import spcal
from spcal.gui.inputs import ReferenceWidget, SampleWidget
from spcal.gui.options import OptionsWidget
from spcal.io import export_nanoparticle_results

logger = logging.getLogger(__name__)


# Todo: warn if files have different elements
# Todo: update fro SPCalLimits / Results


def process_file_detections(
    file: Path,
    options: dict,
    trims: Dict[str, Tuple[int, int]],
    limit_method: str,
    limit_sigma: float,
    limit_error_rates: Tuple[float, float],
    limit_manual: float,
    limit_window: int | None = None,
) -> dict:
    responses = np.genfromtxt(
        file,
        delimiter=options["delimiter"],
        usecols=options["columns"],
        names=options["headers"],
        skip_header=options["first line"],
        converters={0: lambda s: float(s.replace(",", "."))},
        invalid_raise=False,
    )
    if responses.size == 0 or responses.dtype.names is None:
        raise ValueError(f"Unabled to import file '{file.name}'.")

    if options["cps"]:
        dwell = options["dwelltime"]
        for name in responses.dtype.names:
            responses[name] *= dwell  # type: ignore

    results = {}
    for name in responses.dtype.names:
        data = responses[name][trims[name][0] : responses.size - trims[name][1]]

        if limit_method == "Manual Input":
            limits = (
                limit_method,
                {},
                np.array(
                    [(np.mean(data), limit_manual, limit_manual)],
                    dtype=calculate_limits.dtype,
                ),
            )
        else:
            limits = calculate_limits(
                data,
                limit_method,
                limit_sigma,
                limit_error_rates,
                window=limit_window,
            )

        detections, labels, _ = spcal.accumulate_detections(
            data, limits[2]["lc"], limits[2]["ld"]
        )
        results[name] = {
            "detections": detections,
            "detections_std": np.sqrt(detections.size),
            "background": np.mean(data[labels == 0]),
            "background_std": np.std(data[labels == 0]),
            "events": data.size,
            "file": str(file),
            "limit_method": f"{limits[0]},{','.join(f'{k}={v}' for k,v in limits[1].items())}",
            "lod": limits[2]["ld"],
            "inputs": {"dwelltime": options["dwelltime"]},
        }

    return results


class ProcessThread(QtCore.QThread):
    processComplete = QtCore.Signal(str)
    processFailed = QtCore.Signal(str)

    def __init__(
        self,
        infiles: List[Path],
        outfiles: List[Path],
        import_options: dict,
        method: Callable,
        method_kws: Dict[str, Dict[str, float | None]],
        cell_kws: Dict[str, Dict[str, float | None]],
        trims: Dict[str, Tuple[int, int]],
        limit_method: str = "Automatic",
        limit_sigma: float = 3.0,
        limit_error_rates: Tuple[float, float] = (0.05, 0.05),
        limit_manual: float = 0.0,
        limit_window: int | None = None,
        cps_dwelltime: float | None = None,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent)

        self.infiles = infiles
        self.outfiles = outfiles
        self.import_options = import_options

        self.method = method
        self.method_kws = method_kws
        self.cell_kws = cell_kws

        self.trims = trims

        self.limit_method = limit_method
        self.limit_error_rates = limit_error_rates
        self.limit_sigma = limit_sigma
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
                    self.limit_sigma,
                    self.limit_error_rates,
                    limit_manual=self.limit_manual,
                    limit_window=self.limit_window,
                )
            except ValueError as e:
                logger.exception(e)
                self.processFailed.emit(infile.name)
                continue

            try:
                for name in results:
                    self.method_kws[name]["time"] = (
                        results[name]["events"] * self.method_kws[name]["dwelltime"]
                    )
                    if any(x is None for x in self.method_kws[name].values()):
                        logger.warning(
                            f"{infile}:{name}, missing inputs for calibrated results."
                        )
                    else:
                        results[name].update(
                            self.method(
                                results[name]["detections"],
                                results[name]["background"],
                                results[name]["lod"],
                                **self.method_kws[name],
                            )
                        )
                        results[name]["inputs"] = {
                            k: v
                            for k, v in self.method_kws[name].items()
                            if v is not None
                        }

                    if self.cell_kws[name]["cell_diameter"] is not None:
                        if self.cell_kws[name]["molar_mass"] is None:
                            logger.warning(
                                f"{infile}:{name}, missing inputs for cell results."
                            )
                        else:
                            results[name][
                                "cell_concentrations"
                            ] = spcal.cell_concentration(
                                results[name]["masses"],
                                diameter=self.cell_kws[name]["cell_diameter"],
                                molar_mass=self.cell_kws[name]["molar_mass"],
                            )
                            results[name][
                                "lod_cell_concentration"
                            ] = spcal.cell_concentration(
                                results[name]["lod_mass"],
                                diameter=self.cell_kws[name]["cell_diameter"],
                                molar_mass=self.cell_kws[name]["molar_mass"],
                            )
                            results[name]["inputs"].update(self.cell_kws[name])

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

        self.completed_files = []
        self.failed_files = []

        self.progress.setMaximum(len(infiles))
        self.progress.setValue(1)

        method = self.options.efficiency_method.currentText()

        if method in ["Manual Input", "Reference Particle"]:
            method_fn = results_from_nebulisation_efficiency
        elif method == "Mass Response":
            method_fn = results_from_mass_response
        else:
            raise ValueError("Unknown method")

        trims = np.array(
            [self.sample.trimRegion(name) for name in self.sample.detections]
        )
        trim_average = np.mean(trims[:, 0]), self.sample.responses.size - np.mean(
            trims[:, 1]
        )
        trim_max = np.amax(trims[:, 0]), self.sample.responses.size - np.amin(
            trims[:, 1]
        )

        trims = {}
        method_kws = {}
        cell_kws = {}
        for name in self.sample.detections:
            # trims converted to left, -right
            if self.combo_trim.currentText() == "None":
                trims[name] = 0, 0
            elif self.combo_trim.currentText() == "As Sample":
                trim = self.sample.trimRegion(name)
                trims[name] = trim[0], self.sample.responses.size - trim[1]
            elif self.combo_trim.currentText() == "Average":
                trims[name] = trim_average
            elif self.combo_trim.currentText() == "Maximum":
                trims[name] = trim_max

            if not self.trim_left.isChecked():
                trims[name] = 0, trims[name][1]
            if not self.trim_right.isChecked():
                trims[name] = trims[name][0], 0

            if method in ["Manual Input", "Reference Particle"]:
                try:
                    if method == "Manual Input":
                        efficiency = float(self.options.efficiency.text())
                    elif method == "Reference Particle" and name in self.reference.io:
                        efficiency = float(self.reference.io[name].efficiency.text())
                    else:
                        efficiency = None
                except ValueError:
                    efficiency = None

                method_kws[name] = {
                    "density": self.sample.io[name].density.baseValue(),
                    "dwelltime": self.options.dwelltime.baseValue(),
                    "efficiency": efficiency,
                    "mass_fraction": float(self.sample.io[name].massfraction.text()),
                    "uptake": self.options.uptake.baseValue(),
                    "response": self.sample.io[name].response.baseValue(),
                }
            elif method == "Mass Response":
                method_kws = {
                    "density": self.sample.io[name].density.baseValue(),
                    "dwelltime": self.options.dwelltime.baseValue(),
                    "mass_fraction": float(self.sample.io[name].massfraction.text()),
                    "mass_response": self.reference.massresponse.baseValue(),
                }
            else:
                raise ValueError("Unknown method")

            cell_kws[name] = {
                "cell_diameter": self.options.celldiameter.baseValue(),
                "molar_mass": self.sample.io[name].molarmass.baseValue(),
            }

        self.thread = ProcessThread(
            infiles,
            outfiles,
            self.sample.import_options,
            method_fn,
            method_kws,
            cell_kws=cell_kws,
            trims=trims,
            limit_method=self.options.method.currentText(),
            limit_sigma=float(self.options.sigma.text()),
            limit_error_rates=(
                float(self.options.error_rate_alpha.text()),
                float(self.options.error_rate_beta.text()),
            ),
            limit_manual=float(self.options.manual.text() or 0.0),
            limit_window=(
                int(self.options.window_size.text())
                if self.options.window_size.isEnabled()
                else None
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
