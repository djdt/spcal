from PySide2 import QtCore, QtGui, QtWidgets

import numpy as np
from pathlib import Path

import spcal

from spcal.calc import (
    calculate_limits,
    results_from_mass_response,
    results_from_nebulisation_efficiency,
)
from spcal.io import read_nanoparticle_file, export_nanoparticle_results

from spcal.gui.inputs import SampleWidget, ReferenceWidget
from spcal.gui.options import OptionsWidget

from typing import Callable, Dict, List, Optional, Tuple


def process_file_detections(
    file: Path,
    trim: Tuple[Optional[int], Optional[int]],
    limit_method: str,
    limit_sigma: float,
    limit_error_rates: Tuple[float, float],
    limit_manual: float,
    limit_window: int = None,
    cps_dwelltime: float = None,
) -> dict:
    responses, _ = read_nanoparticle_file(file, delimiter=",")
    responses = responses[trim[0] : trim[1]]

    # Convert to counts if required
    if cps_dwelltime is not None:
        responses = responses * cps_dwelltime

    size = responses.size

    if responses is None or size == 0:
        raise ValueError(f"Unabled to import file '{file.name}'.")

    if limit_method == "Manual Input":
        method = "Manual Input"
        method_dict = {}
        lc = ld = limit_manual
    else:
        method, method_dict, (_, lc, ld) = calculate_limits(
            responses,
            limit_method,
            limit_sigma,
            limit_error_rates,
            window=limit_window,
        )

    # if limits is None:
    #     raise ValueError("Limit calculations failed for '{file.name}'.")

    detections, labels, regions = spcal.accumulate_detections(responses, lc, ld)
    background = np.mean(responses[labels == 0])
    background_std = np.std(responses[labels == 0])

    centers = (regions[:, 0] + regions[:, 1]) // 2
    values = np.linspace(0, responses.size, 3 + 1)
    indicies = np.searchsorted(centers, values, side="left")
    detections_std = np.std(np.diff(indicies))

    return {
        "background": background,
        "background_std": background_std,
        "detections": detections,
        "detections_std": detections_std,
        "events": size,
        "file": str(file),
        "limit_method": f"{method},{','.join(f'{k}={v}' for k,v in method_dict.items())}",
        "limit_window": limit_window,
        "lod": ld,
    }


class ProcessThread(QtCore.QThread):
    processComplete = QtCore.Signal(str)
    processFailed = QtCore.Signal(str)

    def __init__(
        self,
        infiles: List[Path],
        outfiles: List[Path],
        method: Callable,
        method_kws: Dict[str, Optional[float]],
        cell_kws: Dict[str, Optional[float]],
        trim: Tuple[Optional[int], Optional[int]] = (None, None),
        limit_method: str = "Automatic",
        limit_sigma: float = 3.0,
        limit_error_rates: Tuple[float, float] = (0.05, 0.05),
        limit_manual: float = 0.0,
        limit_window: int = None,
        cps_dwelltime: float = None,
        parent: QtCore.QObject = None,
    ):
        super().__init__(parent)

        self.infiles = infiles
        self.outfiles = outfiles

        self.method = method
        self.method_kws = method_kws
        self.cell_kws = cell_kws

        self.trim = trim

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
                result = process_file_detections(
                    infile,
                    self.trim,
                    self.limit_method,
                    self.limit_sigma,
                    self.limit_error_rates,
                    limit_manual=self.limit_manual,
                    limit_window=self.limit_window,
                    cps_dwelltime=self.cps_dwelltime,
                )
            except ValueError:
                self.processFailed.emit(infile.name)
                continue

            try:
                result.update(
                    self.method(
                        result["detections"],
                        result["background"],
                        result["lod"],
                        **self.method_kws,
                    )
                )
                result["inputs"] = {k: v for k, v in self.method_kws.items()}

                if (
                    self.cell_kws["celldiameter"] is not None
                    and self.cell_kws["molarmass"] is not None
                ):
                    result["cell_concentrations"] = spcal.cell_concentration(
                        result["masses"],
                        diameter=self.cell_kws["celldiameter"],
                        molarmass=self.cell_kws["molarmass"],
                    )
                    result["lod_cell_concentration"] = spcal.cell_concentration(
                        result["lod_mass"],
                        diameter=self.cell_kws["celldiameter"],
                        molarmass=self.cell_kws["molarmass"],
                    )
                    result["inputs"].update(self.cell_kws)

            except ValueError:
                self.processFailed.emit(infile.name)
                continue

            try:
                export_nanoparticle_results(outfile, result)
            except ValueError:
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
        parent: QtWidgets.QWidget = None,
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

        self.trim_left = QtWidgets.QCheckBox("Use sample left trim.")
        self.trim_left.setChecked(True)
        self.trim_right = QtWidgets.QCheckBox("Use sample right trim.")
        self.trim_right.setChecked(True)

        self.progress = QtWidgets.QProgressBar()
        self.thread: QtCore.QThread = None

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
        self.inputs.layout().addRow(self.trim_left)
        self.inputs.layout().addRow(self.trim_right)

        layout_list = QtWidgets.QVBoxLayout()
        layout_list.addWidget(self.button_files, 0, QtCore.Qt.AlignLeft)
        layout_list.addWidget(self.files, 1)

        layout_horz = QtWidgets.QHBoxLayout()
        layout_horz.addLayout(layout_list)
        layout_horz.addWidget(self.inputs, 0)

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

        limit_method = self.options.method.currentText()
        sigma = float(self.options.sigma.text())
        alpha = float(self.options.error_rate_alpha.text())
        beta = float(self.options.error_rate_alpha.text())
        manual = float(self.options.manual.text() or 0.0)
        if self.sample.table_units.currentText() == "CPS":
            cps_dwelltime = self.options.dwelltime.baseValue()
        else:
            cps_dwelltime = None

        window = (
            int(self.options.window_size.text())
            if self.options.window_size.isEnabled()
            else None
        )

        tleft, tright = None, None
        if self.trim_left.isChecked():
            tleft = self.sample.slider.left()
        if self.trim_right.isChecked():
            tright = self.sample.slider.right()

        method = self.options.efficiency_method.currentText()

        if method in ["Manual Input", "Reference Particle"]:
            if method == "Manual Input":
                efficiency = float(self.options.efficiency.text())
            elif method == "Reference Particle":
                efficiency = float(self.reference.efficiency.text())
            else:
                raise ValueError("Unknown method")

            method = results_from_nebulisation_efficiency
            method_kws = {
                "density": self.sample.density.baseValue(),
                "dwelltime": self.options.dwelltime.baseValue(),
                "efficiency": efficiency,
                "massfraction": float(self.sample.massfraction.text()),
                "time": self.sample.timeAsSeconds(),
                "uptake": self.options.uptake.baseValue(),
                "response": self.options.response.baseValue(),
            }

        elif method == "Mass Response":

            method = results_from_mass_response
            method_kws = {
                "density": self.sample.density.baseValue(),
                "dwelltime": self.options.dwelltime.baseValue(),
                "massfraction": float(self.sample.massfraction.text()),
                "massresponse": self.reference.massresponse.baseValue(),
            }
        else:
            raise ValueError("Unknown method")

        cell_kws = {
            "celldiameter": self.options.celldiameter.baseValue(),
            "molarmass": self.sample.molarmass.baseValue(),
        }

        self.thread = ProcessThread(
            infiles,
            outfiles,
            method,
            method_kws,
            cell_kws=cell_kws,
            trim=(tleft, tright),
            limit_method=limit_method,
            limit_sigma=sigma,
            limit_error_rates=(alpha, beta),
            limit_manual=manual,
            limit_window=window,
            cps_dwelltime=cps_dwelltime,
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
