from PySide2 import QtCore, QtWidgets

from concurrent.futures import as_completed, ProcessPoolExecutor
import numpy as np
from pathlib import Path

import nanopart

from nanopart.calc import (
    calculate_limits,
    results_from_mass_response,
    results_from_nebulisation_efficiency,
)
from nanopart.io import read_nanoparticle_file, export_nanoparticle_results

from nanopart.gui.inputs import SampleWidget, ReferenceWidget
from nanopart.gui.options import OptionsWidget

from typing import List


def process_file_detections(
    file: Path,
    limit_method: str,
    limit_sigma: float,
    limit_epsilon: float,
    cps_dwelltime: float = None,
) -> dict:
    responses, _ = read_nanoparticle_file(file, delimiter=",")

    # Convert to counts if required
    if cps_dwelltime is not None:
        responses = responses * cps_dwelltime

    size = responses.size

    if responses is None or size == 0:
        raise ValueError(f"Unabled to import file '{file.name}'.")

    limits = calculate_limits(responses, limit_method, limit_sigma, limit_epsilon)

    if limits is None:
        raise ValueError("Limit calculations failed for '{file.name}'.")

    detections, labels = nanopart.accumulate_detections(responses, limits[2], limits[3])
    background = np.nanmean(responses[labels == 0])

    return {
        "file": str(file),
        "events": size,
        "detections": detections,
        "number": detections.size,
        "background": background,
        "limit_method": limits[0],
        "lod": limits[3],
    }


def process_file_mass_response(
    file: Path,
    outfile: Path,
    density: float,
    dwelltime: float,
    molarratio: float,
    massresponse: float,
    limit_method: str = "Automatic",
    limit_sigma: float = 3.0,
    limit_epsilon: float = 0.5,
    response_in_cps: bool = False,
) -> bool:
    result = process_file_detections(
        file,
        limit_method,
        limit_sigma,
        limit_epsilon,
        cps_dwelltime=dwelltime if response_in_cps else None,
    )

    result.update(
        results_from_mass_response(
            result["detections"],
            result["background"],
            result["lod"],
            density=density,
            molarratio=molarratio,
            massresponse=massresponse,
        )
    )

    # export_nanoparticle_results(outfile, result)

    return True


def process_file_nebulisation_efficiency(
    file: Path,
    outfile: Path,
    density: float,
    dwelltime: float,
    efficiency: float,
    molarratio: float,
    uptake: float,
    response: float,
    time: float,
    limit_method: str = "Automatic",
    limit_sigma: float = 3.0,
    limit_epsilon: float = 0.5,
    response_in_cps: bool = False,
) -> bool:

    result = process_file_detections(
        file,
        limit_method,
        limit_sigma,
        limit_epsilon,
        cps_dwelltime=dwelltime if response_in_cps else None,
    )

    result.update(
        results_from_nebulisation_efficiency(
            result["detections"],
            result["background"],
            result["lod"],
            density=density,
            dwelltime=dwelltime,
            efficiency=efficiency,
            molarratio=molarratio,
            uptake=uptake,
            response=response,
            time=time,
        )
    )

    # export_nanoparticle_results(outfile, result)

    return True


class BatchProcessDialog(QtWidgets.QDialog):
    fileProccessed = QtCore.Signal()
    proccessingStarted = QtCore.Signal()
    proccessingFinshed = QtCore.Signal()

    def __init__(
        self,
        files: List[str],
        sample: SampleWidget,
        reference: ReferenceWidget,
        options: OptionsWidget,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)

        self.sample = sample
        self.reference = reference
        self.options = options

        self.button_files = QtWidgets.QPushButton("Open Files")
        self.button_files.pressed.connect(self.dialogLoadFiles)
        self.button_output = QtWidgets.QPushButton("Open Directory")
        self.button_output.pressed.connect(self.dialogOpenOuputDir)

        self.button_process = QtWidgets.QPushButton("Start Batch Process")
        self.button_process.setEnabled(len(files) > 0)
        self.button_process.pressed.connect(self.startProcess)

        self.progress = QtWidgets.QProgressBar()

        self.files = QtWidgets.QListWidget()
        self.files.addItems(files)
        self.files.setTextElideMode(QtCore.Qt.ElideLeft)
        self.files.model().rowsInserted.connect(self.completeChanged)
        self.files.model().rowsRemoved.connect(self.completeChanged)

        self.inputs = QtWidgets.QGroupBox("Batch Options")
        self.inputs.setLayout(QtWidgets.QFormLayout())

        self.output_dir = QtWidgets.QLineEdit("")
        self.output_dir.setPlaceholderText("{same as input}")
        self.output_dir.setToolTip("Leave blank to use the input directory.")
        self.output_dir.textChanged.connect(self.completeChanged)

        self.output_name = QtWidgets.QLineEdit("%_result.csv")
        self.output_name.setToolTip("Use '%' to represent the input file name.")
        self.output_name.textChanged.connect(self.completeChanged)

        self.inputs.layout().addRow("Output Name:", self.output_name)
        self.inputs.layout().addRow("Output Directory:", self.output_dir)
        self.inputs.layout().addWidget(self.button_output)

        layout_list = QtWidgets.QVBoxLayout()
        layout_list.addWidget(self.button_files, 0, QtCore.Qt.AlignRight)
        layout_list.addWidget(self.files, 1)

        layout_horz = QtWidgets.QHBoxLayout()
        layout_horz.addLayout(layout_list)
        layout_horz.addWidget(self.inputs, 0)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(layout_horz)
        layout.addWidget(self.progress, 0)
        layout.addWidget(self.button_process, 0, QtCore.Qt.AlignRight)

        self.setLayout(layout)

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
        files, filter = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Batch Process Files", "", "CSV Documents(*.csv);;All files(*)"
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

    def startProcess(self) -> None:
        files = [Path(self.files.item(i).text()) for i in range(self.files.count())]
        outfiles = self.outputsForFiles(files)

        self.progress.setMaximum(len(files))

        limit_method = self.options.method.currentText()
        sigma = float(self.options.sigma.text())
        epsilon = float(self.options.epsilon.text())
        response_in_cps = self.sample.table_units.currentText() == "CPS"

        method = self.options.efficiency_method.currentText()

        if method in ["Manual", "Reference"]:
            if method == "Manual":
                efficiency = float(self.options.efficiency.text())
            elif method == "Reference":
                efficiency = float(self.reference.efficiency.text())

            dwelltime = self.options.dwelltime.baseValue()
            density = self.sample.density.baseValue()
            molarratio = float(self.sample.molarratio.text())
            time = self.sample.timeAsSeconds()
            uptake = self.options.uptake.baseValue()
            response = self.options.response.baseValue()

            with ProcessPoolExecutor() as executor:
                print("exec")
                futures = [
                    executor.submit(
                        process_file_nebulisation_efficiency,
                        file,
                        outfile,
                        density=density,
                        dwelltime=dwelltime,
                        efficiency=efficiency,
                        molarratio=molarratio,
                        uptake=uptake,
                        response=response,
                        time=time,
                        limit_method=limit_method,
                        limit_sigma=sigma,
                        limit_epsilon=epsilon,
                        response_in_cps=response_in_cps,
                    )
                    for file, outfile in zip(files, outfiles)
                ]
        elif method == "Mass Response (None)":
            density = self.sample.density.baseValue()
            dwelltime = self.options.dwelltime.baseValue()
            molarratio = float(self.sample.molarratio.text())
            massresponse = self.reference.massresponse.baseValue()

            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        process_file_mass_response,
                        file,
                        outfile,
                        density=density,
                        dwelltime=dwelltime,
                        molarratio=molarratio,
                        massresponse=massresponse,
                        limit_method=limit_method,
                        limit_sigma=sigma,
                        limit_epsilon=epsilon,
                        response_in_cps=response_in_cps,
                    )
                    for file, outfile in zip(files, outfiles)
                ]

        completed = 0
        for future in as_completed(futures):
            completed += 1
            self.progress.setValue(completed)
            print(future.result())
            print(future.exception())
