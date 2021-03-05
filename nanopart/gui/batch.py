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

    export_nanoparticle_results(outfile, result)

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

    export_nanoparticle_results(outfile, result)

    return True


class BatchProcessDialog(QtWidgets.QFileDialog):
    fileProccessed = QtCore.Signal()
    proccessingStarted = QtCore.Signal()
    proccessingFinshed = QtCore.Signal()

    def __init__(
        self,
        sample: SampleWidget,
        reference: ReferenceWidget,
        options: OptionsWidget,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(
            parent, "Batch Process Files", "", "CSV Documents (.csv);All files (.*)"
        )
        self.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        self.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        self.filesSelected.connect(self.batchProcess)

        self.sample = sample
        self.reference = reference
        self.options = options

        self.progress = QtWidgets.QProgressDialog("Processing...", "", 0, 0, self)
        self.progress.setWindowTitle("Batch Processing")

    def dialogLoadFiles(self) -> QtWidgets.QFileDialog:
        pass

    def batchProcess(self, files: List[str]) -> None:
        files = [Path(file) for file in files]
        self.progress.setMaximum(len(files))
        self.progress.open()

        method = self.options.efficiency_method.currentText()

        limit_method = self.options.method.currentText()
        sigma = float(self.options.sigma.text())
        epsilon = float(self.options.epsilon.text())
        response_in_cps = self.sample.table_units.currentText() == "CPS"

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
                futures = [
                    executor.submit(
                        process_file_nebulisation_efficiency,
                        file,
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
                    for file in files
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
                        density=density,
                        dwelltime=dwelltime,
                        molarratio=molarratio,
                        massresponse=massresponse,
                        limit_method=limit_method,
                        limit_sigma=sigma,
                        limit_epsilon=epsilon,
                        response_in_cps=response_in_cps,
                    )
                    for file in files
                ]

        completed = 0
        for future in as_completed(futures):
            self.fileProccessed.emit(completed)
            completed += 1

        self.fileProccessed.emit(len(futures))
