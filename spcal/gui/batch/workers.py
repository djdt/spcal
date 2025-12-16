import json
from time import sleep
from pathlib import Path
from PySide6 import QtCore

from spcal.datafile import SPCalDataFile, SPCalNuDataFile
from spcal.isotope import SPCalIsotope
from spcal.processing.method import SPCalProcessingMethod

from spcal.io.export import export_spcal_processing_results


# Lots of repeated code here but inheriting a base class causes blocking when running in a QThread


class BatchWorker(QtCore.QObject):
    started = QtCore.Signal(int)
    progress = QtCore.Signal(int, float)
    finished = QtCore.Signal()

    def __init__(
        self,
        paths: list[tuple[Path, Path]],
        method: SPCalProcessingMethod,
        isotopes: list[SPCalIsotope],
        skip_clusters: bool = False,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent)

        self.paths = paths
        self.isotopes = isotopes
        self.method = method
        self.skip_clusters = skip_clusters

    # def openDataFile(self, path: Path) -> SPCalDataFile:
    #     raise NotImplementedError
    #
    # def processDataFile(self, data_file: SPCalDataFile, output_path: Path):
    #     results = self.method.processDataFile(data_file, self.isotopes)
    #     results = self.method.filterResults(results)
    #     if self.skip_clusters:
    #         clusters = {}
    #     else:
    #         clusters = {
    #             key: self.method.processClusters(results, key)
    #             for key in SPCalProcessingMethod.CALIBRATION_KEYS
    #         }
    #     export_spcal_processing_results(
    #         output_path, data_file, list(results.values()), clusters
    #     )
    #
    # @QtCore.Slot()
    # def process(self):
    #     self.started.emit(len(self.paths))
    #     for i, (input, output) in enumerate(self.paths):
    #         if self.thread().isInterruptionRequested():
    #             return
    #         self.progress.emit(i, 0.0)
    #         data_file = self.openDataFile(input)
    #         self.progress.emit(i, 0.5)
    #         if self.thread().isInterruptionRequested():
    #             return
    #         self.processDataFile(data_file, output)
    #         sleep(1)
    #         self.progress.emit(i, 1.0)
    #     self.finished.emit()


class NuBatchWorker(QtCore.QObject):
    started = QtCore.Signal(int)
    progress = QtCore.Signal(int, float)
    finished = QtCore.Signal()

    def __init__(
        self,
        paths: list[tuple[Path, Path]],
        method: SPCalProcessingMethod,
        isotopes: list[SPCalIsotope],
        export_options: dict,
        chunk_size: int = 0,
        max_mass_diff: float = 0.1,
        cyc_number: int | None = None,
        seg_number: int | None = None,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent)

        self.paths = paths
        self.isotopes = isotopes
        self.method = method
        self.export_options = export_options

        self.chunk_size = chunk_size
        self.max_mass_diff = max_mass_diff
        self.cyc_number = cyc_number
        self.seg_number = seg_number

    def openDataFile(
        self, path: Path, first: int = 0, last: int | None = None
    ) -> SPCalDataFile:
        return SPCalNuDataFile.load(
            path,
            max_mass_diff=self.max_mass_diff,
            cycle_number=self.cyc_number,
            segment_number=self.seg_number,
            first_integ_file=first,
            last_integ_file=last,
        )

    def processDataFile(self, data_file: SPCalDataFile, output_path: Path):
        results = self.method.processDataFile(data_file, self.isotopes)
        results = self.method.filterResults(results)
        if self.export_options.get("clusters", False):
            clusters = {
                key: self.method.processClusters(results, key)
                for key in SPCalProcessingMethod.CALIBRATION_KEYS
            }
        else:
            clusters = {}
        export_spcal_processing_results(
            output_path,
            data_file,
            list(results.values()),
            clusters,
            units=self.export_options["units"],
            export_options=self.export_options["options"],
            export_results=self.export_options["results"],
            export_arrays=self.export_options["arrays"],
            export_compositions=self.export_options["clusters"],
        )
        if self.export_options["summary"] is not None:
            raise NotImplementedError

    def processChunk(self, i: int, input: Path, output: Path):
        with input.joinpath("integrated.index").open("r") as fp:
            nintegs = len(json.load(fp))
        for j, first in enumerate(range(0, nintegs, self.chunk_size)):
            last = min(nintegs, first + self.chunk_size)
            if self.thread().isInterruptionRequested():
                return
            data_file = self.openDataFile(input, first=first, last=last)
            self.progress.emit(i, last / nintegs)
            if self.thread().isInterruptionRequested():
                return
            self.processDataFile(
                data_file,
                output.with_stem(output.stem + f"_{j + 1:03}"),
            )

    @QtCore.Slot()
    def process(self):
        self.started.emit(len(self.paths))
        for i, (input, output) in enumerate(self.paths):
            self.progress.emit(i, 0.0)
            if self.chunk_size == 0:
                data_file = self.openDataFile(input)
                if self.thread().isInterruptionRequested():
                    return
                self.progress.emit(i, 0.5)
                self.processDataFile(data_file, output)
                if self.thread().isInterruptionRequested():
                    return
            else:
                self.processChunk(i, input, output)
            self.progress.emit(i, 1.0)
        self.finished.emit()


class BatchTextWorker(BatchWorker):
    pass


class BatchTOFWERKWorker(BatchWorker):
    pass
