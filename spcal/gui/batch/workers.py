import json
from pathlib import Path
from typing import TextIO
from PySide6 import QtCore

from spcal.datafile import SPCalDataFile, SPCalNuDataFile, SPCalTextDataFile
from spcal.isotope import SPCalIsotope
from spcal.processing.method import SPCalProcessingMethod

from spcal.io.export import append_results_summary, export_spcal_processing_results


# Lots of repeated code here but inheriting a base class causes blocking when running in a QThread
def process_data_file_and_export(
    data_file: SPCalDataFile,
    method: SPCalProcessingMethod,
    isotopes: list[SPCalIsotope],
    output_path: Path,
    export_options: dict,
    summary_fp: TextIO | None = None,
):
    results = method.processDataFile(data_file, isotopes)
    results = method.filterResults(results)
    if export_options.get("clusters", False):
        clusters = {
            key: method.processClusters(results, key)
            for key in SPCalProcessingMethod.CALIBRATION_KEYS
        }
    else:
        clusters = {}
    export_spcal_processing_results(
        output_path,
        data_file,
        list(results.values()),
        clusters,
        units=export_options["units"],
        export_options=export_options["options"],
        export_results=export_options["results"],
        export_arrays=export_options["arrays"],
        export_compositions=export_options["clusters"],
    )
    if summary_fp is not None:
        append_results_summary(
            summary_fp, data_file, list(results.values()), export_options["units"]
        )


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
        autoblank: bool = True,
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
        self.autoblank = autoblank

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
            autoblank=self.autoblank,
        )

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
            process_data_file_and_export(
                data_file,
                self.method,
                self.isotopes,
                output.with_stem(output.stem + f"_{j + 1:03}"),
                self.export_options,
            )

    @QtCore.Slot()
    def process(self):
        self.started.emit(len(self.paths))
        if self.export_options["summary"] is not None:
            summary_fp = Path(self.export_options["summary"]).open("w")
        else:
            summary_fp = None

        for i, (input, output) in enumerate(self.paths):
            self.progress.emit(i, 0.0)
            if self.chunk_size == 0:
                data_file = self.openDataFile(input)
                if self.thread().isInterruptionRequested():
                    return
                self.progress.emit(i, 0.5)
                process_data_file_and_export(
                    data_file,
                    self.method,
                    self.isotopes,
                    output,
                    self.export_options,
                    summary_fp=summary_fp,
                )
                if self.thread().isInterruptionRequested():
                    return
            else:
                self.processChunk(i, input, output)
            self.progress.emit(i, 1.0)
        if summary_fp is not None:
            summary_fp.close()
        self.finished.emit()


class TextBatchWorker(QtCore.QObject):
    started = QtCore.Signal(int)
    progress = QtCore.Signal(int, float)
    finished = QtCore.Signal()

    def __init__(
        self,
        paths: list[tuple[Path, Path]],
        method: SPCalProcessingMethod,
        isotopes: list[SPCalIsotope],
        export_options: dict,
        delimiter: str = ",",
        skip_rows: int = 1,
        cps: bool = False,
        drop_fields: list[str] | None = None,
        override_event_time: float | None = None,
        instrument_type: str | None = None,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent)

        self.paths = paths
        self.isotopes = isotopes
        self.method = method
        self.export_options = export_options

        self.delimiter = delimiter
        self.skip_rows = skip_rows
        self.cps = cps
        self.drop_fields = drop_fields
        self.override_event_time = override_event_time
        self.instrument_type = instrument_type

    def openDataFile(self, path: Path) -> SPCalDataFile:
        return SPCalTextDataFile.load(
            path,
            delimiter=self.delimiter,
            skip_rows=self.skip_rows,
            cps=self.cps,
            drop_fields=self.drop_fields,
            override_event_time=self.override_event_time,
            instrument_type=self.instrument_type,
        )

    @QtCore.Slot()
    def process(self):
        self.started.emit(len(self.paths))
        for i, (input, output) in enumerate(self.paths):
            self.progress.emit(i, 0.0)
            data_file = self.openDataFile(input)
            if self.thread().isInterruptionRequested():
                return
            self.progress.emit(i, 0.5)
            process_data_file_and_export(
                data_file, self.method, self.isotopes, output, self.export_options
            )
            if self.thread().isInterruptionRequested():
                return
            self.progress.emit(i, 1.0)
        self.finished.emit()
