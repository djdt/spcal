import json
import numpy as np
from pathlib import Path
from typing import Generator, TextIO
from PySide6 import QtCore

from spcal.datafile import (
    SPCalDataFile,
    SPCalNuDataFile,
    SPCalTOFWERKDataFile,
    SPCalTextDataFile,
)
from spcal.isotope import SPCalIsotope
from spcal.processing import CALIBRATION_KEYS
from spcal.processing.method import SPCalProcessingMethod

from spcal.io.export import append_results_summary, export_spcal_processing_results
from spcal.processing.result import SPCalProcessingResult

import logging

logger = logging.getLogger(__name__)


# Lots of repeated code here but inheriting a base class causes blocking when running in a QThread
def _batch_process_data_file(
    data_file: SPCalDataFile,
    method: SPCalProcessingMethod,
    isotopes: list[SPCalIsotope],
    process_clusters: bool = False,
) -> tuple[list[SPCalProcessingResult], dict[str, np.ndarray]]:
    results = method.processDataFile(data_file, isotopes)
    method.filterResults(results)
    if process_clusters:
        clusters = {
            key: method.processClusters(results, key)
            for key in CALIBRATION_KEYS
        }
    else:
        clusters = {}
    return list(results.values()), clusters


def _batch_export_results(
    data_file: SPCalDataFile,
    outpath: Path,
    results: list[SPCalProcessingResult],
    clusters: dict[str, np.ndarray],
    export_options: dict,
    summary_fp: TextIO | None,
):
    export_spcal_processing_results(
        outpath,
        data_file,
        results,
        clusters,
        units=export_options["units"],
        export_options=export_options["options"],
        export_results=export_options["results"],
        export_arrays=export_options["arrays"],
        export_compositions=export_options["clusters"],
    )
    if summary_fp is not None:
        append_results_summary(summary_fp, data_file, results, export_options["units"])


class NuBatchWorker(QtCore.QObject):
    started = QtCore.Signal(int)
    progress = QtCore.Signal(int, float)
    exception = QtCore.Signal(int, object)
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

    def openDataFileInChunks(
        self,
        path: Path,
        output_path: Path,
        chunk_size: int = 0,
    ) -> Generator[tuple[SPCalDataFile, Path, float], None, None]:
        if chunk_size == 0:
            yield (
                SPCalNuDataFile.load(
                    path,
                    max_mass_diff=self.max_mass_diff,
                    cycle_number=self.cyc_number,
                    segment_number=self.seg_number,
                    autoblank=self.autoblank,
                ),
                output_path,
                0.5,
            )
        else:
            with path.joinpath("integrated.index").open("r") as fp:
                nintegs = len(json.load(fp))
            for j, first in enumerate(range(0, nintegs, self.chunk_size)):
                last = min(nintegs, first + self.chunk_size)
                yield (
                    SPCalNuDataFile.load(
                        path,
                        max_mass_diff=self.max_mass_diff,
                        cycle_number=self.cyc_number,
                        segment_number=self.seg_number,
                        autoblank=self.autoblank,
                    ),
                    output_path.with_stem(output_path.stem + f"_{j + 1:03}"),
                    last / nintegs,
                )

    def processFile(
        self, index: int, path: Path, outpath: Path, summary_fp: TextIO | None
    ):
        self.progress.emit(index, 0.0)
        for data_file, new_output, progress in self.openDataFileInChunks(
            path, outpath, self.chunk_size
        ):
            if self.thread().isInterruptionRequested():
                return
            self.progress.emit(index, progress)

            results, clusters = _batch_process_data_file(
                data_file,
                self.method,
                self.isotopes,
                self.export_options["clusters"],
            )
            if self.thread().isInterruptionRequested():
                return

            _batch_export_results(
                data_file,
                new_output,
                results,
                clusters,
                self.export_options,
                summary_fp,
            )

        self.progress.emit(index, 1.0)

    @QtCore.Slot()
    def process(self):
        self.started.emit(len(self.paths))
        if self.export_options["summary"] is not None:
            summary_fp = Path(self.export_options["summary"]).open("w")
        else:
            summary_fp = None

        for i, (input, output) in enumerate(self.paths):
            try:
                self.processFile(i, input, output, summary_fp)
            except Exception as e:
                self.exception.emit(i, e)
                logger.exception(e)
                continue

        if summary_fp is not None:
            summary_fp.close()
        self.finished.emit()


class TOFWERKBatchWorker(QtCore.QObject):
    started = QtCore.Signal(int)
    progress = QtCore.Signal(int, float)
    exception = QtCore.Signal(int, object)
    finished = QtCore.Signal()

    def __init__(
        self,
        paths: list[tuple[Path, Path]],
        method: SPCalProcessingMethod,
        isotopes: list[SPCalIsotope],
        export_options: dict,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent)

        self.paths = paths
        self.isotopes = isotopes
        self.method = method
        self.export_options = export_options

    def openDataFile(self, path: Path) -> SPCalDataFile:
        return SPCalTOFWERKDataFile.load(path)

    def processFile(
        self, index: int, path: Path, outpath: Path, summary_fp: TextIO | None
    ):
        self.progress.emit(index, 0.0)
        data_file = self.openDataFile(path)
        if self.thread().isInterruptionRequested():
            return
        self.progress.emit(index, 0.5)
        results, clusters = _batch_process_data_file(
            data_file, self.method, self.isotopes, self.export_options["clusters"]
        )
        if self.thread().isInterruptionRequested():
            return
        export_spcal_processing_results(
            outpath,
            data_file,
            results,
            clusters,
            units=self.export_options["units"],
            export_options=self.export_options["options"],
            export_results=self.export_options["results"],
            export_arrays=self.export_options["arrays"],
            export_compositions=self.export_options["clusters"],
        )
        if summary_fp is not None:
            append_results_summary(
                summary_fp, data_file, results, self.export_options["units"]
            )
        self.progress.emit(index, 1.0)

    @QtCore.Slot()
    def process(self):
        self.started.emit(len(self.paths))
        if self.export_options["summary"] is not None:
            summary_fp = Path(self.export_options["summary"]).open("w")
        else:
            summary_fp = None

        for i, (input, output) in enumerate(self.paths):
            try:
                self.processFile(i, input, output, summary_fp)
            except Exception as e:
                self.exception.emit(i, e)
                logger.exception(e)
                continue

        if summary_fp is not None:
            summary_fp.close()
        self.finished.emit()


class TextBatchWorker(QtCore.QObject):
    started = QtCore.Signal(int)
    progress = QtCore.Signal(int, float)
    exception = QtCore.Signal(int, object)
    finished = QtCore.Signal()

    def __init__(
        self,
        paths: list[tuple[Path, Path]],
        method: SPCalProcessingMethod,
        isotopes: list[SPCalIsotope],
        export_options: dict,
        isotope_table: dict[SPCalIsotope, str],
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

        self.isotope_table = isotope_table
        self.delimiter = delimiter
        self.skip_rows = skip_rows
        self.cps = cps
        self.drop_fields = drop_fields
        self.override_event_time = override_event_time
        self.instrument_type = instrument_type

    def openDataFile(self, path: Path) -> SPCalDataFile:
        return SPCalTextDataFile.load(
            path,
            self.isotope_table,
            delimiter=self.delimiter,
            skip_rows=self.skip_rows,
            cps=self.cps,
            drop_fields=self.drop_fields,
            override_event_time=self.override_event_time,
            instrument_type=self.instrument_type,
        )

    def processFile(
        self, index: int, path: Path, outpath: Path, summary_fp: TextIO | None
    ):
        self.progress.emit(index, 0.0)
        data_file = self.openDataFile(path)
        if self.thread().isInterruptionRequested():
            return
        self.progress.emit(index, 0.5)
        results, clusters = _batch_process_data_file(
            data_file, self.method, self.isotopes, self.export_options["clusters"]
        )
        if self.thread().isInterruptionRequested():
            return
        export_spcal_processing_results(
            outpath,
            data_file,
            results,
            clusters,
            units=self.export_options["units"],
            export_options=self.export_options["options"],
            export_results=self.export_options["results"],
            export_arrays=self.export_options["arrays"],
            export_compositions=self.export_options["clusters"],
        )
        if summary_fp is not None:
            append_results_summary(
                summary_fp, data_file, results, self.export_options["units"]
            )
        self.progress.emit(index, 1.0)

    @QtCore.Slot()
    def process(self):
        self.started.emit(len(self.paths))
        if self.export_options["summary"] is not None:
            summary_fp = Path(self.export_options["summary"]).open("w")
        else:
            summary_fp = None

        for i, (input, output) in enumerate(self.paths):
            try:
                self.processFile(i, input, output, summary_fp)
            except Exception as e:
                self.exception.emit(i, e)
                logger.exception(e)
                continue

        if summary_fp is not None:
            summary_fp.close()

        self.finished.emit()
