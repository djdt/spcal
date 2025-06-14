"""Reading and writing single particle data from and to csv files."""

# import csv
import datetime
import importlib.metadata
import logging
from pathlib import Path
from typing import Any, Callable, Set, TextIO

import numpy as np

from spcal.calc import mode
from spcal.result import SPCalResult

logger = logging.getLogger(__name__)


def is_text_file(path: Path) -> bool:
    """Checks path exists and is a '.csv', '.txt' or '.text'."""
    if path.suffix.lower() not in [".csv", ".txt", ".text"]:
        return False
    if path.is_dir() or not path.exists():
        return False
    return True


def guess_text_parameters(lines: list[str]) -> tuple[str, int, int]:
    """Guesses the delimiter, skip_rows and column count.

    Args:
        lines: list of lines in file or header

    Returns:
        delimiter, skip_rows, column_count
    """
    skip_rows = 0

    delimiter = ""
    for line in lines:
        try:
            delimiter = next(d for d in ["\t", ";", ",", " "] if d in line)
            tokens = line.split(delimiter)
            for token in tokens:
                float(token)
            break
        except StopIteration:  # special case where only one column exists
            try:
                float(line)
                break
            except ValueError:
                pass
        except ValueError:
            pass
        skip_rows += 1

    if delimiter == "":
        column_count = 1
    else:
        column_count = max([line.count(delimiter) for line in lines[skip_rows:]]) + 1

    return delimiter, skip_rows, column_count


def _write_if_exists(
    fp: TextIO,
    results: dict[str, SPCalResult],
    fn: Callable[[SPCalResult], Any],
    prefix: str = "",
    postfix: str = "",
    delimiter: str = ",",
    format: str = "{:.8g}",
) -> None:
    values = [fn(result) for result in results.values()]
    if all(x is None for x in values):
        return
    text = delimiter.join(format.format(v) if v is not None else "" for v in values)
    fp.write(prefix + text + postfix + "\n")


def _ufunc_or_none(
    r: SPCalResult, ufunc, key: str, factor: float = 1.0
) -> float | None:
    if not r.canCalibrate(key) or r.indicies.size == 0:
        return None
    return ufunc(r.calibrated(key)) / factor


def read_single_particle_file(
    path: Path | str,
    delimiter: str = ",",
    columns: tuple[int, ...] | np.ndarray | None = None,
    first_line: int = 1,
    convert_cps: float | None = None,
    max_rows: int | None = None,
) -> np.ndarray:
    """Imports data stored as text with elements in columns.

    Args:
        path: path to file
        delimiter: delimiting character between columns
        columns: which columns to import, deafults to all
        first_line: the first data (not header) line
        convert_cps: the dwelltime (in s) if data is stored as counts per second,
        else None

    Returns:
        data, structred array
    """

    def csv_read_lines(fp, delimiter: str = ",", count: int = 0):
        for line in fp:
            if delimiter != ",":
                yield line.replace(",", ".")
            else:
                yield line

    path = Path(path)
    with path.open("r") as fp:
        gen = csv_read_lines(fp, delimiter)
        for i in range(first_line - 1):
            next(gen)

        header = next(gen).strip().split(delimiter)
        if columns is None:
            columns = np.arange(len(header))
        columns = np.asarray(columns)

        dtype = np.dtype([(header[i], np.float32) for i in columns])
        try:  # loadtxt is faster than genfromtxt
            data = np.loadtxt(  # type: ignore
                gen,
                delimiter=delimiter,
                usecols=columns,
                dtype=dtype,
                max_rows=max_rows,
            )
        except ValueError:  # Rewind and try with genfromtxt
            logger.warning(
                "read_single_particle_file: np.loadtxt failed, trying np.genfromtxt"
            )
            fp.seek(0)
            for i in range(first_line):
                next(gen)

            data = np.genfromtxt(  # type: ignore
                gen,
                delimiter=delimiter,
                usecols=columns,
                dtype=dtype,
                max_rows=max_rows,
                converters={c: lambda s: np.float32(s or 0) for c in columns},
                invalid_raise=False,
                loose=True,
            )

    if data.dtype.names is None:
        data.dtype = dtype
    data.dtype.names = tuple(name.replace(" ", "_") for name in data.dtype.names)

    if convert_cps is not None:
        for name in data.dtype.names:
            data[name] = data[name] * convert_cps  # type: ignore

    return data


def append_results_summary(
    path: TextIO | Path,
    results: dict[str, SPCalResult],
    units_for_results: dict[str, tuple[str, float]] | None = None,
) -> None:
    """Export and append flattened results for elements to a file.

    Args:
        path: path to output csv
        results: dict of element: SPCalResult
        units_for_results: units for output of detections and lods
        output_results: write basic results, e.g. means, median
    """

    result_units = {k: v for k, v in SPCalResult.base_units.items()}

    if units_for_results is not None:
        result_units.update(units_for_results)

    if isinstance(path, Path):
        path = path.open("a")

    def write_header(fp: TextIO, results: dict[str, SPCalResult]) -> None:
        fp.write(",,," + ",".join(results.keys()) + "\n")

    def write_detection_results(fp: TextIO, results: dict[str, SPCalResult]) -> None:
        if len(results) == 0:
            return
        file = next(iter(results.values())).file.stem

        _write_if_exists(fp, results, lambda r: r.number, f"{file},Number,#,")
        _write_if_exists(
            fp,
            results,
            lambda r: r.number_error,
            f"{file},Number error,#,",
        )
        _write_if_exists(
            fp,
            results,
            lambda r: r.number_concentration,
            f"{file},Number concentration,#/L,",
        )
        unit, factor = result_units["mass"]
        _write_if_exists(
            fp,
            results,
            lambda r: ((r.mass_concentration or 0.0) / factor) or None,
            f"{file},Mass concentration,{unit}/L,",
        )

        # === Background ===
        _write_if_exists(
            fp,
            results,
            lambda r: r.background,
            f"{file},Background,counts,",
        )
        unit, factor = result_units["size"]
        _write_if_exists(
            fp,
            results,
            lambda r: (
                r.asSize(r.background) / factor if r.canCalibrate("size") else None
            ),
            f"{file},Background,{unit},",
        )
        _write_if_exists(
            fp,
            results,
            lambda r: r.background_error,
            f"{file},Background error,counts,",
        )
        unit, factor = result_units["mass"]
        _write_if_exists(
            fp,
            results,
            lambda r: ((r.ionic_background or 0.0) / factor) or None,
            f"{file},Ionic background,{unit}/L,",
        )

        for label, ufunc in zip(["Mean", "Median", "Mode"], [np.mean, np.median, mode]):
            for key in SPCalResult.base_units.keys():
                unit, factor = result_units[key]
                _write_if_exists(
                    fp,
                    results,
                    lambda r: (_ufunc_or_none(r, ufunc, key, factor)),
                    f"{file},{label},{unit},",
                )

    def write_limits(fp: TextIO, results: dict[str, SPCalResult]) -> None:
        def limit_or_range(
            r: SPCalResult, key: str, factor: float = 1.0, format: str = "{:.8g}"
        ) -> str | None:
            lod = r.limits.detection_limit
            if isinstance(lod, np.ndarray):
                lod = np.array([lod.min(), lod.max()])

            if not r.canCalibrate(key):
                return None

            lod = r.convertTo(lod, key)  # type: ignore
            if isinstance(lod, np.ndarray):
                return (
                    format.format(lod[0] / factor)
                    + " - "
                    + format.format(lod[1] / factor)
                )
            return format.format(lod / factor)

        if len(results) == 0:
            return
        file = next(iter(results.values())).file.stem

        for key in SPCalResult.base_units.keys():
            unit, factor = result_units[key]
            _write_if_exists(
                fp,
                results,
                lambda r: limit_or_range(r, key, factor),
                f"{file},Limit of detection,{unit},",
                format="{}",
            )

    write_header(path, results)
    write_detection_results(path, results)
    write_limits(path, results)


def export_single_particle_results(
    path: Path | str,
    results: dict[str, SPCalResult],
    clusters: dict[str, np.ndarray],
    detection_times: np.ndarray | None = None,
    units_for_inputs: dict[str, tuple[str, float]] | None = None,
    units_for_results: dict[str, tuple[str, float]] | None = None,
    output_inputs: bool = True,
    output_results: bool = True,
    output_compositions: bool = False,
    output_arrays: bool = True,
) -> None:
    """Export results for elements to a file.

    Args:
        path: path to output csv
        results: dict of element: SPCalResult
        clusters: dict of element: cluster indices
        detection_times: array of times for each detection
        units_for_inputs: units for option/sample inputs, defaults to sane
        units_for_results: units for output of detections and lods
        output_inputs: write input instrument and sample parameters
        output_results: write basic results, e.g. means, median
        output_compositions: write cluster means and indices (if output_array)
        output_arrays: write detection data
    """

    input_units = {
        "cell_diameter": ("μm", 1e-6),
        "density": ("g/cm3", 1e3),
        "dwelltime": ("ms", 1e-3),
        "molar_mass": ("g/mol", 1e-3),
        "response": ("counts/(μg/L)", 1e9),
        "time": ("s", 1.0),
        "uptake": ("ml/min", 1e-3 / 60.0),
    }

    result_units = {k: v for k, v in SPCalResult.base_units.items()}

    if units_for_inputs is not None:
        input_units.update(units_for_inputs)
    if units_for_results is not None:
        result_units.update(units_for_results)

    def write_header(fp: TextIO, first_result: SPCalResult) -> None:
        date = datetime.datetime.strftime(datetime.datetime.now(), "%c")
        fp.write(f"# SPCal Export {importlib.metadata.version('spcal')}\n")
        fp.write(f"# Date,{date}\n")
        fp.write(f"# File,{first_result.file}\n")
        fp.write(f"# Acquisition events,{first_result.events}\n")
        fp.write("#\n")

    def write_inputs(fp: TextIO, results: dict[str, SPCalResult]) -> None:
        # Todo: split into insutrment, sample, reference inputs?
        fp.write(f"# Options and inputs,{','.join(results.keys())}\n")
        # fp.write(f"# Dwelltime,{first_result.inputs['dwelltime']},s")
        # fp.write(f"# Uptake,{first_result.inputs['dwelltime']},s")

        input_set: Set[str] = set()  # All inputs across all elements
        for result in results.values():
            input_set.update(result.inputs.keys())

        for input in sorted(list(input_set)):
            unit, factor = input_units.get(input, ("", 1.0))
            _write_if_exists(
                fp,
                results,
                lambda r: (r.inputs.get(input, 0.0) / factor) or None,
                f"# {input.replace('_', ' ').capitalize()},",
                postfix="," + unit,
            )
        fp.write("#\n")

        _write_if_exists(
            fp, results, lambda r: str(r.limits), "# Limit method,", format="{}"
        )

        fp.write("#\n")

    def write_detection_results(fp: TextIO, results: dict[str, SPCalResult]) -> None:
        if len(results) == 0:
            return

        fp.write(f"# Detection results,{','.join(results.keys())}\n")

        _write_if_exists(fp, results, lambda r: r.number, "# Particle number,")
        _write_if_exists(fp, results, lambda r: r.number_error, "# Number error,")
        _write_if_exists(
            fp,
            results,
            lambda r: r.number_concentration,
            "# Number concentration,",
            postfix=",#/L",
        )
        unit, factor = result_units["mass"]
        _write_if_exists(
            fp,
            results,
            lambda r: ((r.mass_concentration or 0.0) / factor) or None,
            "# Mass concentration,",
            postfix=f",{unit}/L",
        )
        fp.write("#\n")

        # === Background ===
        _write_if_exists(
            fp, results, lambda r: r.background, "# Background,", postfix=",counts"
        )
        # _write_if_exists(
        #     fp, results, lambda r: r.asMass(r.background), "#,", postfix=",kg"
        # )
        unit, factor = result_units["size"]
        _write_if_exists(
            fp,
            results,
            lambda r: (
                r.asSize(r.background) / factor if r.canCalibrate("size") else None
            ),
            "#,",
            postfix="," + unit,
        )
        _write_if_exists(
            fp,
            results,
            lambda r: r.background_error,
            "# Background error,",
            postfix=",counts",
        )
        unit, factor = result_units["mass"]
        _write_if_exists(
            fp,
            results,
            lambda r: ((r.ionic_background or 0.0) / factor) or None,
            "# Ionic background,",
            postfix=f",{unit}/L",
        )
        fp.write("#\n")

        for label, ufunc in zip(["Mean", "Median", "Mode"], [np.mean, np.median, mode]):
            fp.write(f"# {label},{','.join(results.keys())}\n")
            for key in SPCalResult.base_units.keys():
                unit, factor = result_units[key]
                _write_if_exists(
                    fp,
                    results,
                    lambda r: (_ufunc_or_none(r, ufunc, key, factor)),
                    "#,",
                    postfix="," + unit,
                )

    def write_compositions(
        fp: TextIO, results: dict[str, SPCalResult], clusters: dict[str, np.ndarray]
    ) -> None:
        from spcal.cluster import cluster_information, prepare_data_for_clustering

        if len(results) == 0:
            return

        keys = ",".join(f"{key},error" for key in results.keys())
        fp.write(f"# Peak composition,count,{keys}\n")
        # TODO filter on demand
        # For filtered?
        # valid = np.zeros(self.results[names[0]].detections["signal"].size, dtype=bool)

        valid = SPCalResult.all_valid_indicies(list(results.values()))

        for key in SPCalResult.base_units.keys():
            data = {}
            for name, result in results.items():
                if result.canCalibrate(key):
                    data[name] = result.calibrated(key, use_indicies=False)[valid]
            if len(data) == 0 or key not in clusters:
                continue

            X = prepare_data_for_clustering(data)
            T = clusters[key]

            if X.shape[0] == 1:  # pragma: no cover, single peak
                means, stds, counts = X, np.zeros_like(X), np.array([1])
            elif X.shape[1] == 1:  # single element
                continue
            else:
                means, stds, counts = cluster_information(X, T)

            # compositions = np.empty(
            #     counts.size, dtype=[(name, np.float64) for name in data]
            # )
            # for i, name in enumerate(data):
            #     compositions[name] = means[:, i]

            fp.write(f"# {key.replace('_', ' ').capitalize()}")
            for i in range(counts.size):
                fp.write(
                    f",{counts[i]},"
                    + ",".join(
                        "{:.4g},{:.4g}".format(m, s)
                        for m, s in zip(means[i, :], stds[i, :])
                    )
                    + "\n"
                )

    def write_limits(fp: TextIO, results: dict[str, SPCalResult]) -> None:
        fp.write(f"# Limits of detection,{','.join(results.keys())}\n")

        def limit_or_range(
            r: SPCalResult, key: str, factor: float = 1.0, format: str = "{:.8g}"
        ) -> str | None:
            lod = r.limits.detection_limit
            if isinstance(lod, np.ndarray):
                lod = np.array([lod.min(), lod.max()])

            if not r.canCalibrate(key):
                return None

            lod = r.convertTo(lod, key)  # type: ignore
            if isinstance(lod, np.ndarray):
                return (
                    format.format(lod[0] / factor)
                    + " - "
                    + format.format(lod[1] / factor)
                )
            return format.format(lod / factor)

        for key in SPCalResult.base_units.keys():
            unit, factor = result_units[key]
            _write_if_exists(
                fp,
                results,
                lambda r: limit_or_range(r, key, factor),
                "#,",
                postfix="," + unit,
                format="{}",
            )
        fp.write("#\n")

    def write_arrays(
        fp: TextIO,
        results: dict[str, SPCalResult],
        clusters: dict[str, np.ndarray],
        detection_times: np.ndarray | None = None,
        export_clusters: bool = False,
    ) -> None:
        fp.write("# Raw detection data\n")

        # Non-filtered indicies
        valid = SPCalResult.all_valid_indicies(list(results.values()))

        # Output data
        data = [] if detection_times is None else [detection_times[valid]]
        header_name = "" if detection_times is None else ",Time"
        header_unit = "" if detection_times is None else ",s"

        for name, result in results.items():
            for key in SPCalResult.base_units.keys():
                if result.canCalibrate(key):
                    unit, factor = result_units[key]
                    header_name += f",{name}"
                    header_unit += f",{unit}"
                    data.append(
                        result.calibrated(key, use_indicies=False)[valid] / factor
                    )

        data = np.stack(data, axis=1)

        if export_clusters:
            idx = np.zeros(valid.size)
            for key in SPCalResult.base_units.keys():
                if key in clusters:
                    header_name += ",cluster idx"
                    header_unit += f",{key}"

            indicies = []
            for cluster in clusters.values():
                idx = np.zeros(valid.size, dtype=int)
                idx[valid] = cluster + 1
                indicies.append(idx)

            indicies = np.stack(indicies, axis=1)
            data = np.concatenate((data, indicies), axis=1)

        fp.write(header_name[1:] + "\n")
        fp.write(header_unit[1:] + "\n")
        for line in data:
            if np.all(
                line == 0.0
            ):  # pragma: no cover, don't write line if all filtered
                continue
            fp.write(
                ",".join("" if x == 0.0 else "{:.8g}".format(x) for x in line) + "\n"
            )
        fp.write("#\n")

    path = Path(path)

    with path.open("w", encoding="utf-8") as fp:
        write_header(fp, next(iter(results.values())))
        if output_inputs:
            write_inputs(fp, results)
        if output_results:
            write_detection_results(fp, results)
            write_limits(fp, results)
        if output_compositions:
            write_compositions(fp, results, clusters)
        if output_arrays:
            write_arrays(
                fp,
                results,
                clusters,
                detection_times,
                export_clusters=output_compositions,
            )
        fp.write("# End of export")
