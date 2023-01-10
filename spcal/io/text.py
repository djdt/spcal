import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Set, TextIO, Tuple

import numpy as np

from spcal import __version__
from spcal.result import SPCalResult


def import_single_particle_file(
    path: Path | str,
    delimiter: str = ",",
    columns: Tuple[int] | None = None,
    first_line: int = 1,
    new_names: Tuple[str] | None = None,
    convert_cps: float | None = None,
) -> Tuple[np.ndarray, List[str]]:
    """Imports data stored as text with elements in columns.

    Args:
        path: path to file
        delimiter: delimiting character between columns
        columns: which columns to import, deafults to all
        first_line: the first data (not header) line
        new_names: rename columns
        convert_cps: the dwelltime (in s) if data is stored as counts per second,
            else None

    Returns:
        data, structred array
        old_names, the original names used in text file
    """
    path = Path(path)
    with path.open("rb") as fp:
        if delimiter != ",":
            gen = (x.replace(b",", b".") for x in fp)
        else:
            gen = (x for x in fp)

        data = np.genfromtxt(
            gen,
            delimiter=delimiter,
            usecols=columns,
            names=True,
            skip_header=first_line - 1,
            invalid_raise=False,
        )
    assert data.dtype.names is not None

    names = list(data.dtype.names)
    if new_names is not None:
        data.dtype.names = new_names

    if convert_cps is not None:
        for name in data.dtype.names:
            data[name] = data[name] * convert_cps  # type: ignore

    return data, names


def export_single_particle_results(
    path: Path | str,
    results: Dict[str, SPCalResult],
    units_for_inputs: Dict[str, Tuple[str, float]] | None = None,
    units_for_results: Dict[str, Tuple[str, float]] | None = None,
    composition_kws: Dict[str, Any] | None = None,
    output_inputs: bool = True,
    output_results: bool = True,
    output_compositions: bool = False,
    output_arrays: bool = True,
) -> None:
    """Export results for elements to a file.

    Args:
        path: path to output csv
        results: dict of SPCalResult for each element
        units_for_inputs: units for option/sample inputs, defaults to sane
        units_for_results: units for output of detections and lods
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
    result_units = {
        "signal": ("counts", 1.0),
        "mass": ("kg", 1.0),
        "size": ("m", 1.0),
        "cell_concentration": ("mol/L", 1.0),
    }

    compostion_distance = 0.03
    if composition_kws is not None and "distance" in composition_kws:
        compostion_distance = composition_kws["distance"]

    if units_for_inputs is not None:
        input_units.update(units_for_inputs)
    if units_for_results is not None:
        result_units.update(units_for_results)

    def write_if_exists(
        fp: TextIO,
        results: Dict[str, SPCalResult],
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

    def write_header(fp: TextIO, first_result: SPCalResult) -> None:
        date = datetime.datetime.strftime(datetime.datetime.now(), "%c")
        fp.write(f"# SPCal Export {__version__}\n")
        fp.write(f"# Date,{date}\n")
        fp.write(f"# File,{first_result.file}\n")
        fp.write(f"# Acquisition events,{first_result.events}\n")
        fp.write("#\n")

    def write_inputs(fp: TextIO, results: Dict[str, SPCalResult]) -> None:
        # Todo: split into insutrment, sample, reference inputs?
        fp.write(f"# Options and inputs,{','.join(results.keys())}\n")
        # fp.write(f"# Dwelltime,{first_result.inputs['dwelltime']},s")
        # fp.write(f"# Uptake,{first_result.inputs['dwelltime']},s")

        input_set: Set[str] = set()  # All inputs across all elements
        for result in results.values():
            input_set.update(result.inputs.keys())

        for input in sorted(list(input_set)):
            unit, factor = input_units.get(input, ("", 1.0))
            write_if_exists(
                fp,
                results,
                lambda r: (r.inputs.get(input, 0.0) / factor) or None,
                f"# {input.replace('_', ' ').capitalize()},",
                postfix="," + unit,
            )
        fp.write("#\n")

        def limit_name_and_params(r: SPCalResult):
            params = ";".join(f"{k}={v:.4g}" for k, v in r.limits.params.items())
            if r.limits.window_size != 0:
                if len(params) > 0:
                    params += ";"
                params += f"window={r.limits.window_size}"
            if len(params) == 0:
                return r.limits.name
            return r.limits.name + " (" + params + ")"

        write_if_exists(
            fp, results, limit_name_and_params, "# Limit method,", format="{}"
        )

        fp.write("#\n")

    def write_detection_results(fp: TextIO, results: Dict[str, SPCalResult]) -> None:
        fp.write(f"# Detection results,{','.join(results.keys())}\n")

        write_if_exists(fp, results, lambda r: r.number, "# Particle number,")
        write_if_exists(fp, results, lambda r: r.number_error, "# Number error,")
        write_if_exists(
            fp,
            results,
            lambda r: r.number_concentration,
            "# Number concentration,",
            postfix=",#/L",
        )
        unit, factor = result_units["mass"]
        write_if_exists(
            fp,
            results,
            lambda r: ((r.mass_concentration or 0.0) / factor) or None,
            "# Mass concentration,",
            postfix=f",{unit}/L",
        )
        fp.write("#\n")

        # === Background ===
        write_if_exists(
            fp, results, lambda r: r.background, "# Background,", postfix=",counts"
        )
        # write_if_exists(
        #     fp, results, lambda r: r.asMass(r.background), "#,", postfix=",kg"
        # )
        unit, factor = result_units["size"]
        write_if_exists(
            fp,
            results,
            lambda r: ((r.asSize(r.background) or 0.0) / factor) or None,
            "#,",
            postfix="," + unit,
        )
        write_if_exists(
            fp,
            results,
            lambda r: r.background_error,
            "# Background error,",
            postfix=",counts",
        )
        unit, factor = result_units["mass"]
        write_if_exists(
            fp,
            results,
            lambda r: ((r.ionic_background or 0.0) / factor) or None,
            "# Ionic background,",
            postfix=f",{unit}/L",
        )
        fp.write("#\n")

        def ufunc_or_none(
            r: SPCalResult, ufunc, key: str, factor: float = 1.0
        ) -> float | None:
            if key not in r.detections:
                return None
            return ufunc(r.detections[key][r.indicies]) / factor

        fp.write(f"# Mean,{','.join(results.keys())}\n")
        for key in ["signal", "mass", "size", "cell_concentration"]:
            unit, factor = result_units[key]
            write_if_exists(
                fp,
                results,
                lambda r: (ufunc_or_none(r, np.mean, key, factor)),
                "#,",
                postfix="," + unit,
            )
        fp.write(f"# Median,{','.join(results.keys())}\n")
        for key in ["signal", "mass", "size", "cell_concentration"]:
            unit, factor = result_units[key]
            write_if_exists(
                fp,
                results,
                lambda r: (ufunc_or_none(r, np.median, key, factor)),
                "#,",
                postfix="," + unit,
            )

    def write_compositions(fp: TextIO, results: Dict[str, SPCalResult]) -> None:
        from spcal.cluster import agglomerative_cluster, prepare_data_for_clustering

        keys = ",".join(f"{key},error" for key in results.keys())
        fp.write(f"# Peak composition,count,{keys}\n")
        # For filtered?
        # valid = np.zeros(self.results[names[0]].detections["signal"].size, dtype=bool)
        # for result in self.results.values():
        #     valid[result.indicies] = True

        # num_valid = np.count_nonzero(valid)
        # if num_valid == 0:
        #     return

        for key in ["signal", "mass", "size", "cell_concentration"]:
            data = {
                name: r.detections[key] if key in r.detections else np.array([0])
                for name, r in results.items()
            }
            if len(data) == 0:
                continue
            fractions = prepare_data_for_clustering(data)

            if fractions.shape[0] == 1:  # single peak
                means, stds, counts = fractions, np.zeros_like(fractions), np.array([1])
            elif fractions.shape[1] == 1:  # single element
                continue
            else:
                means, stds, counts = agglomerative_cluster(
                    fractions, compostion_distance
                )

            compositions = np.empty(
                counts.size, dtype=[(name, np.float64) for name in data]
            )
            for i, name in enumerate(data):
                compositions[name] = means[:, i]

            for i in range(counts.size):
                fp.write(
                    f"# {key.replace('_', ' ').capitalize()},{counts[i]},"
                    + ",".join(
                        "{:.4g},{:.4g}".format(m, s)
                        for m, s in zip(means[i, :], stds[i, :])
                    )
                    + "\n"
                )

    def write_limits(fp: TextIO, results: Dict[str, SPCalResult]) -> None:
        fp.write(f"# Limits of detection,{','.join(results.keys())}\n")

        def limit_or_range(
            r: SPCalResult, key: str, factor: float = 1.0, format: str = "{:.8g}"
        ) -> str | None:
            lod = r.limits.limit_of_detection
            if isinstance(lod, np.ndarray):
                lod = np.array([lod.min(), lod.max()])

            if key == "mass":
                lod = r.asMass(lod)  # type: ignore
            elif key == "size":
                lod = r.asSize(lod)  # type: ignore
            elif key == "cell_concentration":
                lod = r.asCellConcentration(lod)  # type: ignore

            if lod is None:
                return None
            elif isinstance(lod, np.ndarray):
                return (
                    format.format(lod[0] / factor)
                    + " - "
                    + format.format(lod[1] / factor)
                )
            return format.format(lod / factor)

        for key in ["signal", "mass", "size", "cell_concentration"]:
            unit, factor = result_units[key]
            write_if_exists(
                fp,
                results,
                lambda r: limit_or_range(r, key, factor),
                "#,",
                postfix="," + unit,
                format="{}",
            )
        fp.write("#\n")

    def write_arrays(fp: TextIO, results: Dict[str, SPCalResult]) -> None:
        fp.write("# Raw detection data\n")
        # Output data
        data = []
        header_name = ""
        header_unit = ""

        for name, result in results.items():
            for key in ["signal", "mass", "size", "cell_concentration"]:
                if key in result.detections:
                    unit, factor = result_units[key]
                    header_name += f",{name}"
                    header_unit += f",{unit}"
                    data.append(result.detections[key] / factor)

        data = np.stack(data, axis=1)

        fp.write(header_name[1:] + "\n")
        fp.write(header_unit[1:] + "\n")
        for line in data:
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
            write_compositions(fp, results)
        if output_arrays:
            write_arrays(fp, results)
        fp.write("# End of export")
