"""Reading and writing single particle data from and to csv files."""

import datetime
import importlib.metadata
import logging
from pathlib import Path
from typing import TextIO

import numpy as np

from spcal.calc import mode as modefn

from spcal.cluster import cluster_information, prepare_results_for_clustering
from spcal.processing import SPCalProcessingMethod, SPCalProcessingResult
from spcal.datafile import SPCalDataFile

logger = logging.getLogger(__name__)


# def append_results_summary(
#     path: TextIO | Path,
#     results: dict[str, "SPCalResult"],
#     units_for_results: dict[str, tuple[str, float]] | None = None,
# ):
#     """Export and append flattened results for elements to a file.
#
#     Args:
#         path: path to output csv
#         results: dict of element: SPCalResult
#         units_for_results: units for output of detections and lods
#         output_results: write basic results, e.g. means, median
#     """
#
# result_units = {k: v for k, v in "SPCalResult".base_units.items()}
#
# if units_for_results is not None:
#     result_units.update(units_for_results)
#
#     if isinstance(path, Path):
#         path = path.open("a")
#
#     def write_header(fp: TextIO, results: dict[str, SPCalResult]):
#         fp.write(",,," + ",".join(results.keys()) + "\n")
#
#     def write_detection_results(fp: TextIO, results: dict[str, SPCalResult]):
#         if len(results) == 0:
#             return
#         file = next(iter(results.values())).file.stem
#
#         _write_if_exists(fp, results, lambda r: r.number, f"{file},Number,#,")
#         _write_if_exists(
#             fp,
#             results,
#             lambda r: r.number_error,
#             f"{file},Number error,#,",
#         )
#         _write_if_exists(
#             fp,
#             results,
#             lambda r: r.number_concentration,
#             f"{file},Number concentration,#/L,",
#         )
#         unit, factor = result_units["mass"]
#         _write_if_exists(
#             fp,
#             results,
#             lambda r: ((r.mass_concentration or 0.0) / factor) or None,
#             f"{file},Mass concentration,{unit}/L,",
#         )
#
#         # === Background ===
#         _write_if_exists(
#             fp,
#             results,
#             lambda r: r.background,
#             f"{file},Background,counts,",
#         )
#         unit, factor = result_units["size"]
#         _write_if_exists(
#             fp,
#             results,
#             lambda r: (
#                 r.asSize(r.background) / factor if r.canCalibrate("size") else None
#             ),
#             f"{file},Background,{unit},",
#         )
#         _write_if_exists(
#             fp,
#             results,
#             lambda r: r.background_error,
#             f"{file},Background error,counts,",
#         )
#         unit, factor = result_units["mass"]
#         _write_if_exists(
#             fp,
#             results,
#             lambda r: ((r.ionic_background or 0.0) / factor) or None,
#             f"{file},Ionic background,{unit}/L,",
#         )
#
#         for label, ufunc in zip(["Mean", "Median", "Mode"], [np.mean, np.median, mode]):
#             for key in SPCalResult.base_units.keys():
#                 unit, factor = result_units[key]
#                 _write_if_exists(
#                     fp,
#                     results,
#                     lambda r: (_ufunc_or_none(r, ufunc, key, factor)),
#                     f"{file},{label},{unit},",
#                 )

# def write_limits(fp: TextIO, results: dict[str, "SPCalResult"]):
# def limit_or_range(
#     r: SPCalResult, key: str, factor: float = 1.0, format: str = "{:.8g}"
# ) -> str | None:
#     lod = r.limits.detection_limit
#     if isinstance(lod, np.ndarray):
#         lod = np.array([lod.min(), lod.max()])
#
#     if not r.canCalibrate(key):
#         return None
#
#     lod = r.convertTo(lod, key)  # type: ignore
#     if isinstance(lod, np.ndarray):
#         return (
#             format.format(lod[0] / factor)
#             + " - "
#             + format.format(lod[1] / factor)
#         )
#     return format.format(lod / factor)
#
# if len(results) == 0:
#     return
# file = next(iter(results.values())).file.stem
#
# for key in SPCalResult.base_units.keys():
#     unit, factor = result_units[key]
#     _write_if_exists(
#         fp,
#         results,
#         lambda r: limit_or_range(r, key, factor),
#         f"{file},Limit of detection,{unit},",
#         format="{}",
#     )
#
# write_header(path, results)
# write_detection_results(path, results)
# write_limits(path, results)


def _value(v: float | None) -> str:
    return "" if v is None else str(v)


def _scaled(v: float | None, factor: float) -> str:
    return "" if v is None else str(v / factor)


def export_spcal_datafile_header(fp: TextIO, data_file: "SPCalDataFile"):
    date = datetime.datetime.strftime(datetime.datetime.now(), "%c")
    fp.write(f"# SPCal Export {importlib.metadata.version('spcal')}\n")
    fp.write(f"# Date,{date}\n")
    fp.write(f"# File,{data_file.path}\n")
    fp.write(f"# Acquisition events,{data_file.num_events}\n")
    fp.write(f"# Acquisition time,{data_file.total_time}\n")
    fp.write(f"# Event time,{data_file.event_time}\n")
    # Maybe export info if we bother reading it later
    fp.write("#\n")


def export_spcal_result_options(fp: TextIO, results: list["SPCalProcessingResult"]):
    fp.write("# Instrument Options,Uptake (L/h),Efficiency\n")
    options = next(iter(results)).method.instrument_options
    fp.write(f",{_value(options.uptake)},{_value(options.efficiency)}\n")

    fp.write("# Limit Options,Type,Parameters\n")
    for result in results:
        pstring = ";".join(f"{k}={v}" for k, v in result.limit.parameters)
        fp.write(f"# {result.isotope},{result.limit.name},{pstring}\n")

    fp.write(
        "# Isotope Options,Density (kg/m3),Response (L/kg),Mass Fraction,Concentraion (kg/L),Mass Response (kg)\n"
    )
    for result in results:
        options = result.method.isotope_options[result.isotope]
        fp.write(
            f"# {result.isotope},{_value(options.density)},{_value(options.response)},{_value(options.mass_fraction)},"
        )
        fp.write(f"{_value(options.concentration)},{_value(options.mass_response)}\n")


def export_spcal_result_outputs(
    fp: TextIO,
    results: list["SPCalProcessingResult"],
    units: dict[str, tuple[str, float]] | None = None,
):
    fp.write(
        "# Detection Outputs,Number,Number Error,Number Concentration (#/L),Mass Concentration (kg/L)\n"
    )
    for result in results:
        fp.write(
            f"{result.number},{result.number_error},{_value(result.number_concentration)},{_value(result.mass_concentration)}\n"
        )

    if units is None:
        units = {
            "signal": ("cts", 1.0),
            "mass": ("kg", 1.0),
            "size": ("m", 1.0),
            "volume": ("L", 1.0),
        }

    for key in SPCalProcessingMethod.CALIBRATION_KEYS:
        if not any(result.canCalibrate(key) for result in results):
            continue

        unit, factor = units[key]
        fp.write(
            f"# ({key}),Background ({unit}),Background Error ({unit}),LOD ({unit}),Mean ({unit}),Std ({unit}),Median ({unit}),Mode ({unit})\n"
        )
        for result in results:
            if not result.canCalibrate(key):
                continue
            detections = result.calibrated(key)

            values = [
                result.background,
                result.background_error,
                result.limit.detection_threshold,
                result.limit.mean_signal,
                np.mean(detections),
                np.std(detections),
                np.median(detections),
                modefn(detections),
            ]
            fp.write(
                f"# {result.isotope},"
                + ",".join(_scaled(value, factor) for value in values)
                + "\n"
            )


def export_spcal_compositions(
    fp: TextIO,
    results: list["SPCalProcessingResult"],
    clusters: np.ndarray,
    units: dict[str, tuple[str, float]] | None = None,
):
    if units is None:
        units = {
            "signal": ("cts", 1.0),
            "mass": ("kg", 1.0),
            "size": ("m", 1.0),
            "volume": ("L", 1.0),
        }

    for key in SPCalProcessingMethod.CALIBRATION_KEYS:
        if not any(result.canCalibrate(key) for result in results):
            continue

        X = prepare_results_for_clustering(results, key)
        valid = np.any(X != 0, axis=1)
        X = X[valid]
        means, stds, counts = cluster_information(X, clusters[valid])

        unit, factor = units[key]
        fp.write(f"# Detection Compositions,Count,Mean ({unit}),Std ({unit})")
        for i, result in enumerate(results):
            fp.write(
                f"# {result.isotope},{counts[i]},{_scaled(means[i], factor)},{_scaled(stds[i], factor)}\n"
            )


def export_spcal_detection_arrays(
    fp: TextIO,
    results: list["SPCalProcessingResult"],
    clusters: np.ndarray,
    units: dict[str, tuple[str, float]] | None = None,
):
    assert all(result.peak_indicies is not None for result in results)
    npeaks = np.amax([result.peak_indicies[-1] for result in results]) + 1  # type: ignore , checked above


def export_single_particle_results(
    path: Path | str,
    results: dict[str, "SPCalResult"],
    clusters: dict[str, np.ndarray],
    detection_times: np.ndarray | None = None,
    units_for_inputs: dict[str, tuple[str, float]] | None = None,
    units_for_results: dict[str, tuple[str, float]] | None = None,
    output_inputs: bool = True,
    output_results: bool = True,
    output_compositions: bool = False,
    output_arrays: bool = True,
):
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
    pass
    #
    # input_units = {
    #     "density": ("g/cm3", 1e3),
    #     "dwelltime": ("ms", 1e-3),
    #     "molar_mass": ("g/mol", 1e-3),
    #     "response": ("counts/(μg/L)", 1e9),
    #     "time": ("s", 1.0),
    #     "uptake": ("ml/min", 1e-3 / 60.0),
    # }
    #
    # result_units = {k: v for k, v in SPCalResult.base_units.items()}
    #
    #
    # def write_arrays(
    #     fp: TextIO,
    #     results: dict[str, SPCalResult],
    #     clusters: dict[str, np.ndarray],
    #     detection_times: np.ndarray | None = None,
    #     export_clusters: bool = False,
    # ):
    #     fp.write("# Raw detection data\n")
    #
    #     # Non-filtered indicies
    #     valid = SPCalResult.all_valid_indicies(list(results.values()))
    #
    #     # Output data
    #     data = [] if detection_times is None else [detection_times[valid]]
    #     header_name = "" if detection_times is None else ",Time"
    #     header_unit = "" if detection_times is None else ",s"
    #
    #     for name, result in results.items():
    #         for key in SPCalResult.base_units.keys():
    #             if result.canCalibrate(key):
    #                 unit, factor = result_units[key]
    #                 header_name += f",{name}"
    #                 header_unit += f",{unit}"
    #                 data.append(
    #                     result.calibrated(key, use_indicies=False)[valid] / factor
    #                 )
    #
    #     data = np.stack(data, axis=1)
    #
    #     if export_clusters:
    #         idx = np.zeros(valid.size)
    #         for key in SPCalResult.base_units.keys():
    #             if key in clusters:
    #                 header_name += ",cluster idx"
    #                 header_unit += f",{key}"
    #
    #         indicies = []
    #         for cluster in clusters.values():
    #             idx = np.zeros(valid.size, dtype=int)
    #             idx[valid] = cluster + 1
    #             indicies.append(idx)
    #
    #         indicies = np.stack(indicies, axis=1)
    #         data = np.concatenate((data, indicies), axis=1)
    #
    #     fp.write(header_name[1:] + "\n")
    #     fp.write(header_unit[1:] + "\n")
    #     for line in data:
    #         if np.all(
    #             line == 0.0
    #         ):  # pragma: no cover, don't write line if all filtered
    #             continue
    #         fp.write(
    #             ",".join("" if x == 0.0 else "{:.8g}".format(x) for x in line) + "\n"
    #         )
    #     fp.write("#\n")
    #
    # path = Path(path)
    #
    # with path.open("w", encoding="utf-8") as fp:
    #     write_header(fp, next(iter(results.values())))
    #     if output_inputs:
    #         write_inputs(fp, results)
    #     if output_results:
    #         write_detection_results(fp, results)
    #         write_limits(fp, results)
    #     if output_compositions:
    #         write_compositions(fp, results, clusters)
    #     if output_arrays:
    #         write_arrays(
    #             fp,
    #             results,
    #             clusters,
    #             detection_times,
    #             export_clusters=output_compositions,
    #         )
    #     fp.write("# End of export")
