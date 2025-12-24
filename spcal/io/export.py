"""Reading and writing single particle data from and to csv files."""

import datetime
import importlib.metadata
import logging
from pathlib import Path
from typing import TextIO

import numpy as np

from spcal.calc import mode as modefn

from spcal.cluster import cluster_information, prepare_results_for_clustering
from spcal.detection import combine_regions

from spcal.datafile import SPCalDataFile
from spcal.processing import SPCalProcessingResult
from spcal.processing.method import SPCalProcessingMethod

logger = logging.getLogger(__name__)


def _value(v: float | None) -> str:
    return "" if v is None else str(v)


def _scaled(v: float | None, factor: float) -> str:
    return "" if v is None else str(v / factor)


def append_results_summary(
    fp: TextIO,
    data_file: SPCalDataFile,
    results: list[SPCalProcessingResult],
    units: dict[str, tuple[str, float]],
):
    if fp.tell() == 0:
        fp.write("Data File,Isotope,Name,Unit,Value\n")
    for result in results:
        detections = np.resize
        for key in SPCalProcessingMethod.CALIBRATION_KEYS:
            if not result.canCalibrate(key):
                continue
            unit, factor = units[key]
            detections = result.calibrated(key)
            values = [
                ("Background", result.calibrateTo(result.background, key)),
                ("Background Error", result.calibrateTo(result.background_error, key)),
                (
                    "LOD",
                    np.nanmean(
                        result.calibrateTo(result.limit.detection_threshold, key)
                    ),
                ),
                ("Mean", np.mean(detections)),
                ("Std", np.std(detections)),
                ("Median", np.median(detections)),
                ("Mode", modefn(detections)),
            ]
            for name, value in values:
                fp.write(
                    f"{data_file.path},{result.isotope},{name},{unit},{_scaled(value, factor)}\n"
                )


def export_spcal_datafile_header(fp: TextIO, data_file: "SPCalDataFile"):
    date = datetime.datetime.strftime(datetime.datetime.now(), "%c")
    fp.write(f"# SPCal Export {importlib.metadata.version('spcal')}\n")
    fp.write(f"# Date,{date}\n")
    fp.write(f"# File,{data_file.path}\n")
    fp.write(f"# Acquisition events,{data_file.num_events}\n")
    fp.write(f"# Acquisition time,{data_file.total_time}\n")
    fp.write(f"# Event time,{data_file.event_time}\n")
    # Maybe export info if we bother reading it later


def export_spcal_result_options(fp: TextIO, results: list["SPCalProcessingResult"]):
    units = {
        "conc": ("µg/L", 1e-9),
        "density": ("g/cm³", 1e3),
        "massresponse": ("ag", 1e-21),
        "response": ("L/µg", 1e9),
        "uptake": ("ml/min", 1e-3 / 60.0),
    }

    fp.write(f"# Instrument Options,Uptake ({units['uptake'][0]}),Efficiency\n")
    options = next(iter(results)).method.instrument_options
    fp.write(
        f"# ,{_scaled(options.uptake, units['uptake'][1])},{_value(options.efficiency)}\n"
    )

    fp.write("#\n")
    fp.write("# Limit Options,Type,Parameters\n")
    for result in results:
        pstring = ";".join(f"{k}={v}" for k, v in result.limit.parameters.items())
        fp.write(f"# {result.isotope},{result.limit.name},{pstring}\n")

    fp.write("#\n")
    fp.write(
        f"# Isotope Options,Density ({units['density'][0]}),Response ({units['response'][0]}),Mass Fraction,"
        f"Concentration ({units['conc'][0]}),Mass Response ({units['massresponse'][0]})\n"
    )
    for result in results:
        options = result.method.isotope_options[result.isotope]
        fp.write(
            f"# {result.isotope},{_scaled(options.density, units['density'][1])},"
            f"{_scaled(options.response, units['response'][1])},{_value(options.mass_fraction)},"
            f"{_scaled(options.concentration, units['conc'][1])},{_scaled(options.mass_response, units['massresponse'][1])}\n"
        )


def export_spcal_result_outputs(
    fp: TextIO,
    results: list["SPCalProcessingResult"],
    units: dict[str, tuple[str, float]],
):
    fp.write(
        "# Outputs (number),Number,Number Error,Number Concentration (#/L),Mass Concentration (kg/L)\n"
    )
    for result in results:
        fp.write(
            f"# {result.isotope},{result.number},{result.number_error},"
            f"{_value(result.number_concentration)},{_value(result.mass_concentration)}\n"
        )

    for key in SPCalProcessingMethod.CALIBRATION_KEYS:
        if not any(result.canCalibrate(key) for result in results):
            continue

        unit, factor = units[key]
        fp.write(
            f"# Outputs ({key}),Background ({unit}),Background Error ({unit}),LOD ({unit}),Mean ({unit}),Std ({unit}),Median ({unit}),Mode ({unit})\n"
        )
        for result in results:
            if not result.canCalibrate(key):
                continue
            detections = result.calibrated(key)

            values = [
                result.calibrateTo(result.background, key),
                result.calibrateTo(result.background_error, key),
                np.nanmean(result.calibrateTo(result.limit.detection_threshold, key)),
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
    clusters: dict[str, np.ndarray],
    units: dict[str, tuple[str, float]],
):
    for key in SPCalProcessingMethod.CALIBRATION_KEYS:
        if (
            not any(result.canCalibrate(key) for result in results)
            or key not in clusters
        ):
            continue

        X, valid = prepare_results_for_clustering(results, clusters[key].size, key)
        means, stds, counts = cluster_information(X[valid], clusters[key][valid])

        unit, factor = units[key]
        fp.write(f"# Compositions ({key}),Count")
        for result in results:
            fp.write(f",{result.isotope} mean ({unit}),{result.isotope} std ({unit})")
        fp.write("\n")

        for i in range(len(counts)):
            fp.write(f"# {i},{counts[i]}")
            for j, result in enumerate(results):
                fp.write(
                    f",{_scaled(means[i, j], factor)},{_scaled(stds[i, j], factor)}"
                )
            fp.write("\n")


def export_spcal_detection_arrays(
    fp: TextIO,
    results: list["SPCalProcessingResult"],
    clusters: dict[str, np.ndarray],
    units: dict[str, tuple[str, float]],
):
    assert all(result.peak_indicies is not None for result in results)
    npeaks = np.amax([result.number_peak_indicies for result in results])

    all_regions = combine_regions([result.regions for result in results], 2)

    times = results[0].times[
        all_regions[:, 0] + (all_regions[:, 1] - all_regions[:, 0]) // 2
    ]
    datas = [times]
    header = "Times (s)"

    for result in results:
        for key in SPCalProcessingMethod.CALIBRATION_KEYS:
            if not result.canCalibrate(key):
                continue
            unit, factor = units[key]
            header += f",{result.isotope} ({unit})"
            peak_data = np.zeros(npeaks, np.float32)
            np.add.at(
                peak_data,
                result.peak_indicies[result.filter_indicies],  # type: ignore
                result.calibrated(key),
            )
            datas.append(peak_data / factor)

    if clusters is not None:
        for key in SPCalProcessingMethod.CALIBRATION_KEYS:
            if key in clusters:
                datas.append(clusters[key])
                header += f",Cluster ID ({key})"

    # Filter out any all zero (filtered) results
    datas = np.stack(datas, axis=1)
    datas = datas[np.any(datas[:, 1:] > 0, axis=1)]
    np.savetxt(fp, datas, delimiter=",", header=header, fmt="%.12g")


def export_spcal_processing_results(
    path: Path | str,
    data_file: "SPCalDataFile",
    results: list["SPCalProcessingResult"],
    clusters: dict[str, np.ndarray],
    units: dict[str, tuple[str, float]] | None = None,
    export_options: bool = True,
    export_results: bool = True,
    export_compositions: bool = False,
    export_arrays: bool = True,
):
    """Export results for elements to a file.

    Args:
        path: path to output csv
        data_file: the datafile for results.
        results: list of results to export
        clusters: clusters for each calibration key
        units: units for output of detections and lods
        export_inputs: write input instrument and sample parameters
        export_results: write basic results, e.g. means, median
        export_compositions: write cluster means and indices (if export_array)
        export_arrays: write detection data
    """
    if units is None:
        units = {
            "signal": ("cts", 1.0),
            "mass": ("kg", 1.0),
            "size": ("m", 1.0),
            "volume": ("L", 1.0),
        }

    path = Path(path)
    with path.open("w", encoding="utf-8") as fp:
        export_spcal_datafile_header(fp, data_file)
        fp.write("#\n")
        if export_options:
            export_spcal_result_options(fp, results)
            fp.write("#\n")
        if export_results:
            export_spcal_result_outputs(fp, results, units)
            fp.write("#\n")
        if export_compositions:
            export_spcal_compositions(fp, results, clusters, units)
            fp.write("#\n")
        if export_arrays:
            export_spcal_detection_arrays(
                fp, results, clusters if export_compositions else {}, units
            )
        fp.write("# End of export")
