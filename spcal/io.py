import numpy as np
from pathlib import Path
import logging

from typing import Any, Dict, TextIO, List, Tuple, Union

from spcal import __version__

logger = logging.getLogger(__name__)


def read_nanoparticle_file(
    path: Union[Path, str],
) -> Tuple[np.ndarray, Dict]:
    """Imports data and parameters from a NP export.

    Data is expected to be a text or csv in a single column of responses, or twos columns:
    1) aquisition times, 2) responses. If two columns are found then the parameter 'dwelltime'
    will be set as the mean difference of the first coulmn. If 'cps' is read in the file header
    then the parameter 'cps' will be set to True.

    Tested with Agilent and thermo exports.

    Args:
        path: path to the file
        delimiter: text delimiter, default to comma

    Returns:
        signal
        dict of any parameters
    """

    def delimited_translated_columns(path: Path, columns: int = 3):
        """Translates inputs with ';' to have ',' as delimiter and '.' as decimal.
        Ensures at least `columns` columns in data by prepending ','."""
        map = str.maketrans({";": ",", ",": "."})
        with path.open("r") as fp:
            for line in fp:
                if ";" in line:
                    line = line.translate(map)
                count = line.count(",")
                if count < columns:
                    yield "," * (columns - count - 1) + line
                else:
                    yield line

    def read_header_params(path: Path, size: int = 1024) -> Dict:
        with path.open("r") as fp:
            header = fp.read(size)

        parameters = {"cps": "cps" in header.lower()}
        return parameters

    if isinstance(path, str):
        path = Path(path)

    data = np.genfromtxt(
        delimited_translated_columns(path, 3),
        delimiter=",",
        usecols=(0, 1, 2),
        dtype=np.float64,
    )
    parameters = read_header_params(path)

    response = data[:, 2]
    # Remove any invalid rows, e.g. headers
    valid_response = ~np.isnan(response)
    response = response[valid_response]

    if not np.all(np.isnan(data[:, 1])):
        #        3 columns of data, thermo export of [Number, Time, Signal]
        if not np.all(np.isnan(data[:, 0])):
            pass
        else:  # 2 columns of data, agilent export of [Time, Signal]
            pass
        times = data[:, 1][valid_response]
        parameters["dwelltime"] = np.round(np.mean(np.diff(times)), 6)

    logger.info(f"Imported {response.size} points from {path.name}.")
    return response, parameters


def export_nanoparticle_results(path: Path, results: dict) -> None:
    """Writes data from a results dict.

    Structure is Dict[<name>, Dict[<key>, <value>]]
    Valid keys are:
        'file': original file path
        'events': the number of aquisition events
        'inputs': dictionary of inputs in SI units
        'detections': array of NP detections
        'detections_std': stddev of detection count
        'limit_method': method used to calculate LOD and epsilon/sigma, (str, float)
        'limit_window': window size used for thresholding
        'background': mean background (counts)
        'background_size': background equivilent diameter (m)
        'background_concentration': iconic background (kg/L)
        'background_std': stddev of background (counts)
        'lod': value or array of {min, max, mean, median} limits of detection (counts)
        'lod_mass': lod in (kg)
        'lod_size': lod in (m)
        'lod_cell_concentration': lod in (mol/L)
        'number_concentration': NP concentration (#/L)
        'concentration': NP concentration (kg/L)
        'masses': NP mass array (kg)
        'sizes': NP size array (m)
        'cell_concentrations': intracellular concentrations (mol/L)
    """

    def get_key_or_default(results: dict, name: str, key: str, default: Any) -> Any:
        return (
            results[name][key] if name in results and key in results[name] else default
        )

    def write_if_key_exists(
        fp: TextIO, results: dict, key: str, prefix: str, postfix: str = ""
    ) -> None:
        if any(key in results[name] for name in results.keys()):
            line = ",".join(
                str(get_key_or_default(results, name, key, ""))
                for name in results.keys()
            )
            fp.write(f"{prefix}{line}{postfix}\n")

    names = list(results.keys())
    with path.open("w", encoding="utf-8") as fp:
        fp.write(f"# SPCal Export {__version__}\n")
        fp.write(f"# File,'{results[names[0]]['file']}'\n")
        fp.write(f"# Acquisition events,{results[names[0]]['events']}\n")

        # fp.write(f"# Options and inputs\n")
        # for k, v in result["inputs"].items():
        #     fp.write(f'#,{k.replace("_", " ").capitalize()},{v}\n')

        # fp.write(f"# Limit method,{str(result['limit_method']).replace(',', ';')}\n")
        # if result["limit_window"] is not None and result["limit_window"] > 1:
        #     fp.write(f"# Limit window,{result['limit_window']}\n")

        detections = [str(results[name]["detections"].size) for name in names]
        stds = [str(results[name]["detections_std"]) for name in names]
        backgrounds = [str(results[name]["background"]) for name in names]
        background_stds = [str(results[name]["background_std"]) for name in names]

        fp.write(f",{','.join(names)}\n")
        fp.write(f"# Detected particles,{','.join(detections)}\n")
        fp.write(f"# Detection stddev,{','.join(stds)}\n")

        # Background
        fp.write(f"# Background,{','.join(backgrounds)},counts\n")
        write_if_key_exists(fp, results, "background_size", prefix="#,", postfix=",m")
        write_if_key_exists(
            fp,
            results,
            "background_concentration",
            prefix="# Ionic background,",
            postfix=",kg/L",
        )
        fp.write(f"# Background stddev,{','.join(background_stds)},counts\n")

        # LODs
        # if isinstance(result["lod"], np.ndarray):
        #     fp.write("# Limit of detection,Min,Max,Mean,Median\n")
        #     fp.write(f"#,{','.join(str(s) for s in (result['lod']))},counts\n")
        # else:
        #     fp.write(f"# Limit of detection,{result['lod']},counts\n")

        # for key, unit in [
        #     ("lod_mass", "kg"),
        #     ("lod_size", "m"),
        #     ("lod_cell_concentration", "mol/L"),
        # ]:
        #     if key in result:
        #         if isinstance(result[key], np.ndarray):
        #             fp.write(f"#,{','.join(str(s) for s in (result[key]))},{unit}\n")
        #         else:
        #             fp.write(f"#,{result[key]},{unit}\n")

        # Concentrations
        write_if_key_exists(
            fp,
            results,
            "number_concentration",
            prefix="# Number concentration,",
            postfix=",#/L",
        )
        write_if_key_exists(
            fp, results, "concentration", prefix="# Concentration,", postfix=",kg/L"
        )

        means = [np.mean(results[name]["detections"]) for name in names]
        mass_means = np.array(
            [
                np.mean(get_key_or_default(results, name, "masses", 0.0))
                for name in names
            ]
        )
        size_means = np.array(
            [np.mean(get_key_or_default(results, name, "sizes", 0.0)) for name in names]
        )
        conc_means = np.array(
            [
                np.mean(get_key_or_default(results, name, "cell_concentrations", 0.0))
                for name in names
            ]
        )

        # Mean values
        fp.write(f"# Mean,{','.join(str(x) for x in means)},counts\n")
        if np.any(mass_means > 0.0):
            fp.write(f"#,{','.join(str(x) for x in mass_means)},kg\n")
        if np.any(size_means > 0.0):
            fp.write(f"#,{','.join(str(x) for x in size_means)},m\n")
        if np.any(conc_means > 0.0):
            fp.write(f"#,{','.join(str(x) for x in conc_means)},mol/L\n")

        medians = [np.median(results[name]["detections"]) for name in names]
        mass_medians = np.array(
            [
                np.median(get_key_or_default(results, name, "masses", 0.0))
                for name in names
            ]
        )
        size_medians = np.array(
            [
                np.median(get_key_or_default(results, name, "sizes", 0.0))
                for name in names
            ]
        )
        conc_medians = np.array(
            [
                np.median(get_key_or_default(results, name, "cell_concentrations", 0.0))
                for name in names
            ]
        )
        fp.write(f"# Median,{','.join(str(x) for x in medians)},counts\n")
        if np.any(mass_medians > 0.0):
            fp.write(f"#,{','.join(str(x) for x in mass_medians)},kg\n")
        if np.any(size_medians > 0.0):
            fp.write(f"#,{','.join(str(x) for x in size_medians)},m\n")
        if np.any(conc_medians > 0.0):
            fp.write(f"#,{','.join(str(x) for x in conc_medians)},mol/L\n")

        # Output data
        # header = "Signal (counts)"
        # data = [result["detections"]]
        # for key, label in [
        #     ("masses", "Mass (kg)"),
        #     ("sizes", "Size (m)"),
        #     ("cell_concentrations", "Conc. (mol/L)"),
        # ]:
        #     if key in result:
        #         header += "," + label
        #         data.append(result[key])
        # fp.write(header + "\n")
        # np.savetxt(
        #     fp,
        #     np.stack(data, axis=1),
        #     delimiter=",",
        # )
        logger.info(f"Exported results for to {path.name}.")
