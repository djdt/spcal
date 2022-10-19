import numpy as np
from pathlib import Path
import logging

from typing import Any, Dict, TextIO, Tuple, Union

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
        'lod': value or array of limits of detection (counts)
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
        if any(key in results[name] for name in results):
            line = ",".join(
                str(get_key_or_default(results, name, key, ""))
                for name in results.keys()
            )
            fp.write(f"{prefix}{line}{postfix}\n")

    input_units = {
        "density": "kg/m3",
        "dwelltime": "s",
        "molar_mass": "kg/mol",
        "reponse": "counts/(kg/L)",
        "time": "s",
        "uptake": "L/s",
    }

    names = list(results.keys())
    with path.open("w", encoding="utf-8") as fp:
        fp.write(f"# SPCal Export {__version__}\n")
        fp.write(f"# File,'{results[names[0]]['file']}'\n")
        fp.write(f"# Acquisition events,{results[names[0]]['events']}\n")

        # === Options and inputs ===
        fp.write(f"#\n# Options and inputs\n")
        fp.write(f"#,{','.join(names)}\n")
        inputs = set()
        for name in names:
            inputs.update(results[name]["inputs"].keys())
        inputs = sorted(list(inputs))
        for input in inputs:
            values = [str(results[name]["inputs"].get(input, "")) for name in names]
            fp.write(
                f"# {str(input).replace('_', ' ').capitalize()},{','.join(values)},{input_units.get(input, '')}\n"
            )

        # === Limit method and params ===
        fp.write(
            f"# Limit method,{','.join(results[name]['limit_method'].replace(',', ';') for name in names)}\n"
        )
        write_if_key_exists(fp, results, "limit_window", "# Limit window,")

        # === Detection counts ===
        fp.write("#\n# Detection results\n")
        fp.write(f"#,{','.join(names)}\n")

        detections = [results[name]["detections"].size for name in names]
        fp.write(f"# Detected particles,{','.join(str(x) for x in detections)}\n")
        write_if_key_exists(fp, results, "detections_std", prefix="# Detection stddev,")
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

        # === Background ===
        write_if_key_exists(
            fp, results, "background", prefix="# Background,", postfix=",counts"
        )
        write_if_key_exists(fp, results, "background_size", prefix="#,", postfix=",m")
        write_if_key_exists(
            fp,
            results,
            "background_std",
            prefix="# Background stddev,",
            postfix=",counts",
        )
        write_if_key_exists(
            fp,
            results,
            "background_concentration",
            prefix="# Ionic background,",
            postfix=",kg/L",
        )

        # === LODs ===
        def limit_or_range(x: np.ndarray) -> str:
            if np.all(x == 0.0):
                return ""
            return f"{x.min()} - {x.max()}" if x.size > 1 else str(x[0])

        lods = [results[name]["lod"] for name in names]
        lods_mass = [
            get_key_or_default(results, name, "lod_mass", np.array([0.0]))
            for name in names
        ]
        lods_size = [
            get_key_or_default(results, name, "lod_size", np.array([0.0]))
            for name in names
        ]
        lods_conc = [
            get_key_or_default(results, name, "lod_cell_concentration", np.array([0.0]))
            for name in names
        ]

        fp.write(
            f"# Limits of detection,{','.join(limit_or_range(x) for x in lods)},counts\n"
        )
        if any(np.any(x > 0.0) for x in lods_mass):
            fp.write(f"#,{','.join(limit_or_range(x) for x in lods_mass)},kg\n")
        if any(np.any(x > 0.0) for x in lods_size):
            fp.write(f"#,{','.join(limit_or_range(x) for x in lods_size)},m\n")
        if any(np.any(x > 0.0) for x in lods_conc):
            fp.write(f"#,{','.join(limit_or_range(x) for x in lods_conc)},mol/L\n")

        # === Mean values ===
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

        fp.write(f"# Mean,{','.join(str(x or '') for x in means)},counts\n")
        if np.any(mass_means > 0.0):
            fp.write(f"#,{','.join(str(x or '') for x in mass_means)},kg\n")
        if np.any(size_means > 0.0):
            fp.write(f"#,{','.join(str(x or '') for x in size_means)},m\n")
        if np.any(conc_means > 0.0):
            fp.write(f"#,{','.join(str(x or '') for x in conc_means)},mol/L\n")

        # === Median values ===
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
            fp.write(f"#,{','.join(str(x or '') for x in mass_medians)},kg\n")
        if np.any(size_medians > 0.0):
            fp.write(f"#,{','.join(str(x or '') for x in size_medians)},m\n")
        if np.any(conc_medians > 0.0):
            fp.write(f"#,{','.join(str(x or '') for x in conc_medians)},mol/L\n")

        fp.write("#\n# Raw detection data\n")
        # Output data
        max_len = np.amax(detections)
        data = []
        header_name = ""
        header_unit = ""

        for name in names:
            header_name += f",{name}"
            header_unit += ",counts"
            x = results[name]["detections"]
            data.append(np.pad(x, (0, max_len - x.size), constant_values=np.nan))
            for key, unit in [
                ("masses", "kg"),
                ("sizes", "m"),
                ("cell_concentrations", "mol/L"),
            ]:
                if key in results[name]:
                    header_name += f",{name}"
                    header_unit += f",{unit}"
                    x = results[name][key]
                    data.append(
                        np.pad(x, (0, max_len - x.size), constant_values=np.nan)
                    )
        data = np.stack(data, axis=1)

        fp.write(header_name[1:] + "\n")
        fp.write(header_unit[1:] + "\n")
        for line in data:
            fp.write(",".join("" if np.isnan(x) else str(x) for x in line) + "\n")

        logger.info(f"Exported results for to {path.name}.")
