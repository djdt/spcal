import logging
from pathlib import Path
from typing import Any, Dict, List, TextIO, Tuple

import numpy as np

from spcal import __version__

logger = logging.getLogger(__name__)


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
        convert_cps: the dwelltime (in s) if data is stored as counts per second, else None

    Returns:
        data, structred array
        old_names, the original names used in text file
    """
    data = np.genfromtxt(
        path,
        delimiter=delimiter,
        usecols=columns,
        names=True,
        skip_header=first_line - 1,
        converters={0: lambda s: float(s.replace(",", "."))},
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
        "cell_diameter": "m",
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

        counts = [results[name]["indicies"].size for name in names]
        fp.write(f"# Detected particles,{','.join(str(c) for c in counts)}\n")
        fp.write(f"# Detection stdev,{','.join(str(np.sqrt(c)) for c in counts)}\n")
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

        detections = [results[n]["detections"][results[n]["indicies"]] for n in names]
        masses = [
            results[n]["masses"][results[n]["indicies"]]
            if "masses" in results[n]
            else 0.0
            for n in names
        ]
        sizes = [
            results[n]["sizes"][results[n]["indicies"]]
            if "sizes" in results[n]
            else 0.0
            for n in names
        ]
        concs = [
            results[n]["cell_concentrations"][results[n]["indicies"]]
            if "cell_concentrations" in results[n]
            else 0.0
            for n in names
        ]
        # === Mean values ===
        means = np.mean(detections, axis=0)
        mass_means = np.mean(masses, axis=0)
        size_means = np.mean(sizes, axis=0)
        conc_means = np.mean(concs, axis=0)

        fp.write(f"# Mean,{','.join(str(x or '') for x in means)},counts\n")
        if np.any(mass_means > 0.0):
            fp.write(f"#,{','.join(str(x or '') for x in mass_means)},kg\n")
        if np.any(size_means > 0.0):
            fp.write(f"#,{','.join(str(x or '') for x in size_means)},m\n")
        if np.any(conc_means > 0.0):
            fp.write(f"#,{','.join(str(x or '') for x in conc_means)},mol/L\n")

        # === Median values ===
        medians = np.median(detections, axis=0)
        mass_medians = np.median(masses, axis=0)
        size_medians = np.median(sizes, axis=0)
        conc_medians = np.median(concs, axis=0)

        fp.write(f"# Median,{','.join(str(x) for x in medians)},counts\n")
        if np.any(mass_medians > 0.0):
            fp.write(f"#,{','.join(str(x or '') for x in mass_medians)},kg\n")
        if np.any(size_medians > 0.0):
            fp.write(f"#,{','.join(str(x or '') for x in size_medians)},m\n")
        if np.any(conc_medians > 0.0):
            fp.write(f"#,{','.join(str(x or '') for x in conc_medians)},mol/L\n")

        if len(results) > 1:
            # Todo
            fp.write("#\n# Major Peak Compositions\n")

        fp.write("#\n# Raw detection data\n")
        # Output data
        data = []
        header_name = ""
        header_unit = ""

        for name in names:
            header_name += f",{name}"
            header_unit += ",counts"
            data.append(results[name]["detections"])
            for key, unit in [
                ("masses", "kg"),
                ("sizes", "m"),
                ("cell_concentrations", "mol/L"),
            ]:
                if key in results[name]:
                    header_name += f",{name}"
                    header_unit += f",{unit}"
                    data.append(results[name][key])

        data = np.stack(data, axis=1)

        fp.write(header_name[1:] + "\n")
        fp.write(header_unit[1:] + "\n")
        for line in data:
            fp.write(",".join("" if x == 0.0 else str(x) for x in line) + "\n")

        logger.info(f"Exported results for to {path.name}.")
