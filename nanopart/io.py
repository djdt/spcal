import numpy as np
from pathlib import Path

from typing import Dict, Tuple, Union

from nanopart import __version__


def read_nanoparticle_file(
    path: Union[Path, str], delimiter: str = ","
) -> Tuple[np.ndarray, Dict]:
    def delimited_columns(path: Path, delimiter: str = ",", columns: int = 2):
        with path.open("r") as fp:
            for line in fp:
                count = line.count(delimiter)
                if count < columns:
                    line += delimiter * (columns - count - 1)
                yield line

    def read_header_params(path: Path, size: int = 1024) -> Dict:
        with path.open("r") as fp:
            header = fp.read(size)

        parameters = {"cps": "cps" in header.lower()}
        return parameters

    if isinstance(path, str):
        path = Path(path)

    data = np.genfromtxt(
        delimited_columns(path, delimiter, 2), delimiter=delimiter, dtype=np.float64
    )
    parameters = read_header_params(path)

    if np.all(np.isnan(data[:, 1])):  # only one column exists
        response = data[:, 0]
    else:  # assume time and response
        response = data[:, 1]
        times = data[:, 0][~np.isnan(data[:, 0])]
        parameters["dwelltime"] = np.round(np.mean(np.diff(times)), 6)

    response = response[~np.isnan(response)]

    return response, parameters


def export_nanoparticle_results(path: Path, result: dict) -> None:
    with path.open("w") as fp:
        fp.write(f"# NanoPart Export {__version__}\n")
        fp.write(f"# File,'{result['file']}'\n")
        fp.write(f"# Acquisition events,{result['events']}\n")
        fp.write(f"# Detected particles,{result['detections'].size}\n")
        fp.write(f"# Detection stddev,{result['stddev']}\n")
        fp.write(f"# Limit method,{result['limit_method']}\n")
        if result["limit_window"] is not None and result["limit_window"] > 1:
            fp.write(f"# Limit window,{result['limit_window']}\n")

        # Background
        fp.write(f"# Background,{result['background']},counts\n")
        if "background_size" in result:
            fp.write(f"#,{result['background_size']},m\n")
        if "background_concentration" in result:
            fp.write(f"# Ionic background,{result['background_concentration']},kg/L\n")

        # LODs
        if isinstance(result["lod"], np.ndarray):
            fp.write("# Limit of detection,Min,Max,Mean,Median\n")
            fp.write(f"#,{','.join(str(s) for s in (result['lod']))},counts\n")
        else:
            fp.write(f"# Limit of detection,{result['lod']},counts\n")
        if "lod_mass" in result:
            if isinstance(result["lod_mass"], np.ndarray):
                fp.write(f"#,{','.join(str(s) for s in (result['lod_mass']))},kg\n")
            else:
                fp.write(f"#,{result['lod_mass']},kg\n")
        if "lod_size" in result:
            if isinstance(result["lod_size"], np.ndarray):
                fp.write(f"#,{','.join(str(s) for s in (result['lod_size']))},m\n")
            else:
                fp.write(f"#,{result['lod_size']},m\n")

        # Concentrations
        if "number_concentration" in result:
            fp.write(f"# Number concentration,{result['number_concentration']},#/L\n")
        if "concentration" in result:
            fp.write(f"# Concentration,{result['concentration']},kg/L\n")

        # Mean sizes
        fp.write(f"# Mean size,{np.mean(result['sizes'])},m\n")
        fp.write(f"# Median size,{np.median(result['sizes'])},m\n")

        # Output data
        fp.write("Signal (counts),Mass (kg),Size (m)\n")
        np.savetxt(
            fp,
            np.stack((result["detections"], result["masses"], result["sizes"]), axis=1),
            delimiter=",",
        )
