import numpy as np
from pathlib import Path

from typing import Dict, Tuple, Union

from spcal import __version__


def read_nanoparticle_file(
    path: Union[Path, str], delimiter: str = ","
) -> Tuple[np.ndarray, Dict]:
    """Imports data and parameters from a NP export.

    Data is expected to be a text or csv in a single column of responses, or twos columns:
    1) aquisition times, 2) responses. If two columns are found then the parameter 'dwelltime'
    will be set as the mean difference of the first coulmn. If 'cps' is read in the file header
    then the parameter 'cps' will be set to True.

    Tested with Agilent exports.

    Args:
        path: path to the file
        delimiter: text delimiter, default to comma

    Returns:
        signal
        dict of any parameters
    """

    def delimited_columns(path: Path, delimiter: str = ",", columns: int = 2):
        """Ensures at least `columns` columns in data by prepending `delimiter`."""
        with path.open("r") as fp:
            for line in fp:
                count = line.count(delimiter)
                if count < columns:
                    yield delimiter * (columns - count - 1) + line
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
        delimited_columns(path, delimiter, 2),
        delimiter=delimiter,
        usecols=(0, 1),
        dtype=np.float64,
    )
    parameters = read_header_params(path)

    response = data[:, 1]
    # Check if data in two column format
    if not np.all(np.isnan(data[:, 0])):  # time and response
        times = data[:, 0][~np.isnan(data[:, 0])]
        parameters["dwelltime"] = np.round(np.mean(np.diff(times)), 6)

    # Remove any invalid rows, e.g. headers
    response = response[~np.isnan(response)]

    return response, parameters


def export_nanoparticle_results(path: Path, result: dict) -> None:
    """Writes data from a results dict.

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
    with path.open("w") as fp:
        fp.write(f"# SPCal Export {__version__}\n")
        fp.write(f"# File,'{result['file']}'\n")
        fp.write(f"# Acquisition events,{result['events']}\n")

        fp.write(f"# Options and inputs\n")
        for k, v in result["inputs"].items():
            fp.write(f'#,{k.replace("_", " ").capitalize()},{v}\n')

        fp.write(f"# Detected particles,{result['detections'].size}\n")
        fp.write(f"# Detection stddev,{result['detections_std']}\n")
        fp.write(f"# Limit method,{str(result['limit_method']).replace(',', ';')}\n")
        if result["limit_window"] is not None and result["limit_window"] > 1:
            fp.write(f"# Limit window,{result['limit_window']}\n")

        # Background
        fp.write(f"# Background,{result['background']},counts\n")
        if "background_size" in result:
            fp.write(f"#,{result['background_size']},m\n")
        if "background_concentration" in result:
            fp.write(f"# Ionic background,{result['background_concentration']},kg/L\n")
        fp.write(f"# Background stddev,{result['background_std']},counts\n")

        # LODs
        if isinstance(result["lod"], np.ndarray):
            fp.write("# Limit of detection,Min,Max,Mean,Median\n")
            fp.write(f"#,{','.join(str(s) for s in (result['lod']))},counts\n")
        else:
            fp.write(f"# Limit of detection,{result['lod']},counts\n")

        for key, unit in [
            ("lod_mass", "kg"),
            ("lod_size", "m"),
            ("lod_cell_concentration", "mol/L"),
        ]:
            if key in result:
                if isinstance(result[key], np.ndarray):
                    fp.write(f"#,{','.join(str(s) for s in (result[key]))},{unit}\n")
                else:
                    fp.write(f"#,{result[key]},{unit}\n")

        # Concentrations
        if "number_concentration" in result:
            fp.write(f"# Number concentration,{result['number_concentration']},#/L\n")
        if "concentration" in result:
            fp.write(f"# Concentration,{result['concentration']},kg/L\n")

        # Mean values
        fp.write(f"# Mean,{np.mean(result['detections'])},counts\n")
        for key, unit in [
            ("masses", "kg"),
            ("sizes", "m"),
            ("cell_concentrations", "mol/L"),
        ]:
            if key in result:
                fp.write(f"#,{np.mean(result[key])},{unit}\n")
        # Median values
        fp.write(f"# Median,{np.median(result['detections'])},counts\n")
        for key, unit in [
            ("masses", "kg"),
            ("sizes", "m"),
            ("cell_concentrations", "mol/L"),
        ]:
            if key in result:
                fp.write(f"#,{np.median(result[key])},{unit}\n")

        # Output data
        header = "Signal (counts)"
        data = [result["detections"]]
        for key, label in [
            ("masses", "Mass (kg)"),
            ("sizes", "Size (m)"),
            ("cell_concentrations", "Conc. (mol/L)"),
        ]:
            if key in result:
                header += "," + label
                data.append(result[key])
        fp.write(header + "\n")
        np.savetxt(
            fp,
            np.stack(data, axis=1),
            delimiter=",",
        )
