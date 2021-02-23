import numpy as np
from pathlib import Path

from typing import Dict, Tuple, Union


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
