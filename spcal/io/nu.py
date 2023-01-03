import json
from pathlib import Path
from typing import BinaryIO, Dict, List, Tuple

import numpy as np
import numpy.lib.recfunctions as rfn

from spcal.npdb import db


def read_integ_binary(
    path: Path,
    first_cyc_number: int | None = None,
    first_seg_number: int | None = None,
    first_acq_number: int | None = None,
) -> np.ndarray:
    def integ_dtype(size: int) -> np.dtype:
        data_dtype = np.dtype(
            {
                "names": ["center", "signal"],
                "formats": [np.float32, np.float32],
                "itemsize": 4 + 4 + 4 + 1,  # unused f32, unused i8
            }
        )
        return np.dtype(
            [
                ("cyc_number", np.uint32),
                ("seg_number", np.uint32),
                ("acq_number", np.uint32),
                ("num_results", np.uint32),
                ("result", data_dtype, size),
            ]
        )

    with path.open("rb") as fp:
        cyc_number = int.from_bytes(fp.read(4), "little")
        if first_cyc_number is not None and cyc_number != first_cyc_number:
            raise ValueError("read_integ_binary: incorrect FirstCycNum")
        seg_number = int.from_bytes(fp.read(4), "little")
        if first_seg_number is not None and seg_number != first_seg_number:
            raise ValueError("read_integ_binary: incorrect FirstSegNum")
        acq_number = int.from_bytes(fp.read(4), "little")
        if first_acq_number is not None and acq_number != first_acq_number:
            raise ValueError("read_integ_binary: incorrect FirstAcqNum")
        num_results = int.from_bytes(fp.read(4), "little")
        fp.seek(0)
        return np.frombuffer(fp.read(), dtype=integ_dtype(num_results))


def read_nu_directory(
    path: str | Path, threads: int | None = 1
) -> Tuple[np.ndarray, dict]:
    """Read the Nu Instruments raw data directory, retuning data and run info.

    Directory must contain 'run.info', 'integrated.index' and at least one '.integ'
    file. Data is read from '.integ' files listed in the 'integrated.index' and
    are checked for correct starting cyle, segment and acquistion numbers.
    If threads != 1, a ProcessPoolExecutor is used for multithreaded IO.

    Args:
        path: path to data directory
        threads: number of threads to use for import

    Returns:
        concatenated data from the integ files
        dict of parameters from run.info
    """
    path = Path(path)
    run_path = path.joinpath("run.info")
    if not run_path.exists():
        raise FileNotFoundError("read_nu_directory: missing 'run.info'")
    with run_path.open("r") as fp:
        run_info = json.load(fp)

    integ_path = path.joinpath("integrated.index")
    if not integ_path.exists():
        raise FileNotFoundError("read_nu_directory: missing 'integrated.index'")
    with integ_path.open("r") as fp:
        integ_index = json.load(fp)

    # Todo, single threaded is faster on uni PC, check others
    # PC , Single, Multi
    # uni, 1.66  , 3.44
    if threads == 1:
        data = np.concatenate(
            [
                read_integ_binary(
                    path.joinpath(f"{idx['FileNum']}.integ"),
                    idx["FirstCycNum"],
                    idx["FirstSegNum"],
                    idx["FirstAcqNum"],
                )
                for idx in integ_index
            ]
        )
    else:
        import concurrent.futures

        integs = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as ex:
            futures = [
                ex.submit(
                    read_integ_binary,
                    path.joinpath(f"{idx['FileNum']}.integ"),
                    idx["FirstCycNum"],
                    idx["FirstSegNum"],
                    idx["FirstAcqNum"],
                )
                for idx in integ_index
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    integs.append(future.result())
                except Exception as e:
                    print(f"read_nu_directory: exception for {future}, {e}")

        data = np.concatenate(integs)
        order = np.argsort(data["acq_number"])
        data = data[order]

    return data, run_info


def format_nu_integ_data(
    data: np.ndarray, cal_coef: Tuple[float, float], segment_delays: Dict[int, float]
) -> np.ndarray:
    """Converts Nu data into a structured array of integrated signals.

    Uses data and info from `read_integ_binary`. The returned array is a 1d named
    array of signals, with each name being the symbol and isotope with the mass
    closest to that of the integrated regions center, in the format Ag107.
    Requires all data to have the same peak centers and segment number.

    Args:
        data: from `read_integ_binary`
        cal_coef: from run.info 'MassCalCoefficients'
        segment_delays: dict of segments 'Num' to 'AcquisitionTriggerDelayNs'

    Returns:
        1d named array of signals
    """

    def isotopes_for_masses(masses: np.ndarray) -> np.ndarray:
        idx = np.argmin(np.abs(db["isotopes"]["Mass"] - masses[:, None]), axis=1)
        return db["isotopes"][idx]

    # Todo, we could calculate the actual times and only require the Symbol/Isotope to be the same.
    if not np.all(data["result"]["center"] == data["result"]["center"][0]):
        raise ValueError(
            "format_nu_integ_data: unable to format, variable peak centers"
        )
    if not np.all(data["seg_number"] == data["seg_number"][0]):
        raise ValueError("format_nu_integ_data: segments must all be the same")

    # Masses are shared across data
    masses = data["result"]["center"][0] * 0.5
    masses += segment_delays[data["seg_number"][0]]

    # Convert from time to mass (sqrt(m/q) = a + t * b)
    masses = (cal_coef[0] + masses * cal_coef[1]) ** 2

    dtype = np.dtype(
        [
            (f"{iso['Symbol']}{iso['Isotope']}", np.float32)
            for iso in isotopes_for_masses(masses)
        ]
    )

    return rfn.unstructured_to_structured(data["result"]["signal"], dtype=dtype)
