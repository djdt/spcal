import json
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np


def get_masses_from_nu_data(
    data: np.ndarray, cal_coef: Tuple[float, float], segment_delays: Dict[int, float]
) -> np.ndarray:
    """Converts Nu peak centers into masses.

    Args:
        data: from `read_integ_binary`
        delays: array of delays from `get_delays_for_segments`
        cal_coef: from run.info 'MassCalCoefficients'

    Returns:
        1d named array of signals
    """

    delays = np.zeros(max(segment_delays.keys()))
    for k, v in segment_delays.items():
        delays[k - 1] = v
    delays = np.atleast_1d(delays[data["seg_number"] - 1])

    masses = (data["result"]["center"] * 0.5) + delays[:, None]
    # Convert from time to mass (sqrt(m/q) = a + t * b)
    return (cal_coef[0] + masses * cal_coef[1]) ** 2


def read_nu_integ_binary(
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


def is_nu_directory(path: str | Path) -> bool:
    """Checks path is directory containing a 'run.info' and 'integrated.index'"""
    path = Path(path)
    if not path.is_dir():
        return False
    if not path.joinpath("run.info").exists():
        return False
    if not path.joinpath("integrated.index").exists():
        return False

    return True


def read_nu_directory(
    path: str | Path, threads: int | None = 1, raw: bool = False
) -> Union[Tuple[np.ndarray, dict], Tuple[np.ndarray, np.ndarray, dict]]:
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
    if not is_nu_directory(path):
        raise ValueError("read_nu_directory: missing 'run.info' or 'integrated.index'")

    with path.joinpath("run.info").open("r") as fp:
        run_info = json.load(fp)
    with path.joinpath("integrated.index").open("r") as fp:
        integ_index = json.load(fp)

    # Todo, single threaded is faster on uni PC, check others
    # PC , Single, Multi
    # uni, 1.66  , 3.44
    if threads == 1:
        data = np.concatenate(
            [
                read_nu_integ_binary(
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
                    read_nu_integ_binary,
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

    if raw:
        return data, run_info

    segment_delays = {
        s["Num"]: s["AcquisitionTriggerDelayNs"] for s in run_info["SegmentInfo"]
    }

    masses = get_masses_from_nu_data(
        data[0], run_info["MassCalCoefficients"], segment_delays
    )
    return masses, data["result"]["signal"], run_info
