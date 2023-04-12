import json
import logging
from pathlib import Path
from typing import BinaryIO, Dict, Generator, List, Tuple

import numpy as np
import numpy.lib.recfunctions as rfn

logger = logging.getLogger(__name__)


def is_nu_run_info_file(path: Path) -> bool:
    if not path.exists() or path.name != "run.info":
        return False
    return True


def is_nu_directory(path: Path) -> bool:
    """Checks path is directory containing a 'run.info' and 'integrated.index'"""

    if not path.is_dir() or not path.exists():
        return False
    if not path.joinpath("run.info").exists():
        return False
    if not path.joinpath("integrated.index").exists():
        return False

    return True


def blank_nu_integ_data(
    autob_events: List[np.ndarray],
    integ_data: np.ndarray,
    masses: np.ndarray,
    num_acc: int,
    start_coef: Tuple[float, float],
    end_coef: Tuple[float, float],
) -> np.ndarray:
    """Apply the auto-blanking to the integrated data.
    There must be one cycle / segment and no missing acquisitions / data!

    Args:
        autob: list of events from `read_nu_autob_binary`
        integ_data: concatenated data from `read_nu_integ_data_binary`
        masses: 1d array of masses, from `get_masses_from_nu_data`
        num_acc: number of accumulations per acquisition
        start_coef: blanker open coefs 'BlMassCalStartCoef'
        end_coef: blanker close coefs 'BlMassCalEndCoef'

    Returns:
        blanked data
    """
    cycle, seg = integ_data[0]["cyc_number"], integ_data[0]["seg_number"]

    start = None
    for ab in autob_events:
        if ab["cyc_number"] != cycle or ab["seg_number"] != seg:
            continue
        if ab["type"] == 0 and start is None:
            start = ab
        elif ab["type"] == 1 and start is not None:
            idx = np.searchsorted(
                integ_data["acq_number"], (start["acq_number"], ab["acq_number"])
            ).ravel()

            start_masses = (
                start_coef[0] + start_coef[1] * start["edges"][0][::2] * 1.25
            ) ** 2
            end_masses = (
                end_coef[0] + end_coef[1] * start["edges"][0][1::2] * 1.25
            ) ** 2
            mass_idx = np.searchsorted(masses, (start_masses, end_masses))
            # There are a bunch of useless blanking regions
            mass_idx = mass_idx[:, mass_idx[0] != mass_idx[1]]

            for s, e in mass_idx.T:
                integ_data["result"]["signal"][idx[0] : idx[1], s:e] = np.nan

            # Slower
            # slices = tuple(slice(s, e) for s, e in mass_idx.T)
            # integ_data["result"]["signal"][idx[0] : idx[1], np.r_[slices]] = np.nan
            start = None
    return integ_data


def get_dwelltime_from_info(info: dict) -> float:
    """Reads the dwelltime (total acquistion time) from run.info.
    Rounds to the nearest ns.

    Args:
        info: dict of parameters, as returned by `read_nu_directory`

    Returns:
        dwelltime in s
    """
    seg = info["SegmentInfo"][0]
    acqtime = seg["AcquisitionPeriodNs"] * 1e-9
    accumulations = info["NumAccumulations1"] * info["NumAccumulations2"]
    return np.around(acqtime * accumulations, 9)


def get_masses_from_nu_data(
    data: np.ndarray, cal_coef: Tuple[float, float], segment_delays: Dict[int, float]
) -> np.ndarray:
    """Converts Nu peak centers into masses.

    Args:
        data: from `read_integ_binary`
        cal_coef: from run.info 'MassCalCoefficients'
        segment_delays: dict of segment nums and delays from `SegmentInfo`

    Returns:
        2d array of masses
    """

    delays = np.zeros(max(segment_delays.keys()))
    for k, v in segment_delays.items():
        delays[k - 1] = v
    delays = np.atleast_1d(delays[data["seg_number"] - 1])

    masses = (data["result"]["center"] * 0.5) + delays[:, None]
    # Convert from time to mass (sqrt(m/q) = a + t * b)
    return (cal_coef[0] + masses * cal_coef[1]) ** 2


def read_nu_autob_binary(
    path: Path,
    first_cyc_number: int | None = None,
    first_seg_number: int | None = None,
    first_acq_number: int | None = None,
) -> List[np.ndarray]:
    def autob_dtype(size: int) -> np.dtype:
        return np.dtype(
            [
                ("cyc_number", np.uint32),
                ("seg_number", np.uint32),
                ("acq_number", np.uint32),
                ("trig_start_time", np.uint32),
                ("trig_end_time", np.uint32),
                ("type", np.uint8),
                ("num_edges", np.int32),
                ("edges", np.uint32, size),
            ]
        )

    def read_autob_events(fp: BinaryIO) -> Generator[np.ndarray, None, None]:
        while fp:
            data = fp.read(4 + 4 + 4 + 4 + 4 + 1 + 4)
            if not data:
                return
            size = int.from_bytes(data[-4:], "little")
            autob = np.empty(1, dtype=autob_dtype(size))
            autob.data.cast("B")[: len(data)] = data
            if size > 0:
                autob["edges"] = np.frombuffer(fp.read(size * 4), dtype=np.uint32)
            yield autob

    with path.open("rb") as fp:
        autob_events = list(read_autob_events(fp))

    return autob_events


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


def read_nu_directory(
    path: str | Path, max_integ_files: int | None = None, autoblank: bool = True
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Read the Nu Instruments raw data directory, retuning data and run info.

    Directory must contain 'run.info', 'integrated.index' and at least one '.integ'
    file. Data is read from '.integ' files listed in the 'integrated.index' and
    are checked for correct starting cyle, segment and acquistion numbers.

    Args:
        path: path to data directory
        max_integ_files: maximum number of files to read
        autoblank: apply autoblanking to overrange regions

    Returns:
        masses from first acquistion
        signals in counts
        dict of parameters from run.info
    """

    path = Path(path)
    if not is_nu_directory(path):
        raise ValueError("read_nu_directory: missing 'run.info' or 'integrated.index'")

    with path.joinpath("run.info").open("r") as fp:
        run_info = json.load(fp)
    with path.joinpath("autob.index").open("r") as fp:
        autob_index = json.load(fp)
    with path.joinpath("integrated.index").open("r") as fp:
        integ_index = json.load(fp)

    if max_integ_files is not None:
        integ_index = integ_index[:max_integ_files]

    segment_delays = {
        s["Num"]: s["AcquisitionTriggerDelayNs"] for s in run_info["SegmentInfo"]
    }

    accumulations = run_info["NumAccumulations1"] * run_info["NumAccumulations2"]

    # Collect integrated data
    datas = []
    missing_integ = False
    for idx in integ_index:
        integ_path = path.joinpath(f"{idx['FileNum']}.integ")
        if integ_path.exists():
            datas.append(
                read_nu_integ_binary(
                    integ_path,
                    idx["FirstCycNum"],
                    idx["FirstSegNum"],
                    idx["FirstAcqNum"],
                )
            )
        else:
            missing_integ = True
            logger.warning(
                f"read_integ_binary: missing integ {idx['FileNum']}, skipping"
            )
    data = np.concatenate(datas)

    if not np.all(data[0]["cyc_number"] == data[1:]["cyc_number"]):
        logger.warning("read_nu_directory: multiple cycles not supported.")
        data = data[data["cyc_number"] == data[0]["cyc_number"]]
    if not np.all(data[0]["seg_number"] == data[1:]["seg_number"]):
        logger.warning("read_nu_directory: multiple segments not supported.")
        data = data[data["seg_number"] == data[0]["seg_number"]]

    # Get masses from data
    masses = get_masses_from_nu_data(
        data[0], run_info["MassCalCoefficients"], segment_delays
    )

    # Blank out overrange regions
    if autoblank:
        autobs = []
        for idx in autob_index:
            autob_path = path.joinpath(f"{idx['FileNum']}.autob")
            if autob_path.exists():
                autobs.extend(
                    read_nu_autob_binary(
                        autob_path,
                        idx["FirstCycNum"],
                        idx["FirstSegNum"],
                        idx["FirstAcqNum"],
                    )
                )
            else:
                logger.warning(
                    f"read_nu_directory: missing autob {idx['FileNum']}, skipping"
                )

        data = blank_nu_integ_data(
            autobs,
            data,
            masses[0],
            accumulations,
            run_info["BlMassCalStartCoef"],
            run_info["BlMassCalEndCoef"],
        )

    # Account for any missing integ files
    if missing_integ:
        signals = np.full(
            (
                data[-1]["acq_number"] // accumulations,
                data[-1]["result"]["signal"].shape[0],
            ),
            np.nan,
            dtype=np.float32,
        )
        signals[(data["acq_number"] // accumulations) - 1] = (
            data["result"]["signal"] / run_info["AverageSingleIonArea"]
        )
    else:
        signals = data["result"]["signal"] / run_info["AverageSingleIonArea"]

    return masses[0], signals, run_info


def select_nu_signals(
    masses: np.ndarray,
    signals: np.ndarray,
    selected_masses: Dict[str, float],
    max_mass_diff: float = 0.1,
) -> np.ndarray:
    """Reduces signals to the isotopes in selected_masses.
    'masses' must be sorted

    Args:
        masses: from `read_nu_directory`
        signals: from `read_nu_directory`
        selected_masses: dict of isotope name: mass
        max_mass_diff: maximum difference (Da) from mass to allow

    Returns:
        structured array of signals

    Raises:
        ValueError if the smallest mass difference from 'selected_masses' is
            greater than 'max_mass_diff'
    """

    def find_closest_idx(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(x, y, side="left")
        prev_less = np.abs(y - x[np.maximum(idx - 1, 0)]) < np.abs(
            y - x[np.minimum(idx, len(x) - 1)]
        )
        prev_less = (idx == len(x)) | prev_less
        idx[prev_less] -= 1
        return idx

    assert np.all(masses[:-1] <= masses[1:])  # check sorted

    selected = np.fromiter(selected_masses.values(), dtype=np.float32)
    idx = find_closest_idx(masses, selected)

    diffs = np.abs(masses[idx] - selected)

    if np.any(diffs > max_mass_diff):
        raise ValueError(
            "select_nu_signals: could not find mass closer than 'max_mass_diff'"
        )

    dtype = np.dtype(
        {
            "names": list(selected_masses.keys()),
            "formats": [np.float32 for _ in idx],
        }
    )
    return rfn.unstructured_to_structured(signals[:, idx], dtype=dtype)
