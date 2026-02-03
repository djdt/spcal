"""Loading data from Nu Instruments ICP-ToF."""

import json
import logging
from pathlib import Path
from typing import BinaryIO, Callable, Generator

import numpy as np
import numpy.lib.recfunctions as rfn

from spcal.calc import search_sorted_closest

logger = logging.getLogger(__name__)


def is_nu_run_info_file(path: Path) -> bool:
    """Checks file exists and is called 'run.info'."""
    if not path.exists() or path.name != "run.info":
        return False
    return True


def is_nu_directory(path: Path) -> bool:
    """Checks path is directory containing a 'run.info' and 'integrated.index'"""

    if not path.is_dir() or not path.exists():
        return False
    if not path.joinpath("run.info").exists():
        return False
    if not path.joinpath("integrated.index").exists():  # pragma: no cover
        return False

    return True


def read_autob_binary(
    path: Path,
    first_cyc_number: int | None = None,
    first_seg_number: int | None = None,
    first_acq_number: int | None = None,
) -> np.ndarray:
    data_dtype = np.dtype(
        [
            ("cyc_number", np.uint32),
            ("seg_number", np.uint32),
            ("acq_number", np.uint32),
            ("trig_start_time", np.uint32),
            ("trig_end_time", np.uint32),
            ("type", np.uint8),
            ("num_edges", np.int32),
            ("edges", np.uint32, 12),  # so far 12 is the maximum
        ]
    )

    def read_autoblank_events(fp: BinaryIO) -> Generator[np.ndarray, None, None]:
        while fp:
            partial = fp.read(25)
            if len(partial) < 25:
                return
            autob = np.zeros(1, dtype=data_dtype)
            autob.data.cast("B")[:25] = partial
            num = autob["num_edges"][0]
            if num > 0:
                autob["edges"][:num] = np.frombuffer(fp.read(num * 4), dtype=np.uint32)
            yield autob

    with path.open("rb") as fp:
        autob = np.concatenate(list(read_autoblank_events(fp)))

    if autob.size > 0:
        if first_cyc_number is not None and autob[0]["cyc_number"] != first_cyc_number:
            raise ValueError("read_integ_binary: incorrect FirstCycNum")
        if first_seg_number is not None and autob[0]["seg_number"] != first_seg_number:
            raise ValueError("read_integ_binary: incorrect FirstSegNum")
        if first_acq_number is not None and autob[0]["acq_number"] != first_acq_number:
            raise ValueError("read_integ_binary: incorrect FirstAcqNum")

    return autob


def read_integ_binary(
    path: Path,
    first_cyc_number: int | None = None,
    first_seg_number: int | None = None,
    first_acq_number: int | None = None,
    memmap: bool = False,
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
        if (
            first_cyc_number is not None and cyc_number != first_cyc_number
        ):  # pragma: no cover
            raise ValueError("read_integ_binary: incorrect FirstCycNum")
        seg_number = int.from_bytes(fp.read(4), "little")
        if (
            first_seg_number is not None and seg_number != first_seg_number
        ):  # pragma: no cover
            raise ValueError("read_integ_binary: incorrect FirstSegNum")
        acq_number = int.from_bytes(fp.read(4), "little")
        if (
            first_acq_number is not None and acq_number != first_acq_number
        ):  # pragma: no cover
            raise ValueError("read_integ_binary: incorrect FirstAcqNum")
        num_results = int.from_bytes(fp.read(4), "little")

    if memmap:
        return np.memmap(path, dtype=integ_dtype(num_results), mode="r")
    else:
        return np.fromfile(path, dtype=integ_dtype(num_results))


def read_binaries_in_index(
    root: Path,
    index: list[dict],
    binary_ext: str,
    binary_read_fn: Callable[[Path, int, int, int], np.ndarray],
    binary_read_kwargs: dict | None = None,
    cyc_number: int | None = None,
    seg_number: int | None = None,
) -> list[np.ndarray]:
    datas = []
    if binary_read_kwargs is None:
        binary_read_kwargs = {}
    for idx in index:
        binary_path = root.joinpath(f"{idx['FileNum']}.{binary_ext}")
        if binary_path.exists():
            data = binary_read_fn(
                binary_path,
                idx["FirstCycNum"],
                idx["FirstSegNum"],
                idx["FirstAcqNum"],
                **binary_read_kwargs,
            )
            if cyc_number is not None:
                data = data[data["cyc_number"] == cyc_number]
            if seg_number is not None:
                data = data[data["seg_number"] == seg_number]
            datas.append(data)
        else:
            logger.warning(  # pragma: no cover, missing files
                f"collect_data_from_index: missing data file {idx['FileNum']}.{binary_ext}, skipping"
            )
    return datas


def apply_autoblanking(
    autob_events: np.ndarray,
    signals: np.ndarray,
    masses: np.ndarray,
    info: dict,
) -> np.ndarray:
    """Apply the auto-blanking to the integrated data.
    There must be one cycle / segment and no missing acquisitions / data!

    Args:
        autob: list of events from `read_nu_autob_binary`
        signals: 2d array of signals from `get_signals_from_nu_data`
        masses: 1d array of masses, from `get_masses_from_nu_data`
        info: dict of parameters, as returned by `read_nu_directory`

    Returns:
        blanked data
    """
    num_acc = info["NumAccumulations1"] * info["NumAccumulations2"]
    start_coef = info["BlMassCalStartCoef"]
    end_coef = info["BlMassCalEndCoef"]

    regions, mass_regions_list = blanking_regions_from_autob(
        autob_events, num_acc, start_coef, end_coef
    )
    for region, mass_regions in zip(regions, mass_regions_list):
        mass_idx = np.searchsorted(masses, mass_regions)
        # There are a bunch of useless blanking regions
        mass_idx = mass_idx[mass_idx[:, 0] != mass_idx[:, 1]]
        for s, e in mass_idx:
            signals[region[0] : region[1], s:e] = np.nan

    return signals


def blanking_regions_from_autob(
    autob_events: np.ndarray,
    num_acc: int,
    start_coef: tuple[float, float],
    end_coef: tuple[float, float],
) -> tuple[list[tuple[int, int]], list[np.ndarray]]:
    """Extract blanking regions from autoblank data.

    Args:
        autob: list of events from `read_nu_autob_binary`
        num_acc: number of accumulations per acquisition
        start_coef: blanker open coefs 'BlMassCalStartCoef'
        end_coef: blanker close coefs 'BlMassCalEndCoef'

    Returns:
        list of (start, end) of each region, array of (start, end) masses
    """
    regions: list[tuple[int, int]] = []
    mass_regions = []

    start_event = None
    for event in autob_events:
        if event["type"] == 0 and start_event is None:
            start_event = event
        elif event["type"] == 1 and start_event is not None:
            regions.append(
                (
                    int(start_event["acq_number"] // num_acc) - 1,
                    int(event["acq_number"] // num_acc) - 1,
                )
            )

            start_masses = (
                start_coef[0]
                + start_coef[1]
                * start_event["edges"][: start_event["num_edges"]][::2]
                * 1.25
            ) ** 2
            end_masses = (
                end_coef[0]
                + end_coef[1]
                * start_event["edges"][: start_event["num_edges"]][1::2]
                * 1.25
            ) ** 2
            valid = start_masses < end_masses
            mass_regions.append(
                np.stack([start_masses[valid], end_masses[valid]], axis=1)
            )

            start_event = None

    return regions, mass_regions


def eventtime_from_info(info: dict) -> float:
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


def signals_from_integs(integs: list[np.ndarray], info: dict) -> np.ndarray:
    """Converts signals from integ data to counts.

    Inserts nan values for discontinuities in integ index.

    Args:
        integ: from `read_integ_binary`
        info: dict of parameters, as returned by `read_nu_directory`

    Returns:
        signals in counts
    """

    num_acc = info["NumAccumulations1"] * info["NumAccumulations2"]
    segment_acq = np.array([seg["AcquisitionCount"] for seg in info["SegmentInfo"]])

    def indicies_for_integ(integ: np.ndarray) -> np.ndarray:
        idx = (integ["cyc_number"] - 1) * segment_acq[integ["seg_number"] - 1]
        idx += integ["acq_number"]
        return idx // num_acc

    signals = [integ["result"]["signal"] for integ in integs]
    for i in range(len(integs) - 1):
        if (
            indicies_for_integ(integs[i][-1])
            != indicies_for_integ(integs[i + 1][0]) - 1
        ):
            integs[i][-1]["result"]["signal"] = np.nan

    return np.concatenate(signals, axis=0)


def masses_from_integ(integ: np.ndarray, info: dict) -> np.ndarray:
    """Converts Nu peak centers into masses.

    Args:
        integ: from `read_integ_binary`
        info: dict of parameters, as returned by `read_nu_directory`

    Returns:
        2d array of masses
    """

    cal_coef = info["MassCalCoefficients"]
    segment_delays = {
        s["Num"]: s["AcquisitionTriggerDelayNs"] for s in info["SegmentInfo"]
    }

    delays = np.zeros(max(segment_delays.keys()))
    for k, v in segment_delays.items():
        delays[k - 1] = v
    delays = np.atleast_1d(delays[integ["seg_number"] - 1])

    masses = (integ["result"]["center"] * 0.5) + delays[:, None]
    # Convert from time to mass (sqrt(m/q) = a + t * b)
    return (cal_coef[0] + masses * cal_coef[1]) ** 2


def times_from_integs(integs: list[np.ndarray], run_info: dict) -> np.ndarray:
    seg_times = np.array(
        [
            seg["AcquisitionPeriodNs"] * seg["AcquisitionCount"]
            for seg in run_info["SegmentInfo"]
        ]
    )
    seg_periods = np.array(
        [seg["AcquisitionPeriodNs"] for seg in run_info["SegmentInfo"]]
    )

    times = np.sum(seg_times) * (np.concatenate([x["cyc_number"] for x in integs]) - 1)
    times += np.cumsum(np.concatenate([[0], seg_times]))[
        np.concatenate([x["seg_number"] for x in integs]) - 1
    ]
    times += (
        np.concatenate([x["acq_number"] for x in integs])
        * seg_periods[np.concatenate([x["seg_number"] for x in integs]) - 1]
    )
    return times


def read_directory(
    path: str | Path,
    first_integ_file: int = 0,
    last_integ_file: int | None = None,
    autoblank: bool = True,
    cycle: int | None = None,
    segment: int | None = None,
    raw: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Read the Nu Instruments raw data directory, retuning data and run info.

    Directory must contain 'run.info', 'integrated.index' and at least one '.integ'
    file. Data is read from '.integ' files listed in the 'integrated.index' and
    are checked for correct starting cycle, segment and acquisition numbers.

    Args:
        path: path to data directory
        first_integ_file: first integ to read
        last_integ_file: last integ to read, can be used as a max
        autoblank: apply autoblanking to overrange regions
        cycle: limit import to cycle
        segment: limit import to segment
        raw: return raw ADC counts

    Returns:
        masses from first acquisition
        signals in counts
        times in s
        dict of parameters from run.info
    """

    path = Path(path)
    if not is_nu_directory(path):  # pragma: no cover
        raise ValueError("read_nu_directory: missing 'run.info' or 'integrated.index'")

    with path.joinpath("run.info").open("r") as fp:
        run_info = json.load(fp)
    with path.joinpath("autob.index").open("r") as fp:
        autob_index = json.load(fp)
    with path.joinpath("integrated.index").open("r") as fp:
        integ_index = json.load(fp)

    integ_index = integ_index[first_integ_file:last_integ_file]
    if len(integ_index) == 0:
        raise ValueError("no integ files selected")

    # Collect integrated data
    integs = read_binaries_in_index(
        path,
        integ_index,
        "integ",
        read_integ_binary,
        cyc_number=cycle,
        seg_number=segment,
        binary_read_kwargs={"memmap": False},
    )

    # Get masses from data
    masses = masses_from_integ(integs[0], run_info)[0]
    signals = signals_from_integs(integs, run_info)
    times = times_from_integs(integs, run_info) * 1e-9

    if not raw:
        signals /= run_info["AverageSingleIonArea"]

    # Blank out overrange regions
    if autoblank:
        autobs = np.concatenate(
            read_binaries_in_index(
                path,
                autob_index,
                "autob",
                read_autob_binary,
                cyc_number=cycle,
                seg_number=segment,
            )
        )
        signals = apply_autoblanking(autobs, signals, masses, run_info)

    # Account for any missing integ files
    return masses, signals, times, run_info


def select_nu_signals(
    masses: np.ndarray,
    signals: np.ndarray,
    selected_masses: dict[str, float],
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
    selected = np.fromiter(selected_masses.values(), dtype=np.float32)
    idx = search_sorted_closest(masses, selected, check_max_diff=max_mass_diff)

    dtype = np.dtype(
        {
            "names": list(selected_masses.keys()),
            "formats": [np.float32 for _ in idx],
        }
    )
    return rfn.unstructured_to_structured(signals[:, idx], dtype=dtype)
