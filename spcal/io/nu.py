"""Loading data from Nu Instruments ICP-ToF."""

import json
import logging
from math import gamma
from pathlib import Path
from typing import BinaryIO, Generator

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


def get_blanking_regions(
    autob_events: list[np.ndarray],
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
                    int(start_event["acq_number"][0] // num_acc) - 1,
                    int(event["acq_number"][0] // num_acc) - 1,
                )
            )

            start_masses = (
                start_coef[0] + start_coef[1] * start_event["edges"][0][::2] * 1.25
            ) ** 2
            end_masses = (
                end_coef[0] + end_coef[1] * start_event["edges"][0][1::2] * 1.25
            ) ** 2
            valid = start_masses < end_masses
            mass_regions.append(
                np.stack([start_masses[valid], end_masses[valid]], axis=1)
            )

            start_event = None

    return regions, mass_regions


def blank_nu_signal_data(
    autob_events: list[np.ndarray],
    signals: np.ndarray,
    masses: np.ndarray,
    num_acc: int,
    start_coef: tuple[float, float],
    end_coef: tuple[float, float],
) -> np.ndarray:
    """Apply the auto-blanking to the integrated data.
    There must be one cycle / segment and no missing acquisitions / data!

    Args:
        autob: list of events from `read_nu_autob_binary`
        signals: 2d array of signals from `get_signals_from_nu_data`
        masses: 1d array of masses, from `get_masses_from_nu_data`
        num_acc: number of accumulations per acquisition
        start_coef: blanker open coefs 'BlMassCalStartCoef'
        end_coef: blanker close coefs 'BlMassCalEndCoef'

    Returns:
        blanked data
    """
    regions, mass_regions_list = get_blanking_regions(
        autob_events, num_acc, start_coef, end_coef
    )
    for region, mass_regions in zip(regions, mass_regions_list):
        mass_idx = np.searchsorted(masses, mass_regions)
        # There are a bunch of useless blanking regions
        mass_idx = mass_idx[mass_idx[:, 0] != mass_idx[:, 1]]
        for s, e in mass_idx:
            signals[region[0] : region[1], s:e] = np.nan

    return signals


def collect_nu_autob_data(
    root: Path,
    index: list[dict],
    cyc_number: int | None = None,
    seg_number: int | None = None,
) -> list[np.ndarray]:
    autobs = []
    for idx in index:
        autob_path = root.joinpath(f"{idx['FileNum']}.autob")
        if autob_path.exists():
            events = read_nu_autob_binary(
                autob_path,
                idx["FirstCycNum"],
                idx["FirstSegNum"],
                idx["FirstAcqNum"],
            )
            if cyc_number is not None:
                events = [ev for ev in events if ev["cyc_number"] == cyc_number]
            if seg_number is not None:
                events = [ev for ev in events if ev["seg_number"] == seg_number]
            autobs.extend(events)
        else:  # pragma: no cover, missing files
            logger.warning(
                f"collect_nu_autob_data: missing autob {idx['FileNum']}, skipping"
            )
    return autobs


def collect_nu_integ_data(
    root: Path,
    index: list[dict],
    cyc_number: int | None = None,
    seg_number: int | None = None,
) -> list[np.ndarray]:
    integs = []
    for idx in index:
        integ_path = root.joinpath(f"{idx['FileNum']}.integ")
        if integ_path.exists():
            data = read_nu_integ_binary(
                integ_path,
                idx["FirstCycNum"],
                idx["FirstSegNum"],
                idx["FirstAcqNum"],
            )
            if cyc_number is not None:
                data = data[data["cyc_number"] == cyc_number]
            if seg_number is not None:
                data = data[data["seg_number"] == seg_number]
            if data.size > 0:
                integs.append(data)
        else:
            logger.warning(  # pragma: no cover, missing files
                f"collect_nu_integ_data: missing integ {idx['FileNum']}, skipping"
            )
    return integs


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


def get_signals_from_nu_data(integs: list[np.ndarray], num_acc: int) -> np.ndarray:
    """Converts signals from integ data to counts.

    Preserves run length when missing data is present.

    Args:
        integ: from `read_integ_binary`
        num_acc: number of accumulations per acquisition

    Returns:
        signals in counts
    """

    max_acq = max(integ["acq_number"][-1] for integ in integs if integ.size > 0)
    signals = np.full(
        (max_acq // num_acc, integs[0]["result"]["signal"].shape[1]),
        np.nan,
        dtype=np.float32,
    )
    for integ in integs:
        signals[(integ["acq_number"] // num_acc) - 1] = integ["result"]["signal"]

    return signals


def get_masses_from_nu_data(
    integ: np.ndarray, cal_coef: tuple[float, float], segment_delays: dict[int, float]
) -> np.ndarray:
    """Converts Nu peak centers into masses.

    Args:
        integ: from `read_integ_binary`
        cal_coef: from run.info 'MassCalCoefficients'
        segment_delays: dict of segment nums and delays from `SegmentInfo`

    Returns:
        2d array of masses
    """

    delays = np.zeros(max(segment_delays.keys()))
    for k, v in segment_delays.items():
        delays[k - 1] = v
    delays = np.atleast_1d(delays[integ["seg_number"] - 1])

    masses = (integ["result"]["center"] * 0.5) + delays[:, None]
    # Convert from time to mass (sqrt(m/q) = a + t * b)
    return (cal_coef[0] + masses * cal_coef[1]) ** 2


def read_nu_autob_binary(
    path: Path,
    first_cyc_number: int | None = None,
    first_seg_number: int | None = None,
    first_acq_number: int | None = None,
) -> list[np.ndarray]:
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
        fp.seek(0)

        return np.frombuffer(fp.read(), dtype=integ_dtype(num_results))


def read_nu_directory(
    path: str | Path,
    max_integ_files: int | None = None,
    autoblank: bool = True,
    cycle: int | None = None,
    segment: int | None = None,
    raw: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read the Nu Instruments raw data directory, retuning data and run info.

    Directory must contain 'run.info', 'integrated.index' and at least one '.integ'
    file. Data is read from '.integ' files listed in the 'integrated.index' and
    are checked for correct starting cycle, segment and acquisition numbers.

    Args:
        path: path to data directory
        max_integ_files: maximum number of files to read
        autoblank: apply autoblanking to overrange regions
        cycle: limit import to cycle
        segment: limit import to segment
        raw: return raw ADC counts

    Returns:
        masses from first acquisition
        signals in counts
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

    if max_integ_files is not None:
        integ_index = integ_index[:max_integ_files]

    segment_delays = {
        s["Num"]: s["AcquisitionTriggerDelayNs"] for s in run_info["SegmentInfo"]
    }

    accumulations = run_info["NumAccumulations1"] * run_info["NumAccumulations2"]

    # Collect integrated data
    integs = collect_nu_integ_data(
        path, integ_index, cyc_number=cycle, seg_number=segment
    )

    # Get masses from data
    masses = get_masses_from_nu_data(
        integs[0], run_info["MassCalCoefficients"], segment_delays
    )[0]
    signals = get_signals_from_nu_data(integs, accumulations)

    if not raw:
        signals /= run_info["AverageSingleIonArea"]

    # Blank out overrange regions
    if autoblank:
        autobs = collect_nu_autob_data(
            path, autob_index, cyc_number=cycle, seg_number=segment
        )
        signals = blank_nu_signal_data(
            autobs,
            signals,
            masses,
            accumulations,
            run_info["BlMassCalStartCoef"],
            run_info["BlMassCalEndCoef"],
        )

    # Account for any missing integ files
    return masses, signals, run_info


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


def single_ion_distribution(
    counts: np.ndarray, bins: str | int | np.ndarray = "auto"
) -> np.ndarray:
    """Calculates the single ion distribution from calibration data.

    SIA is calculated as an average of all isotopes with a >1e05 and <1e-3
    chance of containing a two ion event.

    Args:
        counts: raw ADC counts
        bins: binning of histogram, see np.histogram_bin_edges

    Returns:
        array of stacked bin centers and counts
    """

    def incgamma(a: float, x: float) -> float:
        xs = np.linspace(0, x, 100)
        return 1.0 / gamma(a) * np.trapezoid(np.exp(-xs) * xs ** (a - 1), x=xs)

    pzeros = np.count_nonzero(counts, axis=0) / counts.shape[0]
    poi2 = np.array([incgamma(2 + 1, pz) for pz in pzeros])
    x = counts[:, (poi2 > 1e-5) & (poi2 < 1e-3)]

    hist, bins = np.histogram(x[x > 0], bins=bins)
    return np.stack(((bins[1:] + bins[:-1]) / 2.0, hist), axis=1)
