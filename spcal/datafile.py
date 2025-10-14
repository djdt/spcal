import logging
import re
from pathlib import Path

import h5py
import numpy as np
from numpy.lib import recfunctions as rfn

from spcal.calc import search_sorted_closest
from spcal.io import nu, text, tofwerk
from spcal.npdb import db

logger = logging.getLogger(__name__)


class SPCalDataFile(object):
    def __init__(
        self,
        path: Path,
        times: np.ndarray,
        instrument_type: str | None = None,
    ):
        self.path = path
        self.instrument_type = instrument_type

        self.times = times

    def __getitem__(self, isotope: str) -> np.ndarray:
        raise NotImplementedError

    @property
    def event_time(self) -> float:
        if self._event_time is None:
            self._event_time = float(np.mean(np.diff(self.times)))
        return self._event_time

    @property
    def isotopes(self) -> list[str]:
        raise NotImplementedError

    @property
    def preferred_isotopes(self) -> list[str]:
        return self.isotopes

    @property
    def num_events(self) -> int:
        raise NotImplementedError

    def isotopeMass(self, isotope: str) -> float:
        raise NotImplementedError

    def isTOF(self) -> bool:
        return self.instrument_type == "TOF"


class SPCalTextDataFile(SPCalDataFile):
    def __init__(
        self,
        path: Path,
        signals: np.ndarray,
        times: np.ndarray,
        delimiter: str = ",",
        skip_rows: int = 1,
        cps: bool = False,
        override_event_time: float | None = None,
        instrument_type: str | None = None,
    ):
        super().__init__(path, times=times, instrument_type=instrument_type)

        if signals.dtype.names is None:
            raise ValueError("expected signals to have a structured dtype")
        self.signals = signals

        self.delimter = delimiter
        self.skip_row = skip_rows
        self.cps = cps
        self.override_event_time = override_event_time

    def __getitem__(self, isotope: str) -> np.ndarray:
        return self.signals[isotope]

    @property
    def isotopes(self) -> list[str]:
        assert self.signals.dtype.names is not None
        return list(self.signals.dtype.names)

    @property
    def num_events(self) -> int:
        return self.signals.shape[0]

    @classmethod
    def load(
        cls,
        path: Path,
        delimiter: str = ",",
        skip_rows: int = 1,
        cps: bool = False,
        max_rows: int | None = None,
        names: list[str] | None = None,
        override_event_time: float | None = None,
    ) -> "SPCalTextDataFile":
        with path.open("r") as fp:
            for i in range(skip_rows - 1):
                fp.readline()

            header = fp.readline().strip().split(delimiter)
            converters = {i: lambda s: float(s or 0.0) for i in range(len(header))}
            dtype = np.float32

            data_start_pos = fp.tell()
            peek = fp.readline()
            if "00:" in peek:  # we are dealing with a thremo iCap export
                converters = {1: lambda s: text.iso_time_to_float_seconds(s)}
            fp.seek(data_start_pos)

            gen = text.replace_comma_decimal(fp, delimiter)

            signals = np.genfromtxt(  # type: ignore
                gen,
                delimiter=delimiter,
                names=names or header,
                dtype=dtype,
                max_rows=max_rows,
                deletechars="",  # todo: see if this causes any issue with calculator or saving
                converters=converters,  # type: ignore , works
                invalid_raise=False,
                loose=True,
            )

        assert signals.dtype.names is not None

        if override_event_time is not None:
            times = np.arange(signals.shape[0]) * override_event_time
        else:
            times = None
            for name in signals.dtype.names:
                if "time" in name.lower():
                    times = signals[name]
                    m = re.search("(?<=[\\W_])([nmuµ]?s)\\b", name.lower())
                    if m is not None:
                        if m.group(1) in ["ms"]:
                            times *= 1e-3
                        elif m.group(1) in ["us", "µs"]:
                            times *= 1e-6
                        elif m.group(1) in ["ns"]:
                            times *= 1e-9
                        elif m.group(1) in ["s"]:
                            pass
                        else:
                            logger.warning(
                                f"unit not found in times column for {path.stem}, assuming seconds"
                            )
                    signals = rfn.drop_fields(signals, [name])
                    break

        if times is None:
            raise ValueError(
                f"unable to read times in {path.stem} and no 'override_event_time' provided"
            )

        if cps:
            signals /= np.diff(times, append=times[-1] - times[-2])

        return cls(
            path,
            signals,
            times,
            cps=cps,
            override_event_time=override_event_time,
        )


class SPCalNuDataFile(SPCalDataFile):
    """Data file for data from a Nu Instruments Vitesse.

    Attributes:
        isotope_table: dict of {isotope name: (index in signals, isotope mass)}
    """

    re_isotope = re.compile("(\\d+)([A-Z][a-z]?)")

    def __init__(
        self,
        path: Path,
        signals: np.ndarray,
        times: np.ndarray,
        masses: np.ndarray,
        info: dict,
        max_mass_diff: float = 0.05,
    ):
        super().__init__(path, times, instrument_type="TOF")

        self.info = info

        self.signals = signals
        self.masses = masses
        self.max_mass_diff = max_mass_diff

        self.isotope_table: dict[str, tuple[int, float, bool]] = {}
        self.generateIsotopeTable()

    def __getitem__(self, isotope: str) -> np.ndarray:
        return self.signals[:, self.isotope_table[isotope][0]].reshape(-1)

    @property
    def isotopes(self) -> list[str]:
        return list(self.isotope_table.keys())

    @property
    def preferred_isotopes(self) -> list[str]:
        return [key for key, val in self.isotope_table.items() if val[2]]

    def generateIsotopeTable(self) -> None:
        """Creates a table of isotope names in format '123Ab' to their indicies
        in signals / masses and their isotopic mass."""
        natural = db["isotopes"][~np.isnan(db["isotopes"]["Composition"])]
        indicies = search_sorted_closest(self.masses, natural["Mass"])
        valid = np.abs(self.masses[indicies] - natural["Mass"]) < self.max_mass_diff
        self.isotope_table = {
            f"{iso['Isotope']}{iso['Symbol']}": (idx, iso["Mass"], iso["Preferred"])
            for iso, idx in zip(natural[valid], indicies[valid])
        }

    def isotopeMass(self, isotope: str) -> float:
        m = SPCalNuDataFile.re_isotope.match(isotope)
        if m is None:
            raise ValueError(f"invalid isotope format {isotope}")
        return self.isotope_table[isotope][1]

    @classmethod
    def load(cls, path: Path, max_mass_diff: float = 0.05) -> "SPCalNuDataFile":
        if path.is_file() and path.stem == "run.info":
            path = path.parent

        masses, signals, info = nu.read_nu_directory(path, raw=False)

        return cls(path, signals, times, masses, info, max_mass_diff=max_mass_diff)


class SPCalTOFWERKDataFile(SPCalDataFile):
    def __init__(
        self, path: Path, signals: np.ndarray, times: np.ndarray, peak_table: np.ndarray
    ):
        self.signals = signals
        self.peak_table = peak_table

        super().__init__(path, times)

    def __getitem__(self, isotope: str) -> np.ndarray:
        idx = self.isotopes.index(isotope)
        return self.signals[..., idx].ravel()

    @property
    def isotopes(self) -> list[str]:
        return [x["label"].decode() for x in self.peak_table]

    @property
    def preferred_isotopes(self) -> list[str]:
        preferred = []
        re_iso = re.compile("\\[(\\d+)([A-Z][a-z]?)\\]+")
        for isotope in self.isotopes:
            m = re_iso.match(isotope)
            if m is None:
                continue
            if db["isotopes"][
                np.logical_and(
                    db["isotopes"]["Isotope"] == int(m.group(1)),
                    db["isotopes"]["Symbol"] == m.group(2),
                )
            ]["Preferred"]:
                preferred.append(isotope)
        return preferred

    @property
    def masses(self) -> np.ndarray:
        return self.peak_table["mass"]

    def isotopeMass(self, isotope: str) -> float:
        return float(self.peak_table[self.peak_table["label"] == isotope]["mass"])

    @classmethod
    def load(cls, path: Path) -> "SPCalTOFWERKDataFile":
        with h5py.File(path) as h5:
            if "PeakData" in h5["PeakData"]:  # type: ignore , supported
                peak_data: np.ndarray = h5["PeakData"]["PeakData"][:]  # type: ignore , returns numpy array
            elif "ToFData" in h5["FullSpectra"]:  # type: ignore , supported
                logger.warning(
                    f"PeakData missing from TOFWERK file {path.stem}, integrating"
                )
                peak_data = tofwerk.integrate_tof_data(h5)
            else:
                raise ValueError(
                    f"PeakData and ToFData are missing, {path.stem} is an invalid file"
                )

            peak_data = np.reshape(peak_data, (-1, peak_data.shape[-1]))

            peak_table: np.ndarray = h5["PeakData"]["PeakTable"][:]  # type: ignore , defined in tofdaq

            time_per_buf: float = h5["TimingData"].attrs["BlockPeriod"][0]  # type: ignore , defined in tofdaq
            times: np.ndarray = h5["TimingData"]["BufTimes"][:]  # type: ignore , defined in tofdaq
            times = (
                times[:, :, None]
                + np.linspace(0.0, time_per_buf * 1e-9, 1000, endpoint=False)[
                    None, None, :
                ]
            ).ravel()

        return cls(path, signals=peak_data, times=times, peak_table=peak_table)


# x = SPCalNuDataFile.load(
#     Path("/home/tom/Downloads/14-38-58 UPW + 80nm Au 90nm UCNP many particles")
# )
# print(x.isotopes)
# print(x.preferred_isotopes)

x = SPCalTextDataFile.load(
    Path("/home/tom/Documents/python/spcal/tests/data/text/tofwerk_export_au_bg.csv")
    # Path("/home/tom/Downloads/Single cell_blank_2025-08-27_15h39m38s.h5")
)
print(x.isotopes)
print(x.preferred_isotopes)
