from pathlib import Path
import re
import datetime

import h5py
import numpy as np
from numpy.lib import recfunctions as rfn

import logging

from spcal.io import nu, tofwerk
from spcal.npdb import db
from spcal.calc import search_sorted_closest

logger = logging.getLogger(__name__)


class SPCalDataFile(object):
    def __init__(
        self,
        path: Path,
        event_times: np.ndarray | float | None = None,
        instrument_type: str | None = None,
    ):
        self.path = path
        self.instrument_type = instrument_type

        self._times = None
        self._event_time = None
        if isinstance(event_times, np.ndarray):
            self._times = event_times
        else:
            self._event_time = event_times

    def __getitem__(self, isotope: str) -> np.ndarray:
        raise NotImplementedError

    @property
    def event_time(self) -> float:
        if self._event_time is None:
            self._event_time = float(np.mean(np.diff(self.event_times)))
        return self._event_time

    @property
    def event_times(self) -> np.ndarray:
        if self._times is None:
            self._times = np.arange(self.num_events) * self.event_time
        return self._times

    @property
    def isotopes(self) -> list[str]:
        raise NotImplementedError

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
        event_times: np.ndarray | float | None = None,
        delimiter: str = ",",
        skip_rows: int = 1,
        cps: bool = False,
        instrument_type: str | None = None,
    ):
        super().__init__(path, event_times=event_times, instrument_type=instrument_type)

        self.signals = signals

        self.delimter = delimiter
        self.skip_row = skip_rows
        self.cps = cps

    def __getitem__(self, isotope: str) -> np.ndarray:
        return self.signals[isotope]

    @property
    def names(self) -> list[str]:
        return list(self.signals.dtype.names or [])

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
    ) -> "SPCalTextDataFile":
        def replace_comma_decimal(fp, delimiter: str = ",", count: int = 0):
            for line in fp:
                if delimiter != ",":
                    yield line.replace(",", ".")
                else:
                    yield line

        def iso_time_to_float_seconds(text: str) -> float:
            time = datetime.time.fromisoformat(text)
            return (
                time.hour * 3600.0
                + time.minute * 60.0
                + time.second
                + time.microsecond * 1e-6
            )

        with path.open("r") as fp:
            for i in range(skip_rows - 1):
                fp.readline()

            header = fp.readline().strip().split(delimiter)
            converters = {i: lambda s: float(s or 0.0) for i in range(len(header))}
            dtype = np.float32

            data_start_pos = fp.tell()
            peek = fp.readline()
            if "00:" in peek:  # we are dealing with a thremo iCap export
                converters = {1: lambda s: iso_time_to_float_seconds(s)}
            fp.seek(data_start_pos)

            gen = replace_comma_decimal(fp, delimiter)

            signals = np.genfromtxt(  # type: ignore
                gen,
                delimiter=delimiter,
                names=header,
                dtype=dtype,
                max_rows=max_rows,
                converters=converters,  # type: ignore , works
                invalid_raise=False,
                loose=True,
            )

        assert signals.dtype.names is not None

        times = None
        for name in signals.dtype.names:
            if "time" in name.lower():
                times = signals[name]
                m = re.search("[\\(\\[]([nmuµ]s)[\\]\\)]", name.lower())
                if m is not None:
                    if m.group(1) in ["ms"]:
                        times *= 1e-3
                    elif m.group(1) in ["us", "µs"]:
                        times *= 1e-6
                    elif m.group(1) in ["ns"]:
                        times *= 1e-9
                signals = rfn.drop_fields(signals, [name])
                break

        return cls(path, signals, event_times=times)


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
        masses: np.ndarray,
        info: dict,
        max_mass_diff: float = 0.05,
    ):
        super().__init__(path, instrument_type="TOF")

        self.info = info

        self.signals = signals
        self.masses = masses
        self.max_mass_diff = max_mass_diff

        self.isotope_table: dict[str, tuple[int, float]] = {}
        self.generateIsotopeTable()

    def __getitem__(self, isotope: str) -> np.ndarray:
        return self.signals[:, self.isotope_table[isotope][0]].reshape(-1)

    @property
    def isotopes(self) -> list[str]:
        return list(self.isotope_table.keys())

    def generateIsotopeTable(self) -> None:
        """Creates a table of isotope names in format '123Ab' to their indicies
        in signals / masses and their isotopic mass."""
        natural = db["isotopes"][~np.isnan(db["isotopes"]["Composition"])]
        indicies = search_sorted_closest(self.masses, natural["Mass"])
        valid = np.abs(self.masses[indicies] - natural["Mass"]) < self.max_mass_diff
        self.isotope_table = {
            f"{iso['Isotope']}{iso['Symbol']}": (idx, iso["Mass"])
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

        return cls(path, signals, masses, info, max_mass_diff=max_mass_diff)


class SPCalTOFWERKDataFile(SPCalDataFile):
    def __init__(self, path: Path, h5: h5py.File):
        super().__init__(path)

        self.h5 = h5

        if "PeakData" in self.h5["PeakData"]:  # type: ignore , supported
            self.signals: np.ndarray = self.h5["PeakData"]["PeakData"][:]  # type: ignore , returns numpy array
        elif "ToFData" in self.h5["FullSpectra"]:  # type: ignore , supported
            logger.warning(
                f"PeakData missing from TOFWERK file {path.stem}, integrating"
            )
            self.signals = tofwerk.integrate_tof_data(self.h5)
        else:
            raise ValueError(
                f"PeakData and ToFData are missing, {path.stem} is an invalid file"
            )

    def __getitem__(self, isotope: str) -> np.ndarray:
        idx = self.isotopes.index(isotope)
        return self.signals[..., idx].ravel()

    @property
    def isotopes(self) -> list[str]:
        return [x["label"].decode() for x in self.h5["PeakData"]["PeakTable"]]  # type: ignore , specified in tofdaq

    @property
    def masses(self) -> np.ndarray:
        return self.h5["PeakData"]["PeakTable"]["mass"]  # type: ignore , specified in tofdaq

    def isotopeMass(self, isotope: str) -> float:
        idx = self.h5["PeakData"]["PeakTable"]["label"] == isotope  # type: ignore , is defined
        return float(self.h5["PeakData"]["PeakTable"][idx]["mass"])  # type: ignore , in tofdaq

    @classmethod
    def load(cls, path: Path) -> "SPCalTOFWERKDataFile":
        return cls(path, h5py.File(path))
