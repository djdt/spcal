import logging
import re
from pathlib import Path

import h5py
import numpy as np
from numpy.lib import recfunctions as rfn

from spcal.calc import search_sorted_closest
from spcal.io import nu, text, tofwerk
from spcal.isotope import ISOTOPE_TABLE, SPCalIsotope

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

        self.selected_isotopes: list[SPCalIsotope] = []

        self.times = times

        self._event_time = None

    def __getitem__(self, isotope: SPCalIsotope) -> np.ndarray:
        raise NotImplementedError

    @property
    def event_time(self) -> float:
        if self._event_time is None:
            self._event_time = float(np.mean(np.diff(self.times)))
        return self._event_time

    @property
    def total_time(self) -> float:
        return self.times[-1] - self.times[0]
        # return self.event_time * self.num_events

    @property
    def isotopes(self) -> list[SPCalIsotope]:
        raise NotImplementedError

    @property
    def num_events(self) -> int:
        raise NotImplementedError

    def isTOF(self) -> bool:
        return self.instrument_type == "tof"


class SPCalTextDataFile(SPCalDataFile):
    def __init__(
        self,
        path: Path,
        signals: np.ndarray,
        times: np.ndarray,
        isotopes: list[SPCalIsotope],
        delimiter: str = ",",
        skip_rows: int = 1,
        cps: bool = False,
        drop_fields: list[str] | None = None,
        override_event_time: float | None = None,
        instrument_type: str | None = None,
    ):
        super().__init__(path, times=times, instrument_type=instrument_type)

        if signals.dtype.names is None:
            raise ValueError("expected signals to have a structured dtype")
        if len(isotopes) != len(signals.dtype.names):
            raise ValueError("number of isotopes does not match names in signals")

        self.signals = signals
        self.isotope_table = {
            iso: name for iso, name in zip(isotopes, signals.dtype.names)
        }

        self.delimter = delimiter
        self.skip_row = skip_rows
        self.cps = cps
        self.override_event_time = override_event_time
        # self.rename_fields = rename_fields
        self.drop_fields = drop_fields

    def __getitem__(self, isotope: SPCalIsotope) -> np.ndarray:
        return self.signals[self.isotope_table[isotope]]

    @property
    def isotopes(self) -> list[SPCalIsotope]:
        return list(self.isotope_table.keys())

    @property
    def num_events(self) -> int:
        return self.signals.shape[0]

    @classmethod
    def load(
        cls,
        path: Path,
        isotopes: list[SPCalIsotope] | None = None,
        delimiter: str = ",",
        skip_rows: int = 1,
        cps: bool = False,
        drop_fields: list[str] | None = None,
        override_event_time: float | None = None,
        instrument_type: str | None = None,
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
                names=header,
                dtype=dtype,
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
                    break

        if times is None:
            raise ValueError(
                f"unable to read times in {path.stem} and no 'override_event_time' provided"
            )

        if drop_fields is None:
            drop_fields = [
                name
                for name in signals.dtype.names
                if any(x in name.lower() for x in ["index", "time"])
            ]
        signals = rfn.drop_fields(signals, drop_fields)
        assert signals.dtype.names is not None

        if isotopes is None:
            isotopes = [SPCalIsotope.fromString(name) for name in signals.dtype.names]

        if cps:
            for name in signals.dtype.names:
                signals[name] /= np.diff(times, append=times[-1] - times[-2])

        if instrument_type is None:
            instrument_type = "quadrupole" if len(signals.dtype.names) == 1 else "tof"

        return cls(
            path,
            signals,
            times,
            isotopes,
            delimiter=delimiter,
            skip_rows=skip_rows,
            cps=cps,
            override_event_time=override_event_time,
            drop_fields=drop_fields,
            instrument_type=instrument_type,
        )


class SPCalNuDataFile(SPCalDataFile):
    """Data file for data from a Nu Instruments Vitesse.

    Attributes:
        isotope_table: dict of {isotope name: (index in signals, isotope mass)}
    """

    # re_isotope = re.compile("(\\d+)([A-Z][a-z]?)")

    def __init__(
        self,
        path: Path,
        signals: np.ndarray,
        times: np.ndarray,
        masses: np.ndarray,
        info: dict,
        max_mass_diff: float = 0.05,
    ):
        super().__init__(path, times, instrument_type="tof")

        self.info = info

        self.signals = signals
        self.masses = masses
        self.max_mass_diff = max_mass_diff

        self.isotope_table: dict[SPCalIsotope, int] = {}
        self.generateIsotopeTable()

    def __getitem__(self, isotope: SPCalIsotope) -> np.ndarray:
        idx = self.isotope_table[isotope]
        return self.signals[:, idx].reshape(-1)

    @property
    def num_events(self) -> int:
        return self.signals.shape[0]

    @property
    def isotopes(self) -> list[SPCalIsotope]:
        return list(self.isotope_table.keys())

    def generateIsotopeTable(self) -> None:
        """Creates a table of isotope names in format '123Ab' to indicies and isotope array."""
        natural_isotopes = [
            iso for iso in ISOTOPE_TABLE.values() if iso.composition is not None
        ]
        natural_masses = np.fromiter(
            (iso.mass for iso in natural_isotopes), dtype=float
        )
        indices = search_sorted_closest(self.masses, natural_masses)
        valid = np.abs(self.masses[indices] - natural_masses) < self.max_mass_diff
        self.isotope_table = {
            iso: idx for idx, iso, v in zip(indices, natural_isotopes, valid) if v
        }

    @classmethod
    def load(cls, path: Path, max_mass_diff: float = 0.05) -> "SPCalNuDataFile":
        if path.is_file() and path.stem == "run.info":
            path = path.parent

        masses, signals, times, info = nu.read_directory(path, raw=False)

        return cls(path, signals, times, masses, info, max_mass_diff=max_mass_diff)


class SPCalTOFWERKDataFile(SPCalDataFile):
    re_isotope = re.compile("\\[(\\d+)([A-Z][a-z]?)\\]+")

    def __init__(
        self, path: Path, signals: np.ndarray, times: np.ndarray, peak_table: np.ndarray
    ):
        self.signals = signals
        self.peak_table = peak_table

        self.isotope_table = {}

        super().__init__(path, times, instrument_type="tof")

    def __getitem__(self, isotope: SPCalIsotope) -> np.ndarray:
        idx = self.isotopes.index(isotope)
        return self.signals[..., idx].ravel()

    @property
    def isotopes(self) -> list[str]:
        return [x["label"].decode() for x in self.peak_table]

    @property
    def masses(self) -> np.ndarray:
        return self.peak_table["mass"]

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
