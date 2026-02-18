import logging
import re
from pathlib import Path

import h5py
import numpy as np
from numpy.lib import recfunctions as rfn

from spcal.calc import search_sorted_closest
from spcal.io import nu, text, tofwerk
from spcal.isotope import (
    ISOTOPE_TABLE,
    RECOMMENDED_ISOTOPES,
    SPCalIsotopeBase,
    SPCalIsotope,
    SPCalIsotopeExpression,
)
from spcal.pratt import Reducer, ReducerException

logger = logging.getLogger(__name__)


class SPCalDataFile(object):
    def __init__(
        self,
        path: Path,
        times: np.ndarray,
        instrument_type: str | None = None,
    ):
        if instrument_type not in ["quadrupole", "tof"]:  # pragma: no cover
            raise ValueError("instrument_type must be one of 'quadrupole', 'tof'")

        self.path = path
        self.instrument_type = instrument_type

        self._selected_isotopes: list[SPCalIsotope] = []

        self.times = times

        self._event_time = None

    def __repr__(self) -> str:  # pragma: no cover
        return f"{type(self).__name__}({self.path.stem})"

    def __getitem__(self, isotope: SPCalIsotopeBase) -> np.ndarray:
        if isinstance(isotope, SPCalIsotope):
            return self.dataForIsotope(isotope)
        elif isinstance(isotope, SPCalIsotopeExpression):
            return self.dataForExpression(isotope)
        else:  # pragma: no cover
            raise ValueError(f"cannot access data for isotope type {type(isotope)}")

    @property
    def selected_isotopes(self) -> list[SPCalIsotope]:
        return self._selected_isotopes

    @selected_isotopes.setter
    def selected_isotopes(self, selected: list[SPCalIsotope]):
        for isotope in selected:
            if isotope not in self.isotopes:  # pragma: no cover
                raise ValueError(f"{isotope} not in {self}")
        self._selected_isotopes = selected

    @property
    def preferred_isotopes(self) -> list[SPCalIsotope]:
        return [
            isotope
            for isotope in self.isotopes
            if isinstance(isotope, SPCalIsotope)
            and isotope.symbol in RECOMMENDED_ISOTOPES
            and RECOMMENDED_ISOTOPES[isotope.symbol] == isotope.isotope
        ]

    @property
    def event_time(self) -> float:
        if self._event_time is None:
            self._event_time = float(np.mean(np.diff(self.times)))
        return self._event_time

    @property
    def total_time(self) -> float:
        return self.times[-1] - self.times[0] + self.event_time

    @property
    def isotopes(self) -> list[SPCalIsotope]:  # pragma: no cover, not implemented
        raise NotImplementedError

    @property
    def num_events(self) -> int:  # pragma: no cover, not implemented
        raise NotImplementedError

    def dataForIsotope(self, isotope: SPCalIsotope) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def dataForExpression(self, expr: SPCalIsotopeExpression) -> np.ndarray:
        reducer = Reducer(
            variables={
                str(token): self.dataForIsotope(token)
                for token in expr.tokens
                if isinstance(token, SPCalIsotope)
            }
        )
        result = reducer.reduceExpr([str(t) for t in expr.tokens])
        if not isinstance(result, np.ndarray):  # pragma: no cover
            raise ReducerException("reduction of expression is not an array")
        result[~np.isfinite(result)] = np.nan  # set all infinite to nan
        return result

    def isTOF(self) -> bool:
        return self.instrument_type == "tof"


class SPCalTextDataFile(SPCalDataFile):
    def __init__(
        self,
        path: Path,
        signals: np.ndarray,
        times: np.ndarray,
        isotope_table: dict[SPCalIsotope, str],
        delimiter: str = ",",
        skip_rows: int = 1,
        cps: bool = False,
        drop_fields: list[str] | None = None,
        override_event_time: float | None = None,
        instrument_type: str | None = None,
    ):
        super().__init__(path, times=times, instrument_type=instrument_type)

        if signals.dtype.names is None:  # pragma: no cover
            raise ValueError("expected `signals` to have a structured dtype")
        for name in isotope_table.values():
            if name not in signals.dtype.names:  # pragma: no cover
                raise ValueError(
                    f"`isotope_table` '{name}' not found in `signals` array"
                )

        self.signals = signals
        self.isotope_table = isotope_table

        self.delimiter = delimiter
        self.skip_row = skip_rows
        self.cps = cps
        self.override_event_time = override_event_time

        self.drop_fields = drop_fields

    @property
    def isotopes(self) -> list[SPCalIsotope]:
        return list(self.isotope_table.keys())

    @property
    def num_events(self) -> int:
        return self.signals.shape[0]

    def dataForIsotope(self, isotope: SPCalIsotope) -> np.ndarray:
        return self.signals[self.isotope_table[isotope]]

    @classmethod
    def load(
        cls,
        path: Path,
        isotope_table: dict[SPCalIsotope, str] | None = None,
        delimiter: str = ",",
        skip_rows: int = 1,
        cps: bool = False,
        drop_fields: list[str] | None = None,
        override_event_time: float | None = None,
        instrument_type: str | None = None,
    ) -> "SPCalTextDataFile":
        signals = text.read_single_particle_file(
            path, delimiter=delimiter, skip_rows=skip_rows
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
                        else:  # pragma: no cover
                            logger.warning(
                                f"unit not found in times column for {path.stem}, assuming seconds"
                            )
                    break

        if times is None:  # pragma: no cover
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

        if isotope_table is None:
            isotope_table = {
                SPCalIsotope.fromString(name): name for name in signals.dtype.names
            }

        if cps:
            for name in signals.dtype.names:
                signals[name] *= np.diff(times, append=times[-1] + (times[-1] - times[-2]))

        if instrument_type is None:
            instrument_type = "quadrupole" if len(signals.dtype.names) == 1 else "tof"

        return cls(
            path,
            signals,
            times,
            isotope_table,
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
        cycle_number: int | None,
        segment_number: int | None,
        integ_files: tuple[int, int | None],
        max_mass_diff: float = 0.05,
    ):
        super().__init__(path, times, instrument_type="tof")

        self.info = info

        self.signals = signals
        self.masses = masses

        self.cycle_number = cycle_number
        self.segment_number = segment_number
        self.max_mass_diff = max_mass_diff
        self.integ_files = integ_files

        self.isotope_table: dict[SPCalIsotope, int] = {}
        self.generateIsotopeTable()

    @property
    def event_time(self) -> float:
        return nu.eventtime_from_info(self.info)

    @property
    def num_events(self) -> int:
        return self.signals.shape[0]

    @property
    def isotopes(self) -> list[SPCalIsotope]:
        return list(self.isotope_table.keys())

    def dataForIsotope(self, isotope: SPCalIsotope) -> np.ndarray:
        idx = self.isotope_table[isotope]
        return self.signals[:, idx]

    def generateIsotopeTable(self):
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
    def load(
        cls,
        path: Path,
        max_mass_diff: float = 0.05,
        cycle_number: int | None = None,
        segment_number: int | None = None,
        first_integ_file: int = 0,
        last_integ_file: int | None = None,
        autoblank: str = "regions",
    ) -> "SPCalNuDataFile":
        if path.is_file() and path.name == "run.info":
            path = path.parent

        masses, signals, times, info = nu.read_directory(
            path,
            cycle=cycle_number,
            segment=segment_number,
            first_integ_file=first_integ_file,
            last_integ_file=last_integ_file,
            autoblank=autoblank,
            raw=False,
        )

        return cls(
            path,
            signals,
            times,
            masses,
            info,
            cycle_number=cycle_number,
            segment_number=segment_number,
            integ_files=(first_integ_file, last_integ_file),
            max_mass_diff=max_mass_diff,
        )


class SPCalTOFWERKDataFile(SPCalDataFile):
    re_isotope = re.compile("\\[(\\d+)([A-Z][a-z]?)\\]+")

    def __init__(
        self, path: Path, signals: np.ndarray, times: np.ndarray, peak_table: np.ndarray
    ):
        super().__init__(path, times, instrument_type="tof")

        self.signals = signals
        self.times = times
        self.peak_table = peak_table

        self._event_time: float | None = None

        self.isotope_table: dict[SPCalIsotope, int] = {}

        self.generateIsotopeTable()

    def generateIsotopeTable(self):
        for i, peak in enumerate(self.peak_table):
            m = SPCalTOFWERKDataFile.re_isotope.match(peak["label"].decode())
            if m is None:
                continue
            self.isotope_table[ISOTOPE_TABLE[(m.group(2), int(m.group(1)))]] = i

    @property
    def event_time(self) -> float:
        if self._event_time is None:
            self._event_time = float(np.mean(np.diff(self.times)))
        return self._event_time

    @property
    def num_events(self) -> int:
        return self.signals.shape[0]

    @property
    def isotopes(self) -> list[SPCalIsotope]:
        return list(self.isotope_table.keys())

    @property
    def masses(self) -> np.ndarray:
        return self.peak_table["mass"]

    def dataForIsotope(self, isotope: SPCalIsotope) -> np.ndarray:
        idx = self.isotope_table[isotope]
        return self.signals[:, idx]

    @classmethod
    def load(cls, path: Path, max_size: int | None = None) -> "SPCalTOFWERKDataFile":
        with h5py.File(path) as h5:
            if "PeakData" in h5["PeakData"]:  # type: ignore , supported
                peak_data: np.ndarray = h5["PeakData"]["PeakData"][:max_size]  # type: ignore , returns numpy array
            elif "ToFData" in h5["FullSpectra"]:  # type: ignore , supported  # pragma: no cover, tested in test_io_tofwerk
                logger.warning(  # pragma: no cover
                    f"PeakData missing from TOFWERK file {path.stem}, integrating"
                )
                peak_data = tofwerk.integrate_tof_data(h5)[:max_size]
            else:  # pragma: no cover
                raise ValueError(
                    f"PeakData and ToFData are missing, {path.stem} is an invalid file"
                )

            peak_data = np.reshape(peak_data, (-1, peak_data.shape[-1]))

            peak_table: np.ndarray = h5["PeakData"]["PeakTable"][:]  # type: ignore , defined in tofdaq

            time_per_buf: float = h5["TimingData"].attrs["BlockPeriod"][0]  # type: ignore , defined in tofdaq
            times: np.ndarray = h5["TimingData"]["BufTimes"][:max_size]  # type: ignore , defined in tofdaq
            times = (
                times[:, :, None]
                + np.linspace(
                    0.0, time_per_buf * 1e-9, h5.attrs["NbrSegments"][0], endpoint=False
                )[None, None, :]
            ).ravel()

        return cls(path, signals=peak_data, times=times, peak_table=peak_table)
