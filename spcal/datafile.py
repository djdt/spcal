"""File import and data access."""
# Copyright 2025 Thomas Lockwood
# SPDX-License-Identifier: GPL-3.0-or-later

import logging
import re
from pathlib import Path
import datetime

import h5py
import bottleneck as bn
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
    """A base class for data files.

    Attributes:

        exclusion_regions: array of start, end times to exclude from processing.
            These regions are set to NaN, shape (..., 2)
    """

    def __init__(
        self,
        format: str,
        path: Path,
        times: np.ndarray,
    ):
        """Args:

        format: the type of data file ('text', 'nu', 'tofwerk')
        path: Path to the data file
        times: array of event times
        """
        self.path = path
        self.format = format

        self._selected_isotopes: list[SPCalIsotope] = []

        self.times = times

        self._event_time = None

        self.exclusion_regions: list[tuple[float, float]] = []

    def __repr__(self) -> str:  # pragma: no cover
        return f"{type(self).__name__}({self.path.stem})"

    def __getitem__(self, isotope: SPCalIsotopeBase) -> np.ndarray:
        if isinstance(isotope, SPCalIsotope):
            signals = self.dataForIsotope(isotope).copy()
        elif isinstance(isotope, SPCalIsotopeExpression):
            signals = self.dataForExpression(isotope)
        else:  # pragma: no cover
            raise ValueError(f"cannot access data for isotope type {type(isotope)}")

        if len(self.exclusion_regions) > 0:
            idx = np.searchsorted(self.times, self.exclusion_regions)
            for start, end in idx:
                signals[start:end] = np.nan

        return signals

    @property
    def selected_isotopes(self) -> list[SPCalIsotope]:
        """Currently selected isotopes."""
        return self._selected_isotopes

    @selected_isotopes.setter
    def selected_isotopes(self, selected: list[SPCalIsotope]):
        for isotope in selected:
            if isotope not in self.isotopes:  # pragma: no cover
                raise ValueError(f"{isotope} not in {self}")
        self._selected_isotopes = selected

    @property
    def preferred_isotopes(self) -> list[SPCalIsotope]:
        """Returns the default isotope for each element."""
        return [
            isotope
            for isotope in self.isotopes
            if isinstance(isotope, SPCalIsotope)
            and isotope.symbol in RECOMMENDED_ISOTOPES
            and RECOMMENDED_ISOTOPES[isotope.symbol] == isotope.isotope
        ]

    @property
    def event_time(self) -> float:
        """The time of a single acuqisition. Sometimes called the 'dwell time'."""
        if self._event_time is None:
            self._event_time = float(np.mean(np.diff(self.times)))
        return self._event_time

    @property
    def total_time(self) -> float:
        """The total time in seconds, excluding NaN regions."""
        return self.times[-1] - self.times[0] + self.event_time

    @property
    def isotopes(self) -> list[SPCalIsotope]:  # pragma: no cover, not implemented
        """The number of isotopes.
        This may be differetn from the number of masses due to isobars."""
        raise NotImplementedError

    @property
    def num_events(self) -> int:  # pragma: no cover, not implemented
        """The number of events (e.g. acquisitions)."""
        raise NotImplementedError

    @property
    def masses(self) -> np.ndarray:  # pragma: no cover, not implemented
        """Masses (m/z) in the data file."""
        raise NotImplementedError

    @property
    def signals(self) -> np.ndarray:  # pragma: no cover, not implemented
        """Signal intensities in the data file.
        Has the shape (num_events, masses)."""
        raise NotImplementedError

    def dataForIsotope(self, isotope: SPCalIsotope) -> np.ndarray:  # pragma: no cover
        """Access signals for `isotope`."""
        raise NotImplementedError

    def dataForExpression(self, expr: SPCalIsotopeExpression) -> np.ndarray:
        """Calculate the result for `expr`."""
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

    def spectra(self, regions: np.ndarray) -> np.ndarray:
        """Access the entire mass spectra between `regions`.

        Args:
            regions: array start, end points of shape (N, 2)

        Returns:
            array of spectra shape (N, masses)
        """
        spectra = np.zeros((regions.shape[0], self.signals.shape[1]), dtype=np.float32)

        for i, region in enumerate(regions):
            spectra[i] = bn.nanmean(self.signals[region[0] : region[1]], axis=0)
        return spectra

    def information(self) -> dict[str, str]:
        return {
            "path": str(self.path.resolve()),
            "format": self.format,
            "event time": f"{self.event_time * 1e-6} µs",
            "total time": f"{datetime.timedelta(seconds=float(self.total_time))}",
            "number events": str(self.num_events),
            "number istopes": str(len(self.isotopes)),
        }

    def isTOF(self) -> bool:
        """Returns True is file is 'suspected' of being from a ToF."""
        return self.format in ["nu", "tofwerk"]


class SPCalTextDataFile(SPCalDataFile):
    """Import and access to data stored as a text file, such as a CSV.

    Create using the `SPCalTextDataFile.load` function."""

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
    ):
        super().__init__("text", path, times=times)

        if signals.dtype.names is None:  # pragma: no cover
            raise ValueError("expected `signals` to have a structured dtype")
        for name in isotope_table.values():
            if name not in signals.dtype.names:  # pragma: no cover
                raise ValueError(
                    f"`isotope_table` '{name}' not found in `signals` array"
                )

        self._signals = signals
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

    @property
    def masses(self) -> np.ndarray:
        return np.array([iso.mass for iso in self.isotope_table.keys()])

    @property
    def signals(self) -> np.ndarray:
        return rfn.structured_to_unstructured(self._signals, dtype=np.float32)

    def dataForIsotope(self, isotope: SPCalIsotope) -> np.ndarray:
        return self._signals[self.isotope_table[isotope]]

    def information(self) -> dict[str, str]:
        info = super().information()
        info.update(
            {
                "delimiter": self.delimiter,
                "skip rows": str(self.skip_row),
                "intensity units": "CPS" if self.cps else "Counts",
                "original isotope names": ",".join(self.isotope_table.values()),
            }
        )

        if self.drop_fields is not None:
            info["drop fields"] = ",".join(self.drop_fields)

        return info

    def isTOF(self) -> bool:
        return len(self.isotope_table) > 1

    @classmethod
    def load(
        cls,
        path: Path | str,
        isotope_table: dict[SPCalIsotope, str] | None = None,
        delimiter: str = ",",
        skip_rows: int = 1,
        cps: bool = False,
        drop_fields: list[str] | None = None,
        override_event_time: float | None = None,
    ) -> "SPCalTextDataFile":
        """Imports single particle data stored in a text file.

        If a column with 'time' in the header is present then the 'event_time' can be
        automatically determined, otherwise `override_event_time` is required.
        If more than one isotopes are imported, the file is assumed to be from a ToF.

        Args:
            path: path to the file
            isotope_table: dictionary mmaping an SPCalIsotope to a column name.
                Note that this uses the name post import via `np.genfromtxt`.
            delimiter: the text / column delimiter
            skip_rows: number of rows to skip, including the header
            cps: convert counts-per-second into counts, required if file is CPS
            drop_fields: names in the file to skip.
                By default names containing 'index' or 'time' are skipped.
            override_event_time: override any automatically determined event-time

        Returns:
            data file

        Raises:
            ValueError is event time cannot be read and is not provided
        """
        if isinstance(path, str):
            path = Path(path)

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
                signals[name] *= np.diff(
                    times, append=times[-1] + (times[-1] - times[-2])
                )

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
        )


class SPCalNuDataFile(SPCalDataFile):
    """Data file for data from a Nu Instruments Vitesse.

    Create using the `SPCalNuDataFile.load` function.

    Attributes:
        isotope_table: dict of  isotopes to their index in signals
    """

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
        autoblanking: str = "regions",
        max_mass_diff: float = 0.05,
    ):
        super().__init__("nu", path, times)

        self.info = info

        self._signals = signals
        self._masses = masses

        self.cycle_number = cycle_number
        self.segment_number = segment_number
        self.max_mass_diff = max_mass_diff
        self.autoblanking = autoblanking
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

    @property
    def masses(self) -> np.ndarray:
        return self._masses

    @property
    def signals(self) -> np.ndarray:
        return self._signals

    def dataForIsotope(self, isotope: SPCalIsotope) -> np.ndarray:
        idx = self.isotope_table[isotope]
        return self.signals[:, idx]

    def generateIsotopeTable(self):
        """Populates `isotope_table` with all available isotopes.

        Multiple isotopes may map to the same index in signals if they are isobars."""
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

    def information(self) -> dict[str, str]:
        info = super().information()
        info.update(
            {
                "cycle number": str(self.cycle_number or "All"),
                "segment number": str(self.segment_number or "All"),
                "max mass diff": f"{self.max_mass_diff:.4f}",
                "integ files used": f"{self.integ_files[0]} to {self.integ_files[1] or 'end'}",
                "autoblanking mode": self.autoblanking,
            }
        )
        for key in [
            "AnalysisDateTime",
            "SampleName",
            "MethodFile",
            "CyclesWritten",
            "Username",
            "AverageSingleIonArea",
        ]:
            if key in self.info:
                info[f"run.info :: {key}"] = self.info[key]

        return info

    @classmethod
    def load(
        cls,
        path: Path | str,
        max_mass_diff: float = 0.05,
        cycle_number: int | None = None,
        segment_number: int | None = None,
        first_integ_file: int = 0,
        last_integ_file: int | None = None,
        autoblank: str = "regions",
    ) -> "SPCalNuDataFile":
        """Import Nu Instruments Vitesse data.

        Args:
            path: Path to the 'run.info' file or the containing directory.
            max_mass_diff: maximum difference (in Da) used for match m/z value to isotopes
            cycle_number: limit import to a cycle, default to all
            segment_number: limit import to a segment, default to all
            first_integ_file: import .integ files starting from this number
            last_integ_file: import .integ files ending at this number, None for all
            autoblank: apply autoblanking to overrange regions or to all masses,
                one of 'off', 'regions', 'all'
        """
        if isinstance(path, str):
            path = Path(path)

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
    """Data file for TOFWERK style data.

    Create using the `SPCalTOFWERKDataFile.load` function.

    Attributes:
        isotope_table: dict of isotopes to signal indicies
    """

    def __init__(
        self,
        path: Path,
        signals: np.ndarray,
        times: np.ndarray,
        peak_table: np.ndarray,
        attrs: dict[str, str] | None = None,
    ):
        super().__init__("tofwerk", path, times)

        self._signals = signals
        self.times = times
        self.peak_table = peak_table

        self.attrs = attrs

        self._event_time: float | None = None

        self.isotope_table: dict[SPCalIsotope, int] = {}

        self.generateIsotopeTable()

    def generateIsotopeTable(self):
        re_isotope = re.compile("\\[(\\d+)([A-Z][a-z]?)\\]+")
        for i, peak in enumerate(self.peak_table):
            m = re_isotope.match(peak["label"].decode())
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

    @property
    def signals(self) -> np.ndarray:
        return self._signals

    def dataForIsotope(self, isotope: SPCalIsotope) -> np.ndarray:
        idx = self.isotope_table[isotope]
        return self.signals[:, idx]

    def information(self) -> dict[str, str]:
        info = super().information()
        for attr in ["TofDAQ Version", "Configuration File", "HDF5 File Creation Time"]:
            if self.attrs is not None and attr in self.attrs:
                info[f"HDF5 :: {attr}"] = self.attrs[attr]

        return info

    @classmethod
    def load(
        cls, path: Path | str, max_size: int | None = None
    ) -> "SPCalTOFWERKDataFile":
        """Import a TOFWERK .h5 file.

        Args:
            path: Path to the .h5
            max_size: maximum number of events to read

        Returns:
            data file
        """
        if isinstance(path, str):
            path = Path(path)

        with h5py.File(path) as h5:
            if "PeakData" in h5["PeakData"]:
                peak_data: np.ndarray = h5["PeakData"]["PeakData"][:max_size]
            elif "ToFData" in h5["FullSpectra"]:
                logger.warning(  # pragma: no cover
                    f"PeakData missing from TOFWERK file {path.stem}, integrating"
                )
                peak_data = tofwerk.integrate_tof_data(h5)[:max_size]
            else:  # pragma: no cover
                raise ValueError(
                    f"PeakData and ToFData are missing, {path.stem} is an invalid file"
                )

            peak_data = np.reshape(peak_data, (-1, peak_data.shape[-1]))

            peak_table: np.ndarray = h5["PeakData"]["PeakTable"][:]

            time_per_buf: float = h5["TimingData"].attrs["BlockPeriod"][0]
            times: np.ndarray = h5["TimingData"]["BufTimes"][:max_size]
            times = (
                times[:, :, None]
                + np.linspace(
                    0.0, time_per_buf * 1e-9, h5.attrs["NbrSegments"][0], endpoint=False
                )[None, None, :]
            ).ravel()

            attrs = {
                key: val.decode() if isinstance(val, bytes) else str(val)
                for key, val in h5.attrs.items()
            }

        return cls(
            path, signals=peak_data, times=times, peak_table=peak_table, attrs=attrs
        )
