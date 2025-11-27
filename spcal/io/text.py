"""Reading single particle data from csv files."""

import re
import numpy.lib.recfunctions as rfn
import datetime
import logging
from pathlib import Path

import numpy as np


logger = logging.getLogger(__name__)


def is_text_file(path: Path) -> bool:
    """Checks path exists and is a '.csv', '.txt' or '.text'."""
    if path.suffix.lower() not in [".csv", ".txt", ".text"]:
        return False
    if path.is_dir() or not path.exists():
        return False
    return True


def guess_text_parameters(lines: list[str]) -> tuple[str, int, int]:
    """Guesses the delimiter, skip_rows and column count.

    Args:
        lines: list of lines in file or header

    Returns:
        delimiter, skip_rows, column_count
    """

    def is_number_or_time(x: str) -> bool:
        try:
            float(x)
            return True
        except ValueError:
            pass
        try:
            datetime.time.fromisoformat(x)
            return True
        except ValueError:
            return False

    skip_rows = 0

    delimiter = ""
    for line in lines:
        try:
            delimiter = next(d for d in ["\t", ";", ",", " "] if d in line)
            tokens = line.split(delimiter)
            if all(is_number_or_time(token) for token in tokens):
                break
        except StopIteration:  # special case where only one column exists
            if is_number_or_time(line):
                break
        skip_rows += 1

    column_count = 1
    if delimiter != "":
        column_count = max([line.count(delimiter) for line in lines[skip_rows:]]) + 1

    return delimiter, skip_rows, column_count


def replace_comma_decimal(fp, delimiter: str = ",", count: int = 0):
    for line in fp:
        if delimiter != ",":
            yield line.replace(",", ".")
        else:
            yield line


def iso_time_to_float_seconds(text: str) -> float:
    time = datetime.time.fromisoformat(text)
    return (
        time.hour * 3600.0 + time.minute * 60.0 + time.second + time.microsecond * 1e-6
    )


def read_single_particle_file(
    path: Path | str,
    delimiter: str = ",",
    columns: tuple[int, ...] | np.ndarray | None = None,
    skip_rows: int = 1,
    convert_cps: float | None = None,
    max_rows: int | None = None,
) -> np.ndarray:
    """Imports data stored as text with elements in columns.

    Args:
        path: path to file
        delimiter: delimiting character between columns
        columns: which columns to import, deafults to all
        first_line: the first data (not header) line
        convert_cps: the dwelltime (in s) if data is stored as counts per second,
        else None

    Returns:
        data, structred array
    """
    with Path(path).open("r") as fp:
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

    return signals
