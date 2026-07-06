"""Reading single particle data from csv files."""
# Copyright 2021 Thomas Lockwood
# SPDX-License-Identifier: GPL-3.0-or-later

import datetime
import logging
from pathlib import Path
import warnings
import re

import numpy as np
from numpy.lib._iotools import ConversionWarning


logger = logging.getLogger(__name__)


def is_text_file(path: Path) -> bool:
    """Checks path exists and is a '.csv', '.txt' or '.text'."""
    if path.suffix.lower() not in [".csv", ".txt", ".text"]:
        return False
    if path.is_dir() or not path.exists():
        return False
    return True


def guess_event_time(
    lines: list[str], delimiter: str = ",", skip_rows: int = 1
) -> tuple[float, str | None]:
    """Try to find a column of times and extract the event time.

    Times are extracted as the median difference of a column containing 'time'.

    Args:
        lines: list of delimited lines, with 2 or more columns
        delimiter: the text delimiter
        skip_rows: number of rows to the first data line

    Returns:
        event time in seconds if found, else None

    Raises:
        StopIteration: 'time' column is not found
        ValueError: incorrectly formated input
    """
    re_time = re.compile("[\\(\\[]([nmuµ]?s|sec)[\\]\\)]")

    header = lines[skip_rows - 1].split(delimiter)

    if len(header) < 2:  # pragma: no cover, error
        raise ValueError("header does not have enough columns")

    for col, name in enumerate(header):
        if "time" not in name.strip().lower():
            continue
        m = re_time.search(name.lower())
        if m is not None:
            if m.group(1) in ["s", "sec"]:
                unit = "s"
            elif m.group(1) == "ms":
                unit = "ms"
            elif m.group(1) in ["us", "µs"]:
                unit = "µs"
            elif m.group(1) == "ns":
                unit = "ns"
            else:
                raise ValueError(f"unknown time column unit '{m.group(1)}'")
        else:
            unit = None
            logger.info(f"found a time column '{name}' but no unit")

        texts = [line.split(delimiter)[col] for line in lines[skip_rows:]]
        if len(texts) == 0:  # pragma: no cover, error
            raise ValueError("time column has no entries")

        if "00:" in texts[0]:
            times = [iso_time_to_float_seconds(t) for t in texts]
        else:
            times = [float(t) for t in texts]
        return float(np.median(np.diff(times))), unit
    raise StopIteration


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
    column_count = 1
    delimiter = ""

    for line in lines:
        try:
            delimiter = next(d for d in ["\t", ";", ",", " "] if d in line)
            tokens = line.split(delimiter)
            print(tokens)
            if all(
                is_number_or_time(token) for token in tokens if token not in ["", "\n"]
            ):
                logger.debug(f"all numbers or times at line {skip_rows}")
                break
        except StopIteration:  # special case where only one column exists
            if is_number_or_time(line):
                logger.debug(f"one column, number or time at line {skip_rows}")
                break
        skip_rows += 1

    if delimiter != "":
        try:
            column_count = (
                max([line.count(delimiter) for line in lines[skip_rows:]]) + 1
            )
        except StopIteration:  # pragma: no cover, warning
            logger.warning(f"could not count columns using delimiter '{delimiter}'")
            column_count = 1

    return delimiter, skip_rows, column_count


def replace_comma_decimal(fp, ncols: int, delimiter: str = ","):
    """Yields lines in a text file with the comma replaced with a period.

    If a line has less delimiters than the expected number of columns - 1, then it is skipped.
    If the delimiter is a comma, no action is performed.

    Args:
        fp: file pointer, e.g. from `.open`
        ncols: expected number of columns
        delimiter: column delimiter

    Yields:
        line with comma replaced, when number of columns is correct
    """
    for line in fp:
        if line.count(delimiter) < ncols - 1:
            continue
        if delimiter != ",":
            yield line.replace(",", ".")
        else:
            yield line


def iso_time_to_float_seconds(text: str) -> float:
    """Convert an ISO time string to a float."""
    time = datetime.time.fromisoformat(text)
    return (
        time.hour * 3600.0 + time.minute * 60.0 + time.second + time.microsecond * 1e-6
    )


def read_single_particle_file(
    path: Path | str,
    delimiter: str = ",",
    skip_rows: int = 1,
) -> np.ndarray:
    """Imports data stored as text with elements in columns.

    Args:
        path: path to file
        delimiter: delimiting character between columns
        skip_rows: the first data (not header) line else None

    Returns:
        data, structred array
    """
    with Path(path).open("r") as fp:
        for _ in range(skip_rows - 1):
            fp.readline()

        header = fp.readline().strip().split(delimiter)
        usecols = [i for i, x in enumerate(header) if x != ""]
        header = [x for x in header if x != ""]

        data_start_pos = fp.tell()
        peek = fp.readline()
        if "00:" in peek:  # we are dealing with a thremo iCap export
            converters = {1: lambda s: iso_time_to_float_seconds(s)}
        else:
            converters = {}

        fp.seek(data_start_pos)
        gen = replace_comma_decimal(fp, len(usecols), delimiter)

        # TODO: protential speed-up by trying loadtxt
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", category=ConversionWarning)
            data = np.genfromtxt(  # ty: ignore[no-matching-overload]
                gen,
                delimiter=delimiter,
                converters=converters,
                names=header,
                dtype=np.float32,
                deletechars="",
                invalid_raise=False,
                usecols=usecols,
                loose=True,
            )

    assert data.dtype.names is not None
    return data
