"""Reading single particle data from csv files."""

import datetime
import logging
from pathlib import Path
import warnings

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


def replace_comma_decimal(fp, ncols: int, delimiter: str = ","):
    for line in fp:
        if line.count(delimiter) < ncols - 1:
            continue
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
    skip_rows: int = 1,
    max_rows: int | None = None,
) -> np.ndarray:
    """Imports data stored as text with elements in columns.

    Args:
        path: path to file
        delimiter: delimiting character between columns
        first_line: the first data (not header) line else None

    Returns:
        data, structred array
    """
    with Path(path).open("r") as fp:
        for i in range(skip_rows - 1):
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

        # todo: protential speed-up by trying loadtxt
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", category=ConversionWarning)
            data = np.genfromtxt(
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


if __name__ == "__main__":
    print(
        read_single_particle_file(
            "/home/tom/Downloads/STD1_AuNPs 50 nm_38.csv", skip_rows=1
        )[-10:]
    )
