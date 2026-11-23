"""Agilent batch binary data import"""
# Copyright 2026 Thomas Lockwood
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path
from xml.etree import ElementTree

import numpy as np


def datafile_mass_info(path: Path) -> np.ndarray:
    """Reads the names and masses from `MSTS_XSpecific.xml`.

    If a MS/MS instrument is used, the masses are updated using `MSTS_XAddition.xml`.

    Args:
        path: path to datafile firectory

    Returns:
        structured array of type (time_segment, name, mz, mz2, acctime)
    """

    msts_xspecific_path = path.joinpath("AcqData", "MSTS_XSpecific.xml")
    msts_xaddition_path = path.joinpath("MSTS_XAddition.xml")

    if msts_xspecific_path.exists():
        masses = read_xml_msts_xspecific(msts_xspecific_path)
    else:  # pragma: no cover
        raise FileNotFoundError("MSTS_XSpecific.xml not found.")

    if msts_xaddition_path.exists():
        xaddition = read_xml_msts_xaddition(msts_xaddition_path)
        for record in xaddition:
            masses["mz"][masses["segment"] == record["segment"]] = record["mz"]
            masses["mz2"][masses["segment"] == record["segment"]] = record["mz2"]

    return masses


def read_datafile(
    path: Path, time_segment: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masses = datafile_mass_info(path)
    msscan = read_msscan(path.joinpath("AcqData", "MSScan.bin"))
    msprofile = read_msprofile(path.joinpath("AcqData", "MSProfile.bin"), len(masses))

    mask = msscan["TimeSegmentID"] == time_segment

    offsets = (msscan["SpectrumParamValues"]["SpectrumOffset"] - 68) // msscan[
        "SpectrumParamValues"
    ]["ByteCount"]
    signals = msprofile[offsets[mask]]["Counts"]
    return signals, masses, msscan["ScanTime"][mask]


# binary import
def read_msscan(path: Path):
    msscan_magic_number = 257
    msscan_header_size = 68
    msscan_dtype = np.dtype(
        [
            ("ScanID", np.int32),
            ("ScanMethodID", np.int32),
            ("TimeSegmentID", np.int32),
            ("ScanTime", np.float64),
            ("MSLevel", np.int32),
            ("ScanType", np.int32),
            ("TIC", np.float64),
            ("BasePeakMZ", np.float64),
            ("BasePeakValue", np.float64),
            ("Status", np.int32),
            ("IonMode", np.int32),
            ("IonPolarity", np.int32),
            ("SamplingPeriod", np.float64),
            (
                "SpectrumParamValues",
                np.dtype(
                    [
                        ("SpectrumFormatID", np.int32),
                        ("SpectrumOffset", np.int64),
                        ("ByteCount", np.int32),
                        ("PointCount", np.int32),
                        ("MinX", np.float64),
                        ("MaxX", np.float64),
                        ("MinY", np.float64),
                        ("MaxY", np.float64),
                    ]
                ),
            ),
            (
                "XSpecificParamType",
                np.dtype([("Offset", np.int64), ("ByteCount", np.int32)]),
            ),
        ]
    )

    with path.open("rb") as fp:
        if (
            int.from_bytes(fp.read(4), "little") != msscan_magic_number
        ):  # pragma: no cover
            raise OSError("Invalid header for MSScan.")
        fp.seek(msscan_header_size + 20)
        offset = int.from_bytes(fp.read(4), "little")
        fp.seek(offset)
        return np.frombuffer(fp.read(), dtype=msscan_dtype)


def read_msscan_xspecific(path: Path) -> np.ndarray:  # pragma: no cover
    msscan_xspecific_magic_number = 275
    msscan_xspecific_header_size = 68
    msscan_xspecific_dtype = np.dtype([("_", np.int32), ("MZ", np.float64)])

    with path.open("rb") as fp:
        if int.from_bytes(fp.read(4), "little") != msscan_xspecific_magic_number:
            raise OSError("Invalid header for MSScan.")
        fp.seek(msscan_xspecific_header_size)
        return np.frombuffer(fp.read(), dtype=msscan_xspecific_dtype)


def read_msprofile(path: Path, n: int) -> np.ndarray:
    msprofile_magic_number = 258
    msprofile_header_size = 68

    def get_msprofile_dtype(n: int):
        return np.dtype(
            [
                ("ID", np.float32, n),
                ("Counts", np.float64, n),
                ("Digital", np.float64, n),
                ("Analog", np.float64, n),
            ]
        )

    with path.open("rb") as fp:
        if (
            int.from_bytes(fp.read(4), "little") != msprofile_magic_number
        ):  # pragma: no cover
            raise OSError("Invalid header for MSProfile.")
        fp.seek(msprofile_header_size)
        return np.frombuffer(fp.read(), dtype=get_msprofile_dtype(n))


# xml import
def read_xml_msts_xspecific(path: Path) -> np.ndarray:
    xspecific_dtype = np.dtype(
        [
            # ("id", np.int32),
            ("segment", np.int32),
            ("name", "U32"),
            ("mz", np.int32),
            ("mz2", np.int32),
            ("acctime", np.float64),
        ]
    )
    xml = ElementTree.parse(path)
    mass_list = []
    for record in xml.iter("IonRecord"):
        segment = int(record.attrib["TimeSegmentID"])
        for mass in record.iter("Masses"):
            mz = int(mass.findtext("Mass", "-1"))
            name = mass.findtext("Name", "")
            acctime = float(mass.findtext("AccumulationTime", np.nan))
            mass_list.append((segment, name, mz, -1, acctime))
    return np.array(mass_list, dtype=xspecific_dtype)


def read_xml_msts_xaddition(path: Path) -> np.ndarray:
    xaddition_dtype = np.dtype(
        [
            # ("id", np.int32),
            ("segment", np.int32),
            ("index", np.int32),
            ("mz", np.int32),
            ("mz2", np.int32),
        ]
    )
    xml = ElementTree.parse(path)
    mass_list = []
    for record in xml.iter("MSTS_XAddition"):
        segment = int(record.attrib["TimeSegmentID"])
        indexed_masses = record.find("IndexedMasses")
        if indexed_masses is None:  # pragma: no cover
            raise ValueError("could not find IndexedMasses")
        for msts_index in indexed_masses.iter("MSTS_XAddition_IndexedMasses"):
            index = int(msts_index.findtext("Index", "-1"))
            mz = int(msts_index.findtext("PrecursorIonMZ", "-1"))
            mz2 = int(msts_index.findtext("ProductIonMZ", "-1"))
            mass_list.append((segment, index, mz, mz2))
    return np.array(mass_list, dtype=xaddition_dtype)


if __name__ == "__main__":
    mscan, masses, times = read_datafile(
        Path("/mnt/storage/ICPMS/20261123_diamonds_sc.b/100nmau.d/")
    )
