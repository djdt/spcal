"""Agilent batch binary data import"""
# Copyright 2026 Thomas Lockwood
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path
from xml.etree import ElementTree

import numpy as np


def binary_read_datafile(path: Path) -> np.ndarray:
    msscan = read_msscan(path.joinpath("AcqData", "MSScan.bin"))
    msprofile = read_msprofile(path.joinpath("AcqData", "MSProfile.bin"), len(masses))
    offsets = (
        msscan["SpectrumParamValues"]["SpectrumOffset"]
        // msscan["SpectrumParamValues"]["ByteCount"]
    )
    dtype = [(str(mass), np.float64) for mass in masses] + [("Time", np.float64)]
    data = np.empty(offsets.size, dtype=dtype)
    for mass in masses:
        size = np.minimum(
            (offsets * len(masses)) + (mass.id - 1), msprofile.shape[0] - 1
        )
        data[str(mass)] = msprofile[size]["Analog"]

    data["Time"] = msscan["ScanTime"] * 60.0  # ScanTime in minutes
    return data


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


# def read_msscan_xspecific(path: Path) -> np.ndarray:  # pragma: no cover
#     msscan_xspecific_magic_number = 275
#     msscan_xspecific_header_size = 68
#     msscan_xspecific_dtype = np.dtype([("_", np.int32), ("MZ", np.float64)])
#
#     with path.open("rb") as fp:
#         if int.from_bytes(fp.read(4), "little") != msscan_xspecific_magic_number:
#             raise OSError("Invalid header for MSScan.")
#         fp.seek(msscan_xspecific_header_size)
#         return np.frombuffer(fp.read(), dtype=msscan_xspecific_dtype)


def read_msprofile(path: Path, n: int) -> np.ndarray:
    msprofile_magic_number = 258
    msprofile_header_size = 68

    def get_msprofile_dtype(n: int):
        return np.dtype(
            [
                ("ID", np.float32, n),
                ("Analog", np.float64, n),
                ("Analog2", np.float64, n),
                ("Digital", np.float64, n),
            ]
        )

    with path.open("rb") as fp:
        if (
            int.from_bytes(fp.read(4), "little") != msprofile_magic_number
        ):  # pragma: no cover
            raise OSError("Invalid header for MSProfile.")
        fp.seek(msprofile_header_size)
        data = np.frombuffer(fp.read(), dtype=get_msprofile_dtype(n))

    flattened = np.empty(np.prod(data.size * n), dtype=get_msprofile_dtype(1))
    for name in data.dtype.names:
        flattened[name] = data[name].flat
    return flattened


# def mass_info_datafile(path: Path) -> list[XSpecificMass]:
#     msts_xspecific_path = path.joinpath("AcqData", "MSTS_XSpecific.xml")
#     msts_xaddition_path = path.joinpath("MSTS_XAddition.xml")
#
#     if msts_xspecific_path.exists():
#         xspecific = msts_xspecific_xml_read_info(msts_xspecific_path)
#     else:  # pragma: no cover
#         raise FileNotFoundError("MSTS_XSpecific.xml not found.")
#
#     masses = {
#         idx: XSpecificMass(idx, name=name, acctime=acctime, mz=mz)
#         for idx, (name, mz, acctime) in xspecific.items()
#     }
#
#     if msts_xaddition_path.exists():
#         xaddition, scan_type = msts_xaddition_xml_read_info(msts_xaddition_path)
#         for idx, (mz, mz2) in xaddition.items():
#             masses[idx].mz = mz
#             if scan_type == "MS_MS":
#                 masses[idx].mz2 = mz2
#
#     return sorted(masses.values(), key=lambda x: x.id)
#
#
def read_msts_xspecific_xml(path: Path) -> np.ndarray:
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


def read_msts_xaddition_xml(path: Path) -> np.ndarray:
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


def datafile_mass_info(path: Path) -> np.ndarray:
    msts_xspecific_path = path.joinpath("AcqData", "MSTS_XSpecific.xml")
    msts_xaddition_path = path.joinpath("MSTS_XAddition.xml")

    if msts_xspecific_path.exists():
        masses = read_msts_xspecific_xml(msts_xspecific_path)
    else:  # pragma: no cover
        raise FileNotFoundError("MSTS_XSpecific.xml not found.")

    if msts_xaddition_path.exists():
        xaddition = read_msts_xaddition_xml(msts_xaddition_path)
        for record in xaddition:
            masses["mz"][masses["segment"] == record["segment"]] = record["mz"]
            masses["mz2"][masses["segment"] == record["segment"]] = record["mz2"]

    return masses


if __name__ == "__main__":
    mscan = datafile_mass_info(
        Path("/mnt/storage/ICPMS/20261123_diamonds_sc.b/100ppmc.d")
    )
    print(mscan)
    print(mscan.dtype)
