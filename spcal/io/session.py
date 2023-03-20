"""Save and restore SPCal sessions."""


from pathlib import Path

import h5py

from spcal import __version__
from spcal.gui.inputs import ReferenceWidget, SampleWidget
from spcal.gui.options import OptionsWidget


def sanitiseImportOptions(options: dict) -> dict:
    safe = {}
    for key, val in options.items():
        if key == "path":  # convert Path to str
            val = str(val)
        elif key == "isotopes":  # hdf5 can't store 'U' arrays
            val = val.astype(
                [(d[0], "S2") if d[0] == "Symbol" else d for d in val.dtype.descr]
            )
        safe[key] = val
    return safe


def restoreImportOptions(options: dict) -> dict:
    restored = {}
    for key, val in options.items():
        if key == "path":  # convert str to Path
            val = Path(val)
        elif key == "isotopes":  # restore 'U' array
            val = val.astype(
                [(d[0], "U2") if d[0] == "Symbol" else d for d in val.dtype.descr]
            )
        restored[key] = val
    return restored


def saveSession(
    path: Path, options: OptionsWidget, sample: SampleWidget, reference: ReferenceWidget
) -> None:
    with h5py.File(path, "w") as h5:
        h5.attrs["version"] = __version__
        options_group = h5.create_group("options")
        for key, val in options.state().items():
            options_group.attrs[key] = val

        if sample.responses.dtype.names is not None:
            h5.create_group("sample", track_order=True)
            dset = h5["sample"].create_dataset(
                "data", data=sample.responses, compression="gzip"
            )
            for key, val in sanitiseImportOptions(sample.import_options).items():
                dset.attrs[key] = val
            for name in sample.responses.dtype.names:
                group = h5["sample"].create_group(name)
                for key, val in sample.io[name].state().items():
                    group.attrs[key] = val

        if reference.responses.dtype.names is not None:
            h5.create_group("reference", track_order=True)
            dset = h5["reference"].create_dataset(
                "data", data=reference.responses, compression="gzip"
            )
            for key, val in sanitiseImportOptions(reference.import_options).items():
                dset.attrs[key] = val
            for name in reference.responses.dtype.names:
                group = h5["reference"].create_group(name)
                for key, val in reference.io[name].state().items():
                    group.attrs[key] = val

        # Results - filters


def restoreSession(
    path: Path, options: OptionsWidget, sample: SampleWidget, reference: ReferenceWidget
) -> None:
    with h5py.File(path, "r") as h5:
        if h5.attrs["version"] < "0.9.9":
            raise ValueError("Unsupported version.")

        options.setState(h5["options"].attrs)

        if "sample" in h5:
            data = h5["sample"]["data"][:]
            import_options = restoreImportOptions(h5["sample"]["data"].attrs)
            sample.loadData(data, import_options)
            for name in h5["sample"].keys():
                if name == "data":
                    continue
                sample.io[name].setState(h5["sample"][name].attrs)

        if "reference" in h5:
            data = h5["reference"]["data"][:]
            import_options = restoreImportOptions(h5["reference"]["data"].attrs)
            reference.loadData(data, import_options)
            for name in h5["reference"].keys():
                if name == "data":
                    continue
                reference.io[name].setState(h5["reference"][name].attrs)

        # Results - filters
