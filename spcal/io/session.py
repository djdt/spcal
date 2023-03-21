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

        for key, input in zip(["sample", "reference"], [sample, reference]):
            if input.responses.dtype.names is not None:
                h5.create_group(key)
                dset = h5[key].create_dataset(
                    "data", data=input.responses, compression="gzip"
                )
                dset.attrs["trim"] = input.trimRegion("")

                group = h5[key].create_group("import options")
                for key, val in sanitiseImportOptions(input.import_options).items():
                    group.attrs[key] = val

                element_group = h5[key].create_group("elements")
                for name in input.responses.dtype.names:
                    group = element_group.create_group(name)
                    for key, val in input.io[name].state().items():
                        group.attrs[key] = val

        # Results - filters


def restoreSession(
    path: Path, options: OptionsWidget, sample: SampleWidget, reference: ReferenceWidget
) -> None:
    with h5py.File(path, "r") as h5:
        if h5.attrs["version"] < "0.9.9":
            raise ValueError("Unsupported version.")

        options.setState(h5["options"].attrs)

        for key, input in zip(["sample", "reference"], [sample, reference]):
            if key in h5:
                data = h5[key]["data"][:]
                import_options = restoreImportOptions(h5[key]["import options"].attrs)
                input.loadData(data, import_options)
                input.graph.region.setRegion(h5[key]["data"].attrs["trim"])
                for name in h5[key]["elements"].keys():
                    input.io[name].setState(h5[key]["elements"][name].attrs)

        # Results - filters
