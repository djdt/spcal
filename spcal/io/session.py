"""Save and restore SPCal sessions."""


from pathlib import Path
from typing import List

import h5py
import numpy as np

from spcal import __version__
from spcal.gui.dialogs.calculator import CalculatorDialog
from spcal.gui.inputs import InputWidget, ReferenceWidget, SampleWidget
from spcal.gui.options import OptionsWidget
from spcal.gui.results import ResultsWidget
from spcal.result import Filter


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


def sanitiseFilters(filters: List[List[Filter]]) -> np.ndarray:
    dtype = np.dtype(
        [
            ("name", "S64"),
            ("unit", "S64"),
            ("operation", "S2"),
            ("value", float),
            ("id", int),
        ]
    )

    size = sum(len(f) for f in filters)
    data = np.empty(size, dtype=dtype)
    i = 0
    for group in filters:
        for id, filter in enumerate(group):
            data[i] = (filter.name, filter.unit, filter.operation, filter.value, id)
            i += 1
    return data


def restoreFilters(data: np.ndarray) -> List[List[Filter]]:
    filters: List[List[Filter]] = []
    group: List[Filter] = []
    for x in data:
        if x["id"] == 0:
            if len(group) > 0:
                filters.append(group)
            group = []
        group.append(
            Filter(
                x["name"].decode(),
                x["unit"].decode(),
                x["operation"].decode(),
                x["value"],
            )
        )
    if len(group) > 0:
        filters.append(group)

    return filters


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
    path: Path,
    options: OptionsWidget,
    sample: SampleWidget,
    reference: ReferenceWidget,
    results: ResultsWidget,
) -> None:
    with h5py.File(path, "w") as h5:
        h5.attrs["version"] = __version__
        options_group = h5.create_group("options")
        for key, val in options.state().items():
            options_group.attrs[key] = val

        expressions_group = h5.create_group("expressions")
        for key, val in CalculatorDialog.current_expressions.items():
            expressions_group.attrs[key] = val

        h5.create_dataset("filters", data=sanitiseFilters(results.filters))

        input: InputWidget
        for input_key, input in zip(["sample", "reference"], [sample, reference]):
            if input.responses.dtype.names is not None:
                input_group = h5.create_group(input_key)
                dset = input_group.create_dataset(
                    "data", data=input.responses, compression="gzip"
                )
                dset.attrs["trim"] = input.trimRegion("")

                import_group = input_group.create_group("import options")
                for key, val in sanitiseImportOptions(input.import_options).items():
                    import_group.attrs[key] = val

                element_group = input_group.create_group("elements")
                for name in input.responses.dtype.names:
                    name_group = element_group.create_group(name)
                    for key, val in input.io[name].state().items():
                        name_group.attrs[key] = val


def restoreSession(
    path: Path,
    options: OptionsWidget,
    sample: SampleWidget,
    reference: ReferenceWidget,
    results: ResultsWidget,
) -> None:
    with h5py.File(path, "r") as h5:
        if tuple(int(x) for x in h5.attrs["version"].split(".")) < (0, 9, 11):
            raise ValueError("Unsupported version.")

        options.setState(h5["options"].attrs)
        for key, val in h5["expressions"].attrs.items():
            CalculatorDialog.current_expressions[key] = val

        input: InputWidget
        for key, input in zip(["sample", "reference"], [sample, reference]):
            if key in h5:
                data = h5[key]["data"][:]
                import_options = restoreImportOptions(h5[key]["import options"].attrs)
                input.loadData(data, import_options)
                input.graph.region.setRegion(h5[key]["data"].attrs["trim"])
                for name in h5[key]["elements"].keys():
                    input.io[name].setState(h5[key]["elements"][name].attrs)

        results.setFilters(restoreFilters(h5["filters"]))
