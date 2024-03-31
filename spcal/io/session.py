"""Save and restore SPCal sessions."""

from PySide6 import QtWidgets

from pathlib import Path

import h5py
import numpy as np

from spcal.gui.inputs import InputWidget, ReferenceWidget, SampleWidget
from spcal.gui.options import OptionsWidget
from spcal.gui.results import ResultsWidget
from spcal.result import ClusterFilter, Filter


def flatten_dict(d: dict, prefix: str = "", sep: str = "/") -> dict:
    flat = {}
    for k, v in d.items():
        newk = prefix + sep + k if prefix else k
        if isinstance(v, dict):
            flat.update(flatten_dict(v, newk, sep))
        else:
            flat[newk] = v
    return flat


def unflatten_dict(d: dict, base: dict | None = None, sep: str = "/") -> dict:
    if base is None:
        base = {}
    for k, v in d.items():
        root = base
        if sep in k:
            *tokens, k = k.split("/")
            for token in tokens:
                root.setdefault(token, {})
                root = root[token]
        root[k] = v
    return base


def sanitiseOptions(options: dict) -> dict:
    return flatten_dict(options)


def restoreOptions(options: dict) -> dict:
    return unflatten_dict(options)


def sanitiseFilters(filters: list[list[Filter]]) -> np.ndarray:
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


def restoreFilters(data: np.ndarray) -> list[list[Filter]]:
    filters: list[list[Filter]] = []
    group: list[Filter] = []
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


def sanitiseClusterFilters(filters: list[ClusterFilter]) -> np.ndarray:
    data = np.empty(len(filters), dtype=[("unit", "S64"), ("index", int)])
    data["unit"] = [filter.unit for filter in filters]
    data["index"] = [filter.idx for filter in filters]
    return data


def restoreClusterFilters(data: np.ndarray) -> list[ClusterFilter]:
    filters: list[ClusterFilter] = []
    for x in data:
        filters.append(ClusterFilter(unit=x["unit"].decode(), idx=x["index"]))
    return filters


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
    return flatten_dict(safe)


def restoreImportOptions(options: dict) -> dict:
    restored = {}
    for key, val in unflatten_dict(options).items():
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
        h5.attrs["version"] = QtWidgets.QApplication.applicationVersion()
        options_group = h5.create_group("options")
        for key, val in sanitiseOptions(options.state()).items():
            options_group.attrs[key] = val

        expressions_group = h5.create_group("expressions")
        for key, val in sample.current_expr.items():
            expressions_group.attrs[key] = val

        h5.create_dataset("filters", data=sanitiseFilters(results.filters))
        h5.create_dataset(
            "cluster filters", data=sanitiseClusterFilters(results.cluster_filters)
        )

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
    # Clear old session
    options.resetInputs()
    sample.resetInputs()
    reference.resetInputs()

    with h5py.File(path, "r") as h5:
        if tuple(int(x) for x in h5.attrs["version"].split(".")) < (0, 9, 14):
            raise ValueError("Unsupported version.")  # pragma: no cover

        options.setState(restoreOptions(h5["options"].attrs))
        for key, val in h5["expressions"].attrs.items():
            sample.current_expr[key] = val
            reference.current_expr[key] = val

        input: InputWidget
        for key, input in zip(["sample", "reference"], [sample, reference]):
            if key in h5:
                data = h5[key]["data"][:]
                import_options = restoreImportOptions(h5[key]["import options"].attrs)
                input.loadData(data, import_options)
                input.graph.region.setRegion(h5[key]["data"].attrs["trim"])
                for name in h5[key]["elements"].keys():
                    input.io[name].setState(h5[key]["elements"][name].attrs)

        results.setFilters(
            restoreFilters(h5["filters"]), restoreClusterFilters(h5["cluster filters"])
        )
