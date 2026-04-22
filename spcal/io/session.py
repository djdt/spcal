"""Load and save SPCal method and datafiles to JSON"""
# Copyright 2023 Thomas Lockwood
# SPDX-License-Identifier: GPL-3.0-or-later

from datetime import datetime
import numpy as np
from typing import Any
from pathlib import Path

import json
from importlib.metadata import version

from spcal.datafile import (
    SPCalDataFile,
    SPCalNuDataFile,
    SPCalTOFWERKDataFile,
    SPCalTextDataFile,
)
from spcal.isotope import SPCalIsotope, SPCalIsotopeExpression, SPCalIsotopeBase
from spcal.processing.filter import (
    SPCalClusterFilter,
    SPCalValueFilter,
)
from spcal.processing.method import SPCalProcessingMethod
from spcal.processing.options import (
    SPCalInstrumentOptions,
    SPCalIsotopeOptions,
    SPCalLimitOptions,
    SPCalProcessingOptions,
)


class SPCalJSONEncoder(json.JSONEncoder):
    def default(self, o: Any):
        # normal types
        if isinstance(o, Path):
            return str(o.absolute())
        if isinstance(o, np.ndarray):
            return o.tolist()
        # isotope
        if isinstance(o, (SPCalIsotope, SPCalIsotopeExpression)):
            return str(o)
        # filter types
        if isinstance(o, SPCalValueFilter):
            return {
                "filter type": "value",
                "isotope": o.isotope,
                "key": o.key,
                "operation": o.opString(),
                "value": o.value,
            }
        if isinstance(o, SPCalClusterFilter):
            return {"filter type": "cluster", "key": o.key, "index": o.index}
        # method
        if isinstance(o, SPCalProcessingMethod):
            return {
                "instrument options": o.instrument_options,
                "isotope options": {str(k): v for k, v in o.isotope_options.items()},
                "limit options": o.limit_options,
                "processing options": o.processing_options,
                "exclusion regions": o.exclusion_regions,
                "result filters": o.result_filters,
                "index filters": o.index_filters,
                "expressions": {
                    expr.name: " ".join(str(x) for x in expr.tokens)
                    for expr in o.expressions
                },
            }
        if isinstance(o, SPCalInstrumentOptions):
            return {"uptake": o.uptake, "efficiency": o.efficiency}
        if isinstance(o, SPCalIsotopeOptions):
            return {
                "density": o.density,
                "response": o.response,
                "mass fraction": o.mass_fraction,
                "concentration": o.concentration,
                "diameter": o.diameter,
                "mass response": o.mass_response,
            }
        if isinstance(o, SPCalLimitOptions):
            return {
                "method": o.limit_method,
                "max iterations": o.max_iterations,
                "window size": o.window_size,
                "default manual limit": o.default_manual_limit,
                "gaussian": o.gaussian_kws,
                "poisson": o.poisson_kws,
                "compound poisson": o.compound_poisson_kws,
                "manual limits": {str(k): v for k, v in o.manual_limits.items()},
                "single ion": o.single_ion_parameters,
            }
        if isinstance(o, SPCalProcessingOptions):
            return {
                "accumulation method": o.accumulation_method,
                "calibration mode": o.calibration_mode,
                "cluster distance": o.cluster_distance,
                "points required": o.points_required,
                "prominence required": o.prominence_required,
            }
        # data files
        if isinstance(o, SPCalDataFile):
            d = {
                "path": o.path,
                "format": o.format,
                "selected isotopes": o.selected_isotopes,
                "exclusion regions": o.exclusion_regions,
            }
            if isinstance(o, SPCalTextDataFile):
                d.update(
                    {
                        "isotope table": {
                            str(k): v for k, v in o.isotope_table.items()
                        },
                        "delimiter": o.delimiter,
                        "skip row": o.skip_row,
                        "cps": o.cps,
                        "override event time": o.override_event_time,
                        "drop fields": o.drop_fields,
                    }
                )
            elif isinstance(o, SPCalNuDataFile):
                d.update(
                    {
                        "max mass diff": o.max_mass_diff,
                        "cycle number": o.cycle_number,
                        "segment number": o.segment_number,
                        "integ files": o.integ_files,
                        "autoblanking": o.autoblanking,
                    }
                )
            return d

        return super().default(o)  # pragma: no cover , default behaviour


def save_session_json(
    path: Path, method: SPCalProcessingMethod, data_files: list[SPCalDataFile]
):
    output = {
        "version": version("spcal"),
        "date": datetime.now().isoformat(timespec="seconds"),
        "method": method,
        "datafiles": data_files,
    }

    with path.open("w") as fp:
        json.dump(output, fp, cls=SPCalJSONEncoder, indent=4)


def decode_json_method(method_dict: dict) -> SPCalProcessingMethod:
    def decode_isotope(
        text: str, expressions: list[SPCalIsotopeExpression]
    ) -> SPCalIsotopeBase:
        try:
            return SPCalIsotope.fromString(text)
        except NameError:
            for expr in expressions:
                if expr.name == text:
                    return expr
        raise NameError(f"cannot assign '{text}' to an isotope or expression")  # pragma: no cover

    def decode_single_ion(x: np.ndarray | None):
        if x is None:  # pragma: no cover
            return None
        x = np.asarray(x)
        params = np.empty(
            x.shape[0], dtype=[("mass", float), ("mu", float), ("sigma", float)]
        )
        params["mass"] = x[:, 0]
        params["mu"] = x[:, 1]
        params["sigma"] = x[:, 2]
        return params

    def decode_filters(
        flist: list[list[dict]], expressions: list[SPCalIsotopeExpression]
    ):
        filters = []
        for filter_list in flist:
            group = []
            for filter in filter_list:
                if filter["filter type"] == "value":
                    group.append(
                        SPCalValueFilter(
                            isotope=decode_isotope(filter["isotope"], expressions),
                            key=filter["key"],
                            operation=SPCalValueFilter.OPERATION_LABELS[
                                filter["operation"]
                            ],
                            value=filter["value"],
                        )
                    )
                elif filter["filter type"] == "cluster":
                    group.append(
                        SPCalClusterFilter(key=filter["key"], index=filter["index"])
                    )
                else:  # pragma: no cover
                    raise ValueError(f"unknown filter type {filter['filter type']}")
            filters.append(group)
        return filters

    expressions = [
        SPCalIsotopeExpression.fromString(k, v)
        for k, v in method_dict["expressions"].items()
    ]

    instrument_options = SPCalInstrumentOptions(
        uptake=method_dict["instrument options"]["uptake"],
        efficiency=method_dict["instrument options"]["efficiency"],
    )
    isotope_options = {
        decode_isotope(k, expressions): SPCalIsotopeOptions(
            v["density"],
            v["response"],
            v["mass fraction"],
            v["concentration"],
            v["diameter"],
            v["mass response"],
        )
        for k, v in method_dict["isotope options"].items()
    }
    limit_options = SPCalLimitOptions(
        method_dict["limit options"]["method"],
        gaussian_kws=method_dict["limit options"]["gaussian"],
        poisson_kws=method_dict["limit options"]["poisson"],
        compound_poisson_kws=method_dict["limit options"]["compound poisson"],
        max_iterations=method_dict["limit options"]["max iterations"],
        window_size=method_dict["limit options"]["window size"],
        single_ion_parameters=decode_single_ion(
            method_dict["limit options"]["single ion"]
        ),
        default_manual_limit=method_dict["limit options"]["default manual limit"],
        manual_limits={
            decode_isotope(k, expressions): v
            for k, v in method_dict["limit options"]["manual limits"].items()
        },
    )

    processing_options = SPCalProcessingOptions(
        accumulation_method=method_dict["processing options"]["accumulation method"],
        points_required=method_dict["processing options"]["points required"],
        prominence_required=method_dict["processing options"]["prominence required"],
        calibration_mode=method_dict["processing options"]["calibration mode"],
        cluster_distance=method_dict["processing options"]["cluster distance"],
    )

    method = SPCalProcessingMethod(
        instrument_options, limit_options, isotope_options, processing_options
    )
    method.expressions = expressions
    method.exclusion_regions = [(s, e) for s, e in method_dict["exclusion regions"]]
    method.result_filters = decode_filters(method_dict["result filters"], expressions)
    method.index_filters = decode_filters(method_dict["index filters"], expressions)
    return method


def decode_json_datafile(file_dict: dict, path: Path | None = None) -> SPCalDataFile:
    if path is None:
        path = Path(file_dict["path"])
    if file_dict["format"] == "text":
        isotope_table = {
            SPCalIsotope.fromString(k): v for k, v in file_dict["isotope table"].items()
        }
        df = SPCalTextDataFile.load(
            path,
            isotope_table,
            delimiter=file_dict["delimiter"],
            skip_rows=file_dict["skip row"],
            cps=file_dict["cps"],
            override_event_time=file_dict["override event time"],
            drop_fields=file_dict["drop fields"],
        )
    elif file_dict["format"] == "nu":
        df = SPCalNuDataFile.load(
            path,
            max_mass_diff=file_dict["max mass diff"],
            cycle_number=file_dict["cycle number"],
            segment_number=file_dict["segment number"],
            first_integ_file=file_dict["integ files"][0],
            last_integ_file=file_dict["integ files"][1],
            autoblank=file_dict["autoblanking"],
        )
    elif file_dict["format"] == "tofwerk":
        df = SPCalTOFWERKDataFile.load(path)
    else:  # pragma: no cover
        raise ValueError(f"unknown data file format '{file_dict['format']}'")

    df.selected_isotopes = [
        SPCalIsotope.fromString(x) for x in file_dict["selected isotopes"]
    ]
    if "exclusion regions" in file_dict:  # added in v2.0.9
        df.exclusion_regions = [(s, e) for s, e in file_dict["exclusion regions"]]
    return df


def load_session_json(path: Path) -> tuple[SPCalProcessingMethod, list[SPCalDataFile]]:
    with path.open() as fp:
        session = json.load(fp)

    method = decode_json_method(session["method"])
    data_files = [decode_json_datafile(df) for df in session["datafiles"]]

    return method, data_files
