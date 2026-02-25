import numpy as np
from typing import Any
from pathlib import Path

import json
from importlib.metadata import version

from spcal.datafile import SPCalDataFile, SPCalNuDataFile, SPCalTextDataFile
from spcal.isotope import SPCalIsotope, SPCalIsotopeExpression
from spcal.processing.filter import (
    SPCalClusterFilter,
    SPCalValueFilter,
)
from spcal.processing.method import SPCalProcessingMethod
from spcal.processing.options import SPCalIsotopeOptions


class SPCalJSONEncoder(json.JSONEncoder):
    def default(self, o: Any):
        if isinstance(o, Path):
            return str(o.absolute())
        if isinstance(o, SPCalIsotope):
            return str(o)
        if isinstance(o, SPCalIsotopeExpression):
            return {"name": o.name, "tokens": o.tokens}
        if isinstance(o, np.ndarray):
            return o.tolist()
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
        if isinstance(o, SPCalDataFile):
            d = {
                "path": str(o.path.absolute()),
                "format": o.format,
                "selected isotopes": o.selected_isotopes,
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

        return super().default(o)


def save_session_json(
    path: Path, method: SPCalProcessingMethod, data_files: list[SPCalDataFile]
):
    def default(obj: object):
        if isinstance(obj, np.ndarray):
            return obj.tolist()

    output = {
        "method": {
            "instrument options": {
                "uptake": method.instrument_options.uptake,
                "efficiency": method.instrument_options.efficiency,
            },
            "isotope options": {
                str(k): {
                    "density": v.density,
                    "response": v.response,
                    "mass fraction": v.mass_fraction,
                    "concentraiton": v.concentration,
                    "diameter": v.diameter,
                    "mass response": v.mass_response,
                }
                for k, v in method.isotope_options.items()
            },
            "limit options": {
                "method": method.limit_options.limit_method,
                "max_iterations": method.limit_options.max_iterations,
                "window_size": method.limit_options.window_size,
                "default manual limit": method.limit_options.default_manual_limit,
                "gaussian": method.limit_options.gaussian_kws,
                "poisson": method.limit_options.poisson_kws,
                "compound poisson": method.limit_options.compound_poisson_kws,
                "manual limits": {
                    str(k): v for k, v in method.limit_options.manual_limits.items()
                },
                "single ion": method.limit_options.single_ion_parameters,
            },
            "processing options": {
                "accumulation method": method.accumulation_method,
                "calibration mode": method.calibration_mode,
                "cluster distance": method.cluster_distance,
                "points required": method.points_required,
                "prominence required": method.prominence_required,
            },
            "exclusion regions": method.exclusion_regions,
            "result filters": method.result_filters,
            "index filters": method.index_filters,
            "expressions": method.expressions,
        },
        "datafiles": files,
    }
    fp = path.open("w")
    json.dump(output, fp, cls=SPCalJSONEncoder, indent=4)


def save_method(
    path: Path, method: SPCalProcessingMethod, data_files: list[SPCalDataFile]
):
    with path.open("w") as fp:
        fp.write(f"# SPCal {version('spcal')} method\n")

        if (
            method.instrument_options.uptake is not None
            or method.instrument_options.efficiency is not None
        ):
            fp.write("[instrument]\n")
            write_if_not_none(
                fp, "uptake", method.instrument_options.uptake, comment="L/s"
            )
            write_if_not_none(fp, "efficiency", method.instrument_options.efficiency)

        for isotope in method.isotope_options.keys():
            option = method.isotope_options[isotope]
            fp.write(f"[isotope.{_escape(str(isotope))}]\n")
            write_if_not_none(fp, "density", option.density, "kg/m3")
            write_if_not_none(fp, "response", option.response, "cts*L/kg")
            write_if_not_none(fp, "mass_fraction", option.mass_fraction)
            write_if_not_none(fp, "concentration", option.concentration, "kg/L")
            write_if_not_none(fp, "mass_response", option.mass_response, "kg/cts")

        fp.write("[limit]\n")
        opts = method.limit_options
        fp.write(f'method = "{opts.limit_method}"\n')
        fp.write(f"max_iterations = {opts.max_iterations}\n")
        fp.write(f"window_size = {opts.window_size}\n")
        fp.write(f"default_manual_limit = {opts.default_manual_limit}\n")
        write_dict(fp, opts.gaussian_kws, prefix="gaussian = ", suffix="\n")
        write_dict(fp, opts.poisson_kws, prefix="poisson = ", suffix="\n")
        write_dict(fp, opts.compound_poisson_kws, prefix="compound = ", suffix="\n")

        if len(method.limit_options.manual_limits) > 0:
            fp.write("[limit.manual]\n")
            for k, v in method.limit_options.manual_limits.items():
                fp.write(f"{k} = {v}\n")

        fp.write("[processing]\n")
        fp.write(f'calibration_mode = "{method.calibration_mode}"\n')
        fp.write(f"points_required = {method.points_required}\n")
        fp.write(f"prominence_required = {method.prominence_required}\n")
        fp.write(f'accumulation_method = "{method.accumulation_method}"\n')
        fp.write(f"cluster_distance = {method.cluster_distance}\n")

        # [expressions]
        if len(method.expressions) > 0:
            fp.write("[expressions]\n")
            for expr in method.expressions:
                write_iter(fp, expr.tokens, prefix=f"{expr.name} = ", suffix="\n")

        # [filters]
        if (
            sum(len(x) for x in method.result_filters) > 0
            or sum(len(x) for x in method.index_filters) > 0
        ):
            fp.write("[filters]\n")
            if sum(len(x) for x in method.result_filters) > 0:
                fp.write("result = ")
                write_filters(fp, method.result_filters)

            if sum(len(x) for x in method.index_filters):
                fp.write("index = ")
                write_filters(fp, method.index_filters)

        # [exclusions]
        if len(method.exclusion_regions) > 0:
            fp.write("[exclusions]\n")
            write_iter(
                fp,
                (f"[{s}, {e}]" for s, e in method.exclusion_regions),
                prefix="regions = ",
                suffix="\n",
            )

        # [datafiles]
        if len(data_files) > 0:
            fp.write("[datafiles]\n")
            for data_file in data_files:
                write_data_file(fp, data_file)


def load_session_json(
    path: Path
) -> SPCalProcessingMethod:

    # json.load(fp, json.JSONEncoder

    method = SPCalProcessingMethod()

    def isotope_from_str(
        text: str, method: SPCalProcessingMethod
    ) -> SPCalIsotope | SPCalIsotopeExpression:
        try:
            return SPCalIsotope.fromString(key)
        except NameError:
            for expr in method.expressions:
                if expr.name == key:
                    return expr
        raise NameError(f"cannot assign '{key}' to an isotope or expression")

    params = tomllib.load(path.open("rb"))
    if "instrument" in params:
        method.instrument_options.uptake = params["instrument"].get("uptake", None)
        method.instrument_options.uptake = params["instrument"].get("efficiency", None)

    # load expressions first, we need isotope names for later
    if "expressions" in params:
        method.expressions = [
            SPCalIsotopeExpression.fromString(name, text)
            for name, text in params["expressions"].items()
        ]

    if "isotope" in params:
        for key in params["isotope"]:
            isotope_options = SPCalIsotopeOptions(
                params["isotope"][key].get("density", None),
                params["isotope"][key].get("response", None),
                params["isotope"][key].get("mass_fraction", None),
                params["isotope"][key].get("concentration", None),
                params["isotope"][key].get("mass_response", None),
            )
            method.isotope_options[isotope_from_str(key, method)] = isotope_options

    method.limit_options.limit_method = params["limit"]["method"]
    method.limit_options.max_iterations = params["limit"]["max_iterations"]
    method.limit_options.window_size = params["limit"]["window_size"]
    method.limit_options.default_manual_limit = params["limit"]["default_manual_limit"]
    method.limit_options.gaussian_kws = params["limit"]["gaussian"]
    method.limit_options.poisson_kws = params["limit"]["poisson"]
    method.limit_options.compound_poisson_kws = params["limit"]["compound"]

    if "manual" in params["limit"]:
        manual = {}
        for k, v in params["limit"]["manual"].items():
            manual[SPCalIsotope.fromString(k)] = v

    method.points_required = params["processing"]["points_required"]
    method.calibration_mode = params["processing"]["calibration_mode"]
    method.prominence_required = params["processing"]["prominence_required"]
    method.accumulation_method = params["processing"]["accumulation_method"]
    method.cluster_distance = params["processing"]["cluster_distance"]

    # load filters
    if "filters" in params:
        if "result" in params["filters"]:
            method.result_filters = []
            for filter_list in params["filters"]["result"]:
                filters = []
                for filter in filter_list:
                    if filter["type"] == "value":
                        filters.append(
                            SPCalValueFilter(
                                isotope_from_str(filter["isotope"], method),
                                filter["key"],
                                SPCalValueFilter.OPERATION_LABELS[filter["operation"]],
                                filter["value"],
                            )
                        )
                method.result_filters.append(filters)
        if "index" in params["filters"]:
            for filter_list in params["filters"]["index"]:
                filters = []
                for filter in filter_list:
                    if filter["type"] == "cluster":
                        filters.append(
                            SPCalClusterFilter(filter["key"], filter["index"])
                        )
                method.result_filters.append(filters)

    # load exclustion regions
    if "exclusions" in params:
        method.exclusion_regions = [(s, e) for s, e in params["exclusions"]["regions"]]

    return method


if __name__ == "__main__":
    params = np.empty(
        100, dtype=[("mass", np.float32), ("mu", np.float32), ("sigma", np.float32)]
    )
    params["mass"] = np.arange(100)
    params["mu"] = np.random.random(100)
    params["sigma"] = np.random.random(100)

    method = SPCalProcessingMethod()
    method.expressions = [SPCalIsotopeExpression("test_expr", ("+", "107Ag", "197Au"))]
    method.isotope_options[SPCalIsotope.fromString("197Au")] = SPCalIsotopeOptions(
        1.0, 2.0, 3.0
    )
    method.isotope_options[method.expressions[0]] = SPCalIsotopeOptions(1.0, 2.0, 3.0)
    method.limit_options.manual_limits = {SPCalIsotope.fromString("107Ag"): 10.2}
    # method.limit_options.single_ion_parameters = params
    method.result_filters = [
        [
            SPCalValueFilter(
                SPCalIsotope.fromString("107Ag"), "signal", np.greater, 2.0
            ),
            SPCalValueFilter(
                SPCalIsotope.fromString("107Ag"), "signal", np.greater, 3.0
            ),
        ],
        [SPCalValueFilter(SPCalIsotope.fromString("107Ag"), "signal", np.greater, 4.0)],
    ]
    method.index_filters = [[SPCalClusterFilter("signal", 0)]]
    method.exclusion_regions = [(0.4, 1.0), (230.2, 276.0)]

    files = [
        SPCalNuDataFile.load(
            Path("/home/tom/Downloads/Raw data/NT031/15-00-55 1 ppb att")
        ),
        SPCalTextDataFile.load(
            Path("/home/tom/Downloads/019SMPL-48-64Ti_count.csv"),
            skip_rows=5,
            isotope_table={SPCalIsotope.fromString("48Ti"): "Ti48_->_64"},
        ),
    ]
    files[0].selected_isotopes = [SPCalIsotope.fromString("197Au")]

    # save_method(Path("/home/tom/Downloads/test.spcal.toml"), method, files)
    save_session_json(Path("/home/tom/Downloads/test.spcal.json"), method, files)
    # load_method(Path("/home/tom/Downloads/test.spcal.toml"))
