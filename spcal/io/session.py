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
from spcal.isotope import SPCalIsotope, SPCalIsotopeExpression
from spcal.processing.filter import (
    SPCalClusterFilter,
    SPCalValueFilter,
)
from spcal.processing.method import SPCalProcessingMethod
from spcal.processing.options import (
    SPCalInstrumentOptions,
    SPCalIsotopeOptions,
    SPCalLimitOptions,
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
        # data files
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
                    "concentration": v.concentration,
                    "diameter": v.diameter,
                    "mass response": v.mass_response,
                }
                for k, v in method.isotope_options.items()
            },
            "limit options": {
                "method": method.limit_options.limit_method,
                "max iterations": method.limit_options.max_iterations,
                "window size": method.limit_options.window_size,
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
            "expressions": {
                expr.name: " ".join(str(x) for x in expr.tokens)
                for expr in method.expressions
            },
        },
        "datafiles": data_files,
    }

    with path.open("w") as fp:
        json.dump(output, fp, cls=SPCalJSONEncoder, indent=4)


def decode_json_method(method_dict: dict) -> SPCalProcessingMethod:
    def decode_isotope(
        text: str, expressions: list[SPCalIsotopeExpression]
    ) -> SPCalIsotope | SPCalIsotopeExpression:
        try:
            return SPCalIsotope.fromString(text)
        except NameError:
            for expr in expressions:
                if expr.name == text:
                    return expr
        raise NameError(f"cannot assign '{text}' to an isotope or expression")

    def decode_single_ion(x: np.ndarray | None):
        if x is None:
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
                else:
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
    )
    limit_options.default_manual_limit = method_dict["limit options"][
        "default manual limit"
    ]
    limit_options.manual_limits = {
        decode_isotope(k, expressions): v
        for k, v in method_dict["limit options"]["manual limits"].items()
    }

    method = SPCalProcessingMethod(
        instrument_options,
        limit_options,
        isotope_options,  # type: ignore
        accumulation_method=method_dict["processing options"]["accumulation method"],
        points_required=method_dict["processing options"]["points required"],
        prominence_required=method_dict["processing options"]["prominence required"],
        calibration_mode=method_dict["processing options"]["calibration mode"],
        cluster_distance=method_dict["processing options"]["cluster distance"],
    )
    method.expressions = expressions
    method.exclusion_regions = method_dict["exclusion regions"]
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
    else:
        raise ValueError(f"unknown data file format '{file_dict['format']}'")

    df.selected_isotopes = [  # type: ignore
        SPCalIsotope.fromString(x) for x in file_dict["selected isotopes"]
    ]
    return df


def load_session_json(path: Path) -> tuple[SPCalProcessingMethod, list[SPCalDataFile]]:
    with path.open() as fp:
        session = json.load(fp)

    method = decode_json_method(session["method"])
    data_files = [decode_json_datafile(df) for df in session["datafiles"]]

    return method, data_files


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
    method.limit_options.single_ion_parameters = params
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
        SPCalNuDataFile.load(Path("/home/tom/Downloads/NT032/14-37-30 1 ppb att")),
        SPCalTextDataFile.load(
            Path("/home/tom/Downloads/019SMPL-48-64Ti_count.csv"),
            skip_rows=5,
            isotope_table={SPCalIsotope.fromString("48Ti"): "Ti48_->_64"},
        ),
    ]
    files[0].selected_isotopes = [SPCalIsotope.fromString("197Au")]

    # save_method(Path("/home/tom/Downloads/test.spcal.toml"), method, files)
    save_session_json(Path("/home/tom/Downloads/test.spcal.json"), method, files)
    load_session_json(Path("/home/tom/Downloads/test.spcal.json"))
