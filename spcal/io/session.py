from typing import Any, TextIO
import tomllib
from pathlib import Path

from importlib.metadata import version

from spcal.isotope import SPCalIsotope
from spcal.processing.method import SPCalProcessingMethod
from spcal.processing.options import SPCalIsotopeOptions


def save_method(path: Path, method: SPCalProcessingMethod):
    def write_if_not_none(
        fp: TextIO, name: str, value: Any, comment: str | None = None
    ):
        if value is None or value == "":
            return

        if isinstance(value, str):
            value = '"' + value + '"'

        fp.write(f"{name} = {value}")
        if comment is not None:
            fp.write(f"  # {comment}")
        fp.write("\n")

    with path.open("w") as fp:
        fp.write(f"# SPCal {version('spcal')} method\n")

        fp.write("[instrument]\n")
        write_if_not_none(fp, "uptake", method.instrument_options.uptake, comment="L/s")
        write_if_not_none(fp, "efficiency", method.instrument_options.efficiency)

        for isotope in method.isotope_options.keys():
            option = method.isotope_options[isotope]
            fp.write(f"[isotope.{isotope}]\n")
            write_if_not_none(fp, "density", option.density, "kg/m3")
            write_if_not_none(fp, "response", option.response, "cts*L/kg")
            write_if_not_none(fp, "mass_fraction", option.mass_fraction)
            write_if_not_none(fp, "concentration", option.concentration, "kg/L")
            write_if_not_none(fp, "mass_response", option.mass_response, "kg/cts")

        fp.write("[limit]\n")
        fp.write(f'method = "{method.limit_options.limit_method}"\n')
        fp.write(f"max_iterations = {method.limit_options.max_iterations}\n")
        fp.write(f"window_size = {method.limit_options.window_size}\n")
        fp.write(
            f"default_manual_limit = {method.limit_options.default_manual_limit}\n"
        )

        for name, kws in zip(
            ["gaussian", "poisson", "compound"],
            [
                method.limit_options.gaussian_kws,
                method.limit_options.poisson_kws,
                method.limit_options.compound_poisson_kws,
            ],
        ):
            fp.write(f"[limit.{name}]\n")
            for k, v in kws.items():
                write_if_not_none(fp, k, v)

        if len(method.limit_options.manual_limits) > 0:
            fp.write("[limit.manual]\n")
            for k, v in method.limit_options.manual_limits.items():
                fp.write(f"{k} = {v}\n")

        fp.write("[processing]\n")
        fp.write(f"points_required = {method.points_required}\n")
        fp.write(f"prominence_required = {method.prominence_required}\n")
        fp.write(f'accumulation_method = "{method.accumulation_method}"\n')
        fp.write(f"cluster_distance = {method.cluster_distance}\n")


def load_method(
    path: Path, method: SPCalProcessingMethod | None = None
) -> SPCalProcessingMethod:
    if method is None:
        method = SPCalProcessingMethod()

    params = tomllib.load(path.open("rb"))
    method.instrument_options.uptake = params["instrument"].get("uptake", None)
    method.instrument_options.uptake = params["instrument"].get("efficiency", None)

    if "isotope" in params:
        for key in params["isotope"]:
            isotope_options = SPCalIsotopeOptions(
                params["isotope"][key].get("density", None),
                params["isotope"][key].get("response", None),
                params["isotope"][key].get("mass_fraction", None),
                params["isotope"][key].get("concentration", None),
                params["isotope"][key].get("mass_response", None),
            )
            method.isotope_options[SPCalIsotope.fromString(key)] = isotope_options

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
    method.prominence_required = params["processing"]["prominence_required"]
    method.accumulation_method = params["processing"]["accumulation_method"]
    method.cluster_distance = params["processing"]["cluster_distance"]

    return method

if __name__ == "__main__":
    method = SPCalProcessingMethod()
    method.isotope_options[SPCalIsotope.fromString("197Au")] = SPCalIsotopeOptions(
        1.0, 2.0, 3.0
    )
    method.limit_options.manual_limits = {SPCalIsotope.fromString("107Ag"): 10.2}

    save_method(Path("/home/tom/Downloads/test.spcal.toml"), method)
    load_method(Path("/home/tom/Downloads/test.spcal.toml"))
