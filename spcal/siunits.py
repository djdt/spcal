"""Multipliers for various SI units."""

density_units = {"g/cm³": 1e-3 * 1e6, "kg/m³": 1.0}

mass_units = {
    "ag": 1e-21,
    "fg": 1e-18,
    "pg": 1e-15,
    "ng": 1e-12,
    "µg": 1e-9,
    "mg": 1e-6,
    "g": 1e-3,
    "kg": 1.0,
}

mass_concentration_units = {
    "fg/L": 1e-18,
    "pg/L": 1e-15,
    "ng/L": 1e-12,
    "µg/L": 1e-9,
    "mg/L": 1e-6,
    "g/L": 1e-3,
    "kg/L": 1.0,
}

molar_concentration_units = {
    "amol/L": 1e-18,
    "fmol/L": 1e-15,
    "pmol/L": 1e-12,
    "nmol/L": 1e-9,
    "µmol/L": 1e-6,
    "mmol/L": 1e-3,
    "mol/L": 1.0,
}

number_concentration_units = {"#/L": 1.0, "#/ml": 1000.0}

response_units = {
    "L/pg": 1e15,
    "L/ng": 1e12,
    "L/µg": 1e9,
    "L/mg": 1e6,
}

signal_units = {"cts": 1.0}

size_units = {"nm": 1e-9, "µm": 1e-6, "mm": 1e-3, "m": 1.0}

time_units = {"ns": 1e-9, "µs": 1e-6, "ms": 1e-3, "s": 1.0}

flowrate_units = {
    "ml/min": 1e-3 / 60.0,
    "ml/s": 1e-3,
    "L/min": 1.0 / 60.0,
    "L/s": 1.0,
}

volume_units = {"nm³": 1e-27, "µm³": 1e-18, "mm³": 1e-9, "m³": 1.0}
