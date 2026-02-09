"""Loader for the databases used by SPCal.

The following databases are stored in a single numpy archive, 'npdb.npz'.
Databases are all structured numpt arrays with the following names and keys.

elements:
    Number: number, e.g. 3
    Symbol: symbol, e.g. Li
    Name: string name, e.g. Lithium
    Mass: molecular wieght in g/mol
    Density: density of pure element in g/cm3
    Ref: source of data

inorganic:
    Formula: formula of inorganic material
    Name: name of material
    CAS: CAS number
    Density: density in h/cm3
    Ref: source of data

polymer:
    Formula: formula of plastic
    Name: common name
    CAS: CAS number
    Density: density in g/mol, polymer densities vary
    Ref: source of data

refs:
    Ref: refrence ID
    Source: name of source
    ID: DOI / ISBN or other ID

"""

from importlib.resources import files

import numpy as np

db = np.load(
    files("spcal.resources").joinpath("npdb.npz").open("rb"), allow_pickle=False
)
