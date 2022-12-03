# import argparse
from pathlib import Path

import numpy as np

from spcal import detection, poisson

# parser = argparse.ArgumentParser()
# parser.add_argument("csv")
# parser.add_argument("--usecols", nargs="+", type=int)
# parser.add_argument("--names", nargs="+")


# args = parser.parse_args()

colkeys = {
    "Fe": 1,
    "Fe57": 2,
    "Ni": 3,
    "Co": 4,
    "Ni60": 5,
    "Zn": 7,
    "Ag": 8,
    "Ag109": 9,
    "Au": 10,
}

csvs = Path("csvs").glob("*.csv")

datas = {}

for csv in csvs:
    names = []
    cols = []
    for key, col in colkeys.items():
        if key in csv.stem:
            cols.append(col)
            names.append(key)

    print(csv.stem, cols, names)
    data = np.genfromtxt(csv, delimiter=",", skip_header=1, usecols=cols, names=names)

    detections = {}
    labels = {}
    regions = {}
    for name in data.dtype.names:
        ub = np.mean(data[name])
        yc, yd = poisson.formula_c(ub)
        detections[name], labels[name], regions[name] = detection.accumulate_detections(
            data[name], yc + ub, yd + ub
        )

    d, l, r = detection.combine_detections(detections, labels, regions)
    datas[csv.stem] = d

np.savez_compressed("fractions.npz", **datas)
