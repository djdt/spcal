from PySide6 import QtGui

color_schemes = {
    "IBM Carbon": [
        QtGui.QColor("#6929c4"),  # Purple 70
        QtGui.QColor("#1192e8"),  # Cyan 50
        QtGui.QColor("#005d5d"),  # Teal 70
        QtGui.QColor("#9f1853"),  # Magenta 70
        QtGui.QColor("#fa4d56"),  # Red 50
        QtGui.QColor("#570408"),  # Red 90
        QtGui.QColor("#198038"),  # Green 60
        QtGui.QColor("#002d9c"),  # Blue 80
        QtGui.QColor("#ee538b"),  # Magenta 50
        QtGui.QColor("#b26800"),  # Yellow 50
        QtGui.QColor("#009d9a"),  # Teal 50
        QtGui.QColor("#012749"),  # Cyan 90
        QtGui.QColor("#8a3800"),  # Orange 70
        QtGui.QColor("#a56eff"),  # Purple 50
    ],
    "Base16": [  # ordered for clarity
        QtGui.QColor("#ab4642"),  # 08 red
        QtGui.QColor("#7cafc2"),  # 0d blue
        QtGui.QColor("#dc9656"),  # 09 orange
        QtGui.QColor("#a1b56c"),  # 0b green
        QtGui.QColor("#f7ca88"),  # 0a yellow
        QtGui.QColor("#ba8baf"),  # 0e magenta
        QtGui.QColor("#86c1b9"),  # 0c teal
        QtGui.QColor("#a16946"),  # 0f brown
        QtGui.QColor("#b8b8b8"),  # 03 grey
    ],
    "ColorBrewer Set1": [
        QtGui.QColor("#e41a1c"),
        QtGui.QColor("#377eb8"),
        QtGui.QColor("#4daf4a"),
        QtGui.QColor("#984ea3"),
        QtGui.QColor("#ff7f00"),
        QtGui.QColor("#ffff33"),
        QtGui.QColor("#a65628"),
        QtGui.QColor("#f781bf"),
    ],
    "Tableau 10": [
        QtGui.QColor("#1f77b4"),
        QtGui.QColor("#ff7f0e"),
        QtGui.QColor("#2ca02c"),
        QtGui.QColor("#d62728"),
        QtGui.QColor("#9467bd"),
        QtGui.QColor("#8c564b"),
        QtGui.QColor("#e377c2"),
        QtGui.QColor("#7f7f7f"),
        QtGui.QColor("#bcbd22"),
        QtGui.QColor("#17becf"),
    ],
    "Tol Bright": [  # https://cran.r-project.org/web/packages/khroma/vignettes/tol.html
        QtGui.QColor("#4477aa"),
        QtGui.QColor("#ee6677"),
        QtGui.QColor("#228833"),
        QtGui.QColor("#ccbb44"),
        QtGui.QColor("#66ccee"),
        QtGui.QColor("#aa3377"),
        QtGui.QColor("#bbbbbb"),
    ],
    "Okabe Ito": [  # https://jfly.uni-koeln.de/color/
        QtGui.QColor(0, 0, 0),
        QtGui.QColor(230, 159, 0),
        QtGui.QColor(86, 180, 233),
        QtGui.QColor(0, 158, 115),
        QtGui.QColor(240, 228, 66),
        QtGui.QColor(0, 114, 178),
        QtGui.QColor(213, 94, 0),
        QtGui.QColor(204, 121, 167),
    ],
}

# https://github.com/sjmgarnier/viridisLite/blob/master/data-raw/viridis_map.csv
viridis_32 = [
    QtGui.QColor.fromRgbF(0.27602238, 0.04416723, 0.37016418),
    QtGui.QColor.fromRgbF(0.28192358, 0.08966622, 0.41241521),
    QtGui.QColor.fromRgbF(0.283072, 0.13089477, 0.44924127),
    QtGui.QColor.fromRgbF(0.27957399, 0.17059884, 0.47999675),
    QtGui.QColor.fromRgbF(0.27182812, 0.20930306, 0.50443413),
    QtGui.QColor.fromRgbF(0.26057103, 0.2469217, 0.52282822),
    QtGui.QColor.fromRgbF(0.2468114, 0.28323662, 0.53594093),
    QtGui.QColor.fromRgbF(0.2316735, 0.3181058, 0.54483444),
    QtGui.QColor.fromRgbF(0.21620971, 0.35153548, 0.55062743),
    QtGui.QColor.fromRgbF(0.20123854, 0.38366989, 0.55429441),
    QtGui.QColor.fromRgbF(0.18723083, 0.41474645, 0.55654717),
    QtGui.QColor.fromRgbF(0.17427363, 0.4450441, 0.55779216),
    QtGui.QColor.fromRgbF(0.16214155, 0.47483821, 0.55813967),
    QtGui.QColor.fromRgbF(0.15047605, 0.50436904, 0.55742968),
    QtGui.QColor.fromRgbF(0.13914708, 0.53381201, 0.55529773),
    QtGui.QColor.fromRgbF(0.12872938, 0.56326503, 0.55122927),
    QtGui.QColor.fromRgbF(0.12114807, 0.59273889, 0.54464114),
    QtGui.QColor.fromRgbF(0.12008079, 0.62216081, 0.53494633),
    QtGui.QColor.fromRgbF(0.13006688, 0.65138436, 0.52160791),
    QtGui.QColor.fromRgbF(0.15389405, 0.68020343, 0.50417217),
    QtGui.QColor.fromRgbF(0.19109018, 0.70836635, 0.48228395),
    QtGui.QColor.fromRgbF(0.2393739, 0.73558828, 0.45568838),
    QtGui.QColor.fromRgbF(0.29647899, 0.76156142, 0.42422341),
    QtGui.QColor.fromRgbF(0.36074053, 0.78596419, 0.38781353),
    QtGui.QColor.fromRgbF(0.43098317, 0.80847343, 0.34647607),
    QtGui.QColor.fromRgbF(0.5062713, 0.82878621, 0.30036211),
    QtGui.QColor.fromRgbF(0.58567772, 0.84666139, 0.24989748),
    QtGui.QColor.fromRgbF(0.66805369, 0.86199932, 0.19629307),
    QtGui.QColor.fromRgbF(0.75188414, 0.87495143, 0.14322828),
    QtGui.QColor.fromRgbF(0.83526959, 0.88602943, 0.1026459),
    QtGui.QColor.fromRgbF(0.91624212, 0.89609127, 0.1007168),
    QtGui.QColor.fromRgbF(0.99324789, 0.90615657, 0.1439362),
]


symbols = ["t", "o", "s", "d", "+", "star", "t1", "x"]
