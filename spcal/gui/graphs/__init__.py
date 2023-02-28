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

symbols = ["t", "o", "s", "d", "+", "star", "t1", "x"]
