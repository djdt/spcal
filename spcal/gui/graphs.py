# @Todo pyqtgraph is much faster than QtCharts for many points, redo plotting in that 
from PySide2 import QtWidgets
import numpy as np
import pyqtgraph

from spcal.gui.dialogs import ImportDialog

app = QtWidgets.QApplication()

view  = pyqtgraph.GraphicsView(background="white")
layout = pyqtgraph.GraphicsLayout()
view.setCentralWidget(layout)


def plot(x):
    print(x)
    for name in x.dtype.names:
        p = layout.addPlot(title=name)
        layout.nextRow()
        p.plot(x[name])
        p.plot(x[name])
    view.show()



dlg = ImportDialog("/home/tom/Downloads/AuAg.csv")

dlg.dataImported.connect(plot)
dlg.open()

app.exec_()
