from PySide6 import QtCore, QtGui, QtWidgets

from . import icons

# Set Some Qt attributes
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
QtGui.QIcon.setThemeName("spcal")
