from PySide6 import QtCore, QtGui, QtWidgets

# Set Some Qt attributes
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
QtGui.QIcon.setFallbackThemeName("spcal")
QtGui.QIcon.setFallbackSearchPaths([":/icons"])
