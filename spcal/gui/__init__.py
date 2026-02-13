from PySide6 import QtCore, QtGui


def scheme_from_palette() -> QtCore.Qt.ColorScheme:
    palette = QtGui.QGuiApplication.palette()
    if palette.window().color().value() > palette.windowText().color().value():
        return QtCore.Qt.ColorScheme.Light
    else:
        return QtCore.Qt.ColorScheme.Dark


# Set the icon theme
scheme = QtGui.QGuiApplication.styleHints().colorScheme()
if scheme == QtCore.Qt.ColorScheme.Unknown:
    scheme = scheme_from_palette()

if scheme == QtCore.Qt.ColorScheme.Dark:
    QtGui.QIcon.setThemeName("spcal-dark")
else:
    QtGui.QIcon.setThemeName("spcal")
