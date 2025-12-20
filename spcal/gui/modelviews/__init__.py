from PySide6 import QtCore

# Value and Unit widgets
ValueErrorRole = QtCore.Qt.ItemDataRole.UserRole + 1
BaseValueRole = QtCore.Qt.ItemDataRole.UserRole + 2
BaseValueErrorRole = ValueErrorRole + 3
CurrentUnitRole = QtCore.Qt.ItemDataRole.UserRole + 4
UnitsRole = QtCore.Qt.ItemDataRole.UserRole + 5
UnitLabelRole = QtCore.Qt.ItemDataRole.UserRole + 6

# Isotope
IsotopeRole = QtCore.Qt.ItemDataRole.UserRole + 10

# DataFile
DataFileRole = QtCore.Qt.ItemDataRole.UserRole + 20

# Options
IsotopeOptionRole = QtCore.Qt.ItemDataRole.UserRole + 30
