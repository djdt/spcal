from PySide6 import QtCore

# Value and Unit widgets
ValueErrorRole = QtCore.Qt.ItemDataRole.UserRole + 10
BaseValueRole = QtCore.Qt.ItemDataRole.UserRole + 11
BaseValueErrorRole = ValueErrorRole + 12
CurrentUnitRole = QtCore.Qt.ItemDataRole.UserRole + 13
UnitsRole = QtCore.Qt.ItemDataRole.UserRole + 14
UnitLabelRole = QtCore.Qt.ItemDataRole.UserRole + 15

# Isotope
IsotopeRole = QtCore.Qt.ItemDataRole.UserRole + 20

# DataFile
DataFileRole = QtCore.Qt.ItemDataRole.UserRole + 30

# Options
IsotopeOptionRole = QtCore.Qt.ItemDataRole.UserRole + 40
