import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.objects import KeepMenuOpenFilter
from spcal.npdb import db

element_positions = {
    "H": (0, 0),
    "He": (0, 17),
    "Li": (1, 0),
    "Be": (1, 1),
    "B": (1, 12),
    "C": (1, 13),
    "N": (1, 14),
    "O": (1, 15),
    "F": (1, 16),
    "Ne": (1, 17),
    "Na": (2, 0),
    "Mg": (2, 1),
    "Al": (2, 12),
    "Si": (2, 13),
    "P": (2, 14),
    "S": (2, 15),
    "Cl": (2, 16),
    "Ar": (2, 17),
    "K": (3, 0),
    "Ca": (3, 1),
    "Sc": (3, 2),
    "Ti": (3, 3),
    "V": (3, 4),
    "Cr": (3, 5),
    "Mn": (3, 6),
    "Fe": (3, 7),
    "Co": (3, 8),
    "Ni": (3, 9),
    "Cu": (3, 10),
    "Zn": (3, 11),
    "Ga": (3, 12),
    "Ge": (3, 13),
    "As": (3, 14),
    "Se": (3, 15),
    "Br": (3, 16),
    "Kr": (3, 17),
    "Rb": (4, 0),
    "Sr": (4, 1),
    "Y": (4, 2),
    "Zr": (4, 3),
    "Nb": (4, 4),
    "Mo": (4, 5),
    "Tc": (4, 6),
    "Ru": (4, 7),
    "Rh": (4, 8),
    "Pd": (4, 9),
    "Ag": (4, 10),
    "Cd": (4, 11),
    "In": (4, 12),
    "Sn": (4, 13),
    "Sb": (4, 14),
    "Te": (4, 15),
    "I": (4, 16),
    "Xe": (4, 17),
    "Cs": (5, 0),
    "Ba": (5, 1),
    "Hf": (5, 3),
    "Ta": (5, 4),
    "W": (5, 5),
    "Re": (5, 6),
    "Os": (5, 7),
    "Ir": (5, 8),
    "Pt": (5, 9),
    "Au": (5, 10),
    "Hg": (5, 11),
    "Tl": (5, 12),
    "Pb": (5, 13),
    "Bi": (5, 14),
    "Po": (5, 15),
    "At": (5, 16),
    "Rn": (5, 17),
    "Fr": (6, 0),
    "Ra": (6, 1),
    "Rf": (6, 3),
    "Db": (6, 4),
    "Sg": (6, 5),
    "Bh": (6, 6),
    "Hs": (6, 7),
    "La": (7, 2),
    "Ce": (7, 3),
    "Pr": (7, 4),
    "Nd": (7, 5),
    "Pm": (7, 6),
    "Sm": (7, 7),
    "Eu": (7, 8),
    "Gd": (7, 9),
    "Tb": (7, 10),
    "Dy": (7, 11),
    "Ho": (7, 12),
    "Er": (7, 13),
    "Tm": (7, 14),
    "Yb": (7, 15),
    "Lu": (7, 16),
    "Ac": (8, 2),
    "Th": (8, 3),
    "Pa": (8, 4),
    "U": (8, 5),
    "Np": (8, 6),
    "Pu": (8, 7),
    "Am": (8, 8),
    "Cm": (8, 9),
    "Bk": (8, 10),
    "Cf": (8, 11),
    "Es": (8, 12),
    "Fm": (8, 13),
    "Md": (8, 14),
    "No": (8, 15),
    "Lr": (8, 16),
}


class PeriodicTableButton(QtWidgets.QToolButton):
    isotopesChanged = QtCore.Signal()

    def __init__(
        self,
        isotopes: np.ndarray,
        enabled: np.ndarray | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)

        self.isotopes = isotopes
        self.symbol = isotopes["Symbol"][0]
        self.name = db["elements"][db["elements"]["Symbol"] == self.symbol]["Name"][0]
        self.number = isotopes["Number"][0]

        self.indicator: QtGui.QColot | None = None

        self.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.DelayedPopup)
        self.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.setMinimumSize(QtCore.QSize(45, 45))

        self.action = QtGui.QAction(self.symbol, parent=self)
        self.action.setToolTip(self.name)
        self.action.setCheckable(True)
        self.setDefaultAction(self.action)

        self.actions = {iso["Isotope"]: self.createAction(iso) for iso in self.isotopes}
        if enabled is not None:
            for isotope, action in self.actions.items():
                action.setEnabled(isotope in enabled["Isotope"])

        isotopes_menu = QtWidgets.QMenu("Isotopes", parent=self)
        isotopes_menu.addActions(list(self.actions.values()))

        isotopes_menu.installEventFilter(KeepMenuOpenFilter(isotopes_menu))

        self.setMenu(isotopes_menu)
        self.setEnabled(any(action.isEnabled() for action in self.actions.values()))

        self.clicked.connect(self.selectPreferredIsotopes)
        self.isotopesChanged.connect(self.updateChecked)

    def preferred(self) -> np.ndarray:
        pref = self.isotopes["Preferred"] > 0
        if not np.any(pref):
            return self.isotopes[np.nanargmax(self.isotopes["Composition"])]
        return self.isotopes[pref]

    def createAction(self, isotope: np.ndarray) -> QtGui.QAction:
        text = f"{isotope['Isotope']:3}: {isotope['Mass']:.4f}"
        if not np.isnan(isotope["Composition"]):
            text += f"\t{isotope['Composition'] * 100.0:.2f}%"

        action = QtGui.QAction(text, parent=self)
        action.setCheckable(True)
        action.toggled.connect(self.isotopesChanged)
        return action

    def enabledIsotopes(self) -> np.ndarray:
        nums = np.array([n for n, action in self.actions.items() if action.isEnabled()])
        return self.isotopes[np.isin(self.isotopes["Isotope"], nums)]

    def selectedIsotopes(self) -> np.ndarray:
        nums = np.array([n for n, action in self.actions.items() if action.isChecked()])
        return self.isotopes[np.isin(self.isotopes["Isotope"], nums)]

    def selectPreferredIsotopes(self, checked: bool) -> None:
        preferred = self.preferred()
        for num, action in self.actions.items():
            if checked and np.isin(num, preferred["Isotope"]):
                action.setChecked(True)
            else:
                action.setChecked(False)

        self.update()

    def updateChecked(self) -> None:
        if len(self.selectedIsotopes()) > 0:
            self.setChecked(True)
        else:
            self.setChecked(False)
        self.update()

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        self.showMenu()  # pragma: no cover

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)

        painter = QtGui.QPainter(self)
        option = QtWidgets.QStyleOptionToolButton()
        self.initStyleOption(option)

        font = self.font()
        font.setPointSizeF(font.pointSizeF() * 0.66)
        painter.setFont(font)

        # Draw element number
        self.style().drawItemText(
            painter,
            option.rect.adjusted(2, 0, 0, 0),
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop,
            self.palette(),
            self.isEnabled(),
            str(self.number),
        )

        # Draw number selected
        num = len(self.selectedIsotopes())
        if num > 0:
            self.style().drawItemText(
                painter,
                option.rect.adjusted(2, 0, 0, 0),
                QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignBottom,
                self.palette(),
                self.isEnabled(),
                f"{num}/{len(self.isotopes)}",
            )

        # Draw color icon
        if self.indicator is not None:
            rect = QtCore.QRectF(0.0, 0.0, 10.0, 10.0)
            rect.moveTopRight(option.rect.topRight() + QtCore.QPoint(-2, 3))
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.setBrush(QtGui.QBrush(self.indicator))
            painter.drawEllipse(rect)


class PeriodicTableSelector(QtWidgets.QWidget):
    isotopesChanged = QtCore.Signal()

    def __init__(
        self,
        enabled_isotopes: np.ndarray | None = None,
        selected_isotopes: np.ndarray | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.pkeys: list[int] = []

        if enabled_isotopes is None:
            enabled_isotopes = db["isotopes"]

        self.buttons: dict[str, PeriodicTableButton] = {}
        for symbol in element_positions.keys():
            # Limit to chosen ones
            isotopes = db["isotopes"][db["isotopes"]["Symbol"] == symbol]
            self.buttons[symbol] = PeriodicTableButton(
                isotopes, enabled_isotopes[enabled_isotopes["Symbol"] == symbol]
            )
            self.buttons[symbol].isotopesChanged.connect(self.isotopesChanged)

        layout = QtWidgets.QGridLayout()
        for symbol, (row, col) in element_positions.items():
            layout.addWidget(self.buttons[symbol], row, col)
        layout.setRowStretch(row + 1, 1)  # Last row stretch

        self.isotopesChanged.connect(self.findCollisions)
        self.setLayout(layout)

    def enabledIsotopes(self) -> np.ndarray:
        enabled: list[np.ndarray] = []
        for button in self.buttons.values():
            enabled.extend(button.enabledIsotopes())
        return np.stack(enabled)

    def selectedIsotopes(self) -> np.ndarray | None:
        selected: list[np.ndarray] = []
        for button in self.buttons.values():
            selected.extend(button.selectedIsotopes())
        if len(selected) == 0:
            return None
        return np.stack(selected)

    def setSelectedIsotopes(self, isotopes: np.ndarray | None) -> None:
        self.blockSignals(True)
        for button in self.buttons.values():
            for action in button.actions.values():
                action.setChecked(False)
        if isotopes is not None:
            for isotope in isotopes:
                self.buttons[isotope["Symbol"]].actions[isotope["Isotope"]].setChecked(
                    True
                )
        self.blockSignals(False)
        self.isotopesChanged.emit()

    def setIsotopeColors(
        self, isotopes: np.ndarray, colors: list[QtGui.QColor]
    ) -> None:
        """Set the indicator colors for ``isotopes`` to ``colors.

        Will change text to BrightText ColorRole if a dark color is used.
        Sets other buttons to the default color.
        """
        for button in self.buttons.values():
            button.indicator = None

        for isotope, color in zip(isotopes, colors):
            self.buttons[isotope["Symbol"]].indicator = color

    def findCollisions(self) -> None:
        selected = self.selectedIsotopes()

        for symbol, button in self.buttons.items():
            if selected is None:  # pragma: no cover
                other_selected = []
            else:
                other_selected = selected[selected["Symbol"] != symbol]["Isotope"]

            collisions = 0
            for num, action in button.actions.items():
                if num in other_selected:
                    action.setIcon(QtGui.QIcon.fromTheme("folder-important"))
                    collisions += 1
                else:
                    action.setIcon(QtGui.QIcon())

            if collisions > 0:
                button.setIcon(QtGui.QIcon.fromTheme("folder-important"))
            else:
                button.setIcon(QtGui.QIcon())


#     def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
#         super().keyPressEvent(event)
#         self.pkeys.append(int(event.key()))
#         if len(self.pkeys) > 12:
#             self.pkeys.pop(0)

#         if self.pkeys == [85, 85, 68, 68, 76, 82, 76, 82, 65, 66, 83, 83]:
#             self.buttons["Au"].setText("Dz")
#             self.buttons["C"].setText("Sv")
#             self.buttons["Cu"].setText("To")
#             self.buttons["Hg"].setText("Jk")
#             self.buttons["Ho"].setText("Mk")
