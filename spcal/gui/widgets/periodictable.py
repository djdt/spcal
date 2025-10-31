import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.objects import KeepMenuOpenFilter
from spcal.isotope import ISOTOPE_TABLE, RECOMMENDED_ISOTOPES, SPCalIsotope
from spcal.npdb import db

ELEMENT_PERIOD_INFO: dict[str, tuple[str, int, tuple[int, int]]] = {
    # symbol: (name, number, position in periodic table)
    "H": ("Hydrogen", 1, (0, 0)),
    "He": ("Helium", 2, (0, 17)),
    "Li": ("Lithium", 3, (1, 0)),
    "Be": ("Beryllium", 4, (1, 1)),
    "B": ("Boron", 5, (1, 12)),
    "C": ("Carbon", 6, (1, 13)),
    "N": ("Nitrogen", 7, (1, 14)),
    "O": ("Oxygen", 8, (1, 15)),
    "F": ("Fluorine", 9, (1, 16)),
    "Ne": ("Neon", 10, (1, 17)),
    "Na": ("Sodium", 11, (2, 0)),
    "Mg": ("Magnesium", 12, (2, 1)),
    "Al": ("Aluminium", 13, (2, 12)),
    "Si": ("Silicon", 14, (2, 13)),
    "P": ("Phosphorus", 15, (2, 14)),
    "S": ("Suplhur", 16, (2, 15)),
    "Cl": ("Chlorine", 17, (2, 16)),
    "Ar": ("Argon", 18, (2, 17)),
    "K": ("Potassium", 19, (3, 0)),
    "Ca": ("Calcium", 20, (3, 1)),
    "Sc": ("Scandium", 21, (3, 2)),
    "Ti": ("Titanium", 22, (3, 3)),
    "V": ("Vanadium", 23, (3, 4)),
    "Cr": ("Chromium", 24, (3, 5)),
    "Mn": ("Manganese", 25, (3, 6)),
    "Fe": ("Iron", 26, (3, 7)),
    "Co": ("Cobalt", 27, (3, 8)),
    "Ni": ("Nickle", 28, (3, 9)),
    "Cu": ("Copper", 29, (3, 10)),
    "Zn": ("Zinc", 30, (3, 11)),
    "Ga": ("Gallium", 31, (3, 12)),
    "Ge": ("Germanium", 32, (3, 13)),
    "As": ("Aresenic", 33, (3, 14)),
    "Se": ("Selenium", 34, (3, 15)),
    "Br": ("Bromine", 35, (3, 16)),
    "Kr": ("Krypton", 36, (3, 17)),
    "Rb": ("Rubidium", 37, (4, 0)),
    "Sr": ("Strontium", 38, (4, 1)),
    "Y": ("Yttrium", 39, (4, 2)),
    "Zr": ("Zirconium", 40, (4, 3)),
    "Nb": ("Noibium", 41, (4, 4)),
    "Mo": ("Molybdenum", 42, (4, 5)),
    "Tc": ("Technetium", 43, (4, 6)),
    "Ru": ("Ruthenium", 44, (4, 7)),
    "Rh": ("Rhodium", 45, (4, 8)),
    "Pd": ("Paladaium", 46, (4, 9)),
    "Ag": ("Silver", 47, (4, 10)),
    "Cd": ("Cadmium", 48, (4, 11)),
    "In": ("Indium", 49, (4, 12)),
    "Sn": ("Tin", 50, (4, 13)),
    "Sb": ("Antimony", 51, (4, 14)),
    "Te": ("Tellurium", 52, (4, 15)),
    "I": ("Iodine", 53, (4, 16)),
    "Xe": ("Xenon", 54, (4, 17)),
    "Cs": ("Caesium", 55, (5, 0)),
    "Ba": ("Barium", 56, (5, 1)),
    "Hf": ("Hafnium", 72, (5, 3)),
    "Ta": ("Tantalum", 73, (5, 4)),
    "W": ("Tungsten", 74, (5, 5)),
    "Re": ("Rhenium", 75, (5, 6)),
    "Os": ("Osmium", 76, (5, 7)),
    "Ir": ("Iridium", 77, (5, 8)),
    "Pt": ("Platinum", 78, (5, 9)),
    "Au": ("Gold", 79, (5, 10)),
    "Hg": ("Mercury", 80, (5, 11)),
    "Tl": ("Thallium", 81, (5, 12)),
    "Pb": ("Lead", 82, (5, 13)),
    "Bi": ("Bismuth", 83, (5, 14)),
    "Po": ("Polonium", 84, (5, 15)),
    "At": ("Astatine", 85, (5, 16)),
    "Rn": ("Radon", 86, (5, 17)),
    "Fr": ("Francium", 87, (6, 0)),
    "Ra": ("Radium", 88, (6, 1)),
    "Rf": ("Rutherfordium", 104, (6, 3)),
    "Db": ("Dubnium", 105, (6, 4)),
    "Sg": ("Seaborgium", 106, (6, 5)),
    "Bh": ("Bohrium", 107, (6, 6)),
    "Hs": ("Hassium", 108, (6, 7)),
    "La": ("Lanthanum", 57, (7, 2)),
    "Ce": ("Cerium", 58, (7, 3)),
    "Pr": ("Praseodymium", 59, (7, 4)),
    "Nd": ("Neodymium", 60, (7, 5)),
    "Pm": ("Promethium", 61, (7, 6)),
    "Sm": ("Samarium", 62, (7, 7)),
    "Eu": ("Europium", 63, (7, 8)),
    "Gd": ("Gadolinium", 64, (7, 9)),
    "Tb": ("Terbium", 65, (7, 10)),
    "Dy": ("Dysprosium", 66, (7, 11)),
    "Ho": ("Holmium", 67, (7, 12)),
    "Er": ("Erbium", 68, (7, 13)),
    "Tm": ("Thulium", 69, (7, 14)),
    "Yb": ("Ytterbium", 70, (7, 15)),
    "Lu": ("Lutetium", 71, (7, 16)),
    "Ac": ("Actinium", 89, (8, 2)),
    "Th": ("Thorium", 90, (8, 3)),
    "Pa": ("Protactinium", 91, (8, 4)),
    "U": ("Uranium", 92, (8, 5)),
    "Np": ("Neptunium", 93, (8, 6)),
    "Pu": ("Plutonium", 94, (8, 7)),
    "Am": ("Americium", 95, (8, 8)),
    "Cm": ("Curium", 96, (8, 9)),
    "Bk": ("Berkelium", 97, (8, 10)),
    "Cf": ("Californium", 98, (8, 11)),
    "Es": ("Einsteinium", 99, (8, 12)),
    "Fm": ("Fermium", 100, (8, 13)),
    "Md": ("Mendelevium", 101, (8, 14)),
    "No": ("Nobelium", 102, (8, 15)),
    "Lr": ("Lawrencium", 103, (8, 16)),
}


class PeriodicTableButton(QtWidgets.QToolButton):
    isotopesChanged = QtCore.Signal()

    def __init__(
        self,
        isotopes: list[SPCalIsotope],
        enabled: list[SPCalIsotope] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)

        self.isotopes = isotopes

        self.indicator: QtGui.QColor | None = None

        self.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.DelayedPopup)
        self.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.setMinimumSize(QtCore.QSize(45, 45))

        self.action = QtGui.QAction(self.isotopes[0].symbol, parent=self)
        self.action.setToolTip(ELEMENT_PERIOD_INFO[self.isotopes[0].symbol][0])
        self.action.setCheckable(True)
        self.setDefaultAction(self.action)

        self.isotope_actions = {
            iso.isotope: self.createAction(iso) for iso in self.isotopes
        }
        if enabled is not None:
            self.setEnabledIsotopes(enabled)

        isotopes_menu = QtWidgets.QMenu("Isotopes", parent=self)
        isotopes_menu.addActions(list(self.isotope_actions.values()))

        isotopes_menu.installEventFilter(KeepMenuOpenFilter(isotopes_menu))

        self.setMenu(isotopes_menu)
        self.setEnabled(
            any(action.isEnabled() for action in self.isotope_actions.values())
        )

        self.clicked.connect(self.selectPreferredIsotopes)
        self.isotopesChanged.connect(self.updateChecked)

    def preferred(self) -> SPCalIsotope | None:
        enabled = self.enabledIsotopes()
        for isotope in enabled:
            if (
                isotope.symbol in RECOMMENDED_ISOTOPES
                and RECOMMENDED_ISOTOPES[isotope.symbol] == isotope.isotope
            ):
                return isotope

        preferred = max(enabled, key=lambda iso: iso.composition or -1)
        if preferred.composition is None:
            return None
        return preferred

    def createAction(self, isotope: SPCalIsotope) -> QtGui.QAction:
        text = f"{isotope.isotope:3}: {isotope.mass:.4f}"
        if isotope.composition is not None:
            text += f"\t{isotope.composition * 100.0:.2f}%"

        action = QtGui.QAction(text, parent=self)
        action.setCheckable(True)
        action.toggled.connect(self.isotopesChanged)
        return action

    def enabledIsotopes(self) -> list[SPCalIsotope]:
        nums = [n for n, action in self.isotope_actions.items() if action.isEnabled()]
        return [iso for iso in self.isotopes if iso.isotope in nums]

    def setEnabledIsotopes(self, enabled: list[SPCalIsotope]):
        enabled_nums = [iso.isotope for iso in enabled]
        for isotope, action in self.isotope_actions.items():
            if isotope in enabled_nums:
                action.setEnabled(True)
            else:
                action.setEnabled(False)
                action.setChecked(False)

    def selectedIsotopes(self) -> list[SPCalIsotope]:
        nums = [n for n, action in self.isotope_actions.items() if action.isChecked()]
        return [iso for iso in self.isotopes if iso.isotope in nums]

    def selectPreferredIsotopes(self, checked: bool) -> None:
        preferred = self.preferred()
        if preferred is None:
            return
        for num, action in self.isotope_actions.items():
            if checked and num == preferred.isotope:
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
            option.rect.adjusted(2, 0, 0, 0),  # type: ignore
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop,
            self.palette(),
            self.isEnabled(),
            str(ELEMENT_PERIOD_INFO[self.isotopes[0].symbol][1]),
        )

        # Draw number selected
        num = len(self.selectedIsotopes())
        if num > 0:
            self.style().drawItemText(
                painter,
                option.rect.adjusted(2, 0, 0, 0),  # type: ignore
                QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignBottom,
                self.palette(),
                self.isEnabled(),
                f"{num}/{len(self.isotopes)}",
            )

        # Draw color icon
        if self.indicator is not None:
            rect = QtCore.QRectF(0.0, 0.0, 10.0, 10.0)
            rect.moveTopRight(option.rect.topRight() + QtCore.QPoint(-2, 3))  # type: ignore
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
        for symbol in ELEMENT_PERIOD_INFO.keys():
            # Limit to chosen ones
            isotopes = [v for k, v in ISOTOPE_TABLE.items() if k[0] == symbol]
            self.buttons[symbol] = PeriodicTableButton(isotopes)
            self.buttons[symbol].isotopesChanged.connect(self.isotopesChanged)

        layout = QtWidgets.QGridLayout()
        row = 0
        for symbol, (_, _, (row, col)) in ELEMENT_PERIOD_INFO.items():
            layout.addWidget(self.buttons[symbol], row, col)
        layout.setRowStretch(row + 1, 1)  # Last row stretch

        self.isotopesChanged.connect(self.findCollisions)
        self.setLayout(layout)

    def enabledIsotopes(self) -> list[SPCalIsotope]:
        enabled: list[SPCalIsotope] = []
        for button in self.buttons.values():
            enabled.extend(button.enabledIsotopes())
        return enabled

    def setEnabledIsotopes(self, enabled: list[SPCalIsotope]) -> None:
        for symbol, button in self.buttons.items():
            button_enabled = [iso for iso in enabled if iso.symbol == symbol]
            button.setEnabled(len(button_enabled) > 0)
            button.setEnabledIsotopes(button_enabled)

    def selectedIsotopes(self) -> list[SPCalIsotope]:
        selected: list[SPCalIsotope] = []
        for button in self.buttons.values():
            selected.extend(button.selectedIsotopes())
        return selected

    def setSelectedIsotopes(self, selected: list[SPCalIsotope]) -> None:
        self.blockSignals(True)
        for button in self.buttons.values():
            for action in button.isotope_actions.values():
                action.setChecked(False)

        for isotope in selected:
            self.buttons[isotope.symbol].isotope_actions[isotope.isotope].setChecked(
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
            if len(selected) == 0:  # pragma: no cover
                other_selected = []
            else:
                other_selected = [
                    iso.isotope for iso in selected if iso.symbol != symbol
                ]

            collisions = 0
            for num, action in button.isotope_actions.items():
                if num in other_selected:
                    action.setIcon(QtGui.QIcon.fromTheme("folder-important"))
                    collisions += 1
                else:
                    action.setIcon(QtGui.QIcon())

            if collisions > 0:
                button.setIcon(QtGui.QIcon.fromTheme("folder-important"))
            else:
                button.setIcon(QtGui.QIcon())


if __name__ == "__main__":
    app = QtWidgets.QApplication()
    table = PeriodicTableSelector()
    table.show()
    app.exec()
