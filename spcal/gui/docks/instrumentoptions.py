import configparser
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.util import create_action
from spcal.gui.widgets import UnitsWidget, ValueWidget
from spcal.siunits import time_units


class SPCalDockWidgetTitleButton(QtWidgets.QToolButton):
    def __init__(
        self,
        icon: QtWidgets.QStyle.StandardPixmap | QtGui.QIcon,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setAutoRaise(True)
        self.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)
        palette = self.palette()
        palette.setColor(
            palette.ColorRole.Highlight, palette.color(palette.ColorRole.Window)
        )
        palette.setColor(palette.ColorRole.__str__)
        self.setPalette(palette)

        if isinstance(icon, QtGui.QIcon):
            self.setIcon(icon)
            # self.
        else:
            self.setIcon(
                self.style().standardIcon(
                    icon,
                    QtWidgets.QStyleOptionDockWidget(),
                )
            )


class SaveLoadDockTitleBar(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)

        self.label = QtWidgets.QLabel()
        self.setWindowTitle("title")

        self.button_menu = SPCalDockWidgetTitleButton(
            QtWidgets.QStyle.StandardPixmap.SP_ArrowDown
        )
        self.button_float = SPCalDockWidgetTitleButton(
            QtWidgets.QStyle.StandardPixmap.SP_TitleBarNormalButton
        )
        self.button_close = SPCalDockWidgetTitleButton(
            QtWidgets.QStyle.StandardPixmap.SP_TitleBarCloseButton
        )

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout_bar = QtWidgets.QHBoxLayout()
        layout_bar.addWidget(self.label)
        layout_bar.addWidget(self.button_menu)
        layout_bar.addWidget(self.button_float)
        layout_bar.addWidget(self.button_close)
        layout_bar.setContentsMargins(0, 0, 0, 0)

        layout.addLayout(layout_bar)
        layout.addWidget(QtWidgets.QFrame(frameShape=QtWidgets.QFrame.HLine))
        self.setLayout(layout)

    def changeEvent(self, event: QtCore.QEvent) -> None:
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.Type.WindowTitleChange:
            self.label.setText(self.windowTitle())


class SPCalInstrumentOptionsDock(QtWidgets.QDockWidget):
    optionsChanged = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Instrument Options")
        # self.setTitleBarWidget(SaveLoadDockTitleBar())

        uptake_units = {
            "ml/min": 1e-3 / 60.0,
            "ml/s": 1e-3,
            "L/min": 1.0 / 60.0,
            "L/s": 1.0,
        }

        # load stored options
        settings = QtCore.QSettings()
        sf = int(settings.value("SigFigs", 4))

        # Instrument wide options
        self.dwelltime = UnitsWidget(
            time_units,
            default_unit="ms",
            validator=QtGui.QDoubleValidator(0.0, 10.0, 10),
            format=sf,
        )
        self.dwelltime.setReadOnly(True)

        self.uptake = UnitsWidget(
            uptake_units,
            default_unit="ml/min",
            format=sf,
        )
        self.efficiency = ValueWidget(
            validator=QtGui.QDoubleValidator(0.0, 1.0, 10), format=sf
        )

        self.dwelltime.setToolTip(
            "ICP-MS event-time, updated from imported files if time column exists."
        )
        self.uptake.setToolTip("ICP-MS sample flow rate.")
        self.efficiency.setToolTip(
            "Transport efficiency. Can be calculated using a reference particle."
        )

        self.efficiency_method = QtWidgets.QComboBox()
        self.efficiency_method.addItems(
            ["Manual Input", "Reference Particle", "Mass Response"]
        )
        self.efficiency_method.currentTextChanged.connect(self.efficiencyMethodChanged)
        for i, tooltip in enumerate(
            [
                "Manually enter the transport efficiency.",
                "Calculate the efficiency using a reference particle.",
                "Use the mass response of a reference particle.",
            ]
        ):
            self.efficiency_method.setItemData(i, tooltip, QtCore.Qt.ToolTipRole)
        self.efficiency_method.setToolTip(
            self.efficiency_method.currentData(QtCore.Qt.ToolTipRole)
        )
        self.efficiency_method.currentIndexChanged.connect(
            lambda i: self.efficiency_method.setToolTip(
                self.efficiency_method.itemData(i, QtCore.Qt.ToolTipRole)
            )
        )

        # Complete Changed
        self.uptake.baseValueChanged.connect(self.optionsChanged)
        self.efficiency.valueChanged.connect(self.optionsChanged)

        self.save_button = QtWidgets.QToolButton()
        self.save_button.setAutoRaise(True)
        self.save_button.setPopupMode(
            QtWidgets.QToolButton.ToolButtonPopupMode.MenuButtonPopup
        )
        self.save_button.setIcon(QtGui.QIcon.fromTheme("document-save"))

        self.action_save = create_action(
            "document-save",
            "Save Options",
            "Save instrument options to a file.",
            lambda: self.saveToFile(),
        )
        self.action_load = create_action(
            "document-open",
            "Load Options",
            "Load instrument options from a file.",
            self.loadFromFile,
        )

        buttons_layout = QtWidgets.QHBoxLayout()
        buttons_layout.addWidget(self.save_button)

        # pm = self.style().pixelMetric(QtWidgets.QStyle.PixelMetric.PM_DockWidgetTitleMargin, None, self)
        # tw = self.fontMetrics().boundingRect(self.windowTitle())
        # self.save_button.move(pm*2+tw, 0)
        # self.save_button.move(0, pm*2 + self.save_button.height())

        form_layout = QtWidgets.QFormLayout()

        form_layout.addRow("Uptake:", self.uptake)
        form_layout.addRow("Event time:", self.dwelltime)
        form_layout.addRow("Trans. Efficiency:", self.efficiency)
        form_layout.addRow("", self.efficiency_method)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(form_layout)
        layout.addWidget(
            self.save_button,
            QtCore.Qt.AlignmentFlag.AlignBottom | QtCore.Qt.AlignmentFlag.AlignLeft,
        )

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setWidget(widget)

        self.save_button.setDefaultAction(self.action_save)
        self.save_button.addAction(self.action_load)

    def saveToFile(self, file: str | Path | None = None) -> None:
        state = self.state()

        if file is None:
            # maybe add a global?
            settings = QtCore.QSettings()
            num = settings.beginReadArray("RecentFiles")
            settings.setArrayIndex(0)
            if num > 0:
                dir = settings.value("Path")
            else:
                dir = ""

            file, filter = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save Instrument Options",
                str(Path(dir).with_suffix(".spcalio")),
                "SPCal Instrument Options (*.spcalio)",
            )

        if file != "":
            config = configparser.ConfigParser()
            config["Instrument Options"] = state
            with open(file, "w") as fp:
                config.write(fp)

    def loadFromFile(self, file: str | Path) -> None:
        print("LOAD")
        pass

    def state(self) -> dict:
        state_dict = {
            "uptake": self.uptake.baseValue(),
            "dwelltime": self.dwelltime.baseValue(),
            "efficiency": self.efficiency.value(),
            "efficiency method": self.efficiency_method.currentText(),
        }
        return {k: v for k, v in state_dict.items() if v is not None}

    def setState(self, state: dict) -> None:
        self.blockSignals(True)
        if "uptake" in state:
            self.uptake.setBaseValue(state["uptake"])
            self.uptake.setBestUnit()
        if "dwelltime" in state:
            self.dwelltime.setBaseValue(state["dwelltime"])
            self.dwelltime.setBestUnit()
        if "efficiency" in state:
            self.efficiency.setValue(state["efficiency"])

        self.efficiency_method.setCurrentText(state["efficiency method"])
        self.blockSignals(False)

        self.optionsChanged.emit()

    def efficiencyMethodChanged(self, method: str) -> None:
        if method == "Manual Input":
            self.uptake.setEnabled(True)
            self.efficiency.setEnabled(True)
        elif method == "Reference Particle":
            self.uptake.setEnabled(True)
            self.efficiency.setEnabled(False)
        elif method == "Mass Response":
            self.uptake.setEnabled(False)
            self.efficiency.setEnabled(False)

        self.optionsChanged.emit()

    def isComplete(self) -> bool:
        if self.window_size.isEnabled() and not self.window_size.hasAcceptableInput():
            return False

        method = self.efficiency_method.currentText()
        if method == "Manual Input":
            return all(
                [
                    self.dwelltime.hasAcceptableInput(),
                    self.uptake.hasAcceptableInput(),
                    self.efficiency.hasAcceptableInput(),
                ]
            )
        elif method == "Reference Particle":
            return all(
                [
                    self.dwelltime.hasAcceptableInput(),
                    self.uptake.hasAcceptableInput(),
                ]
            )
        elif method == "Mass Response":
            return all(
                [
                    self.dwelltime.hasAcceptableInput(),
                ]
            )
        else:
            raise ValueError(f"Unknown method {method}.")

    def resetInputs(self) -> None:
        self.blockSignals(True)
        self.uptake.setValue(None)
        self.dwelltime.setValue(None)
        self.efficiency.setValue(None)
        self.blockSignals(False)
        self.optionsChanged.emit()

    def setSignificantFigures(self, num: int | None = None) -> None:
        if num is None:
            num = int(QtCore.QSettings().value("SigFigs", 4))
        for widget in self.findChildren(ValueWidget):
            if widget.view_format[1] == "g":
                widget.setViewFormat(num)
