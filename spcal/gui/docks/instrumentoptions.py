from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.widgets import UnitsWidget, ValueWidget
from spcal.processing import SPCalInstrumentOptions
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
        layout.addWidget(QtWidgets.QFrame(frameShape=QtWidgets.QFrame.Shape.HLine))
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

        uptake_units = {
            "ml/min": 1e-3 / 60.0,
            "ml/s": 1e-3,
            "L/min": 1.0 / 60.0,
            "L/s": 1.0,
        }

        # load stored options
        settings = QtCore.QSettings()
        sf = int(settings.value("SigFigs", 4))  # type: ignore

        # Instrument wide options
        self.event_time = UnitsWidget(
            time_units,
            default_unit="ms",
            base_value_min=0.0,
            base_value_max=10.0,
            sigfigs=sf,
        )
        self.event_time.setReadOnly(True)

        self.uptake = UnitsWidget(uptake_units, default_unit="ml/min")
        self.efficiency = ValueWidget(min=0.0, max=1.0, sigfigs=sf)

        self.event_time.setToolTip(
            "ICP-MS time per event, updated from imported files if time column exists."
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
            self.efficiency_method.setItemData(
                i, tooltip, QtCore.Qt.ItemDataRole.ToolTipRole
            )
        self.efficiency_method.setToolTip(
            self.efficiency_method.currentData(QtCore.Qt.ItemDataRole.ToolTipRole)
        )
        self.efficiency_method.currentIndexChanged.connect(
            lambda i: self.efficiency_method.setToolTip(
                self.efficiency_method.itemData(i, QtCore.Qt.ItemDataRole.ToolTipRole)
            )
        )

        # Complete Changed
        self.uptake.baseValueChanged.connect(self.optionsChanged)
        self.efficiency.valueChanged.connect(self.optionsChanged)

        form_layout = QtWidgets.QFormLayout()
        form_layout.addRow("Uptake:", self.uptake)
        form_layout.addRow("Event time:", self.event_time)
        form_layout.addRow("Trans. Efficiency:", self.efficiency)
        form_layout.addRow("", self.efficiency_method)

        # layout = QtWidgets.QVBoxLayout()
        # layout.addLayout(form_layout)

        widget = QtWidgets.QWidget()
        widget.setLayout(form_layout)
        self.setWidget(widget)

    def asInstrumentOptions(self) -> SPCalInstrumentOptions:
        return SPCalInstrumentOptions(
            self.event_time.baseValue(),
            self.uptake.baseValue(),
            self.efficiency.value(),
        )

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
        method = self.efficiency_method.currentText()
        if method == "Manual Input":
            return all(
                [
                    self.event_time.hasAcceptableInput(),
                    self.uptake.hasAcceptableInput(),
                    self.efficiency.hasAcceptableInput(),
                ]
            )
        elif method == "Reference Particle":
            return all(
                [
                    self.event_time.hasAcceptableInput(),
                    self.uptake.hasAcceptableInput(),
                ]
            )
        elif method == "Mass Response":
            return all(
                [
                    self.event_time.hasAcceptableInput(),
                ]
            )
        else:
            raise ValueError(f"Unknown method {method}.")

    def resetInputs(self) -> None:
        self.blockSignals(True)
        self.uptake.setValue(None)
        self.event_time.setValue(None)
        self.efficiency.setValue(None)
        self.blockSignals(False)
        self.optionsChanged.emit()

    def setSignificantFigures(self, num: int) -> None:
        for widget in self.findChildren(ValueWidget):
            widget.setSigFigs(num)
