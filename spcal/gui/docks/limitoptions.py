from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.limitoptions import (
    CompoundPoissonOptions,
    GaussianOptions,
    PoissonOptions,
)
from spcal.gui.widgets import CollapsableWidget
from spcal.gui.widgets.values import ValueWidget
from spcal.processing import SPCalLimitOptions


class SPCalLimitOptionsDock(QtWidgets.QDockWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Limit Options")

        self.window_size = ValueWidget(
            1000, format=("f", 0), validator=QtGui.QIntValidator(3, 1000000)
        )
        self.window_size.setEditFormat(0, format="f")
        self.window_size.setToolTip("Size of window for moving thresholds.")
        self.window_size.setEnabled(False)
        self.check_window = QtWidgets.QCheckBox("Use window")
        self.check_window.setToolTip(
            "Calculate threhold for each point using data from surrounding points."
        )
        self.check_window.toggled.connect(self.window_size.setEnabled)

        layout_window_size = QtWidgets.QHBoxLayout()
        layout_window_size.addWidget(self.window_size, 1)
        layout_window_size.addWidget(self.check_window, 1)

        self.limit_method = QtWidgets.QComboBox()
        self.limit_method.addItems(
            [
                "Automatic",
                "Highest",
                "Compound Poisson",
                "Gaussian",
                "Poisson",
                "Manual Input",
            ]
        )
        self.limit_method.setItemData(
            0,
            "Automatically determine the best method.",
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )
        self.limit_method.setItemData(
            1,
            "Use the highest of Gaussian and Poisson.",
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )
        self.limit_method.setItemData(
            2,
            "Estimate ToF limits using a compound distribution based on the "
            "number of accumulations and the single ion distribution..",
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )
        self.limit_method.setItemData(
            3,
            "Threshold using the mean and standard deviation.",
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )
        self.limit_method.setItemData(
            4,
            "Threshold using Formula C from the MARLAP manual.",
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )
        self.limit_method.setItemData(
            5,
            "Manually define limits in the sample and reference tabs.",
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )
        self.limit_method.setToolTip(
            self.limit_method.currentData(QtCore.Qt.ItemDataRole.ToolTipRole)
        )
        self.limit_method.currentIndexChanged.connect(
            lambda i: self.limit_method.setToolTip(
                self.limit_method.itemData(i, QtCore.Qt.ItemDataRole.ToolTipRole)
            )
        )

        self.check_iterative = QtWidgets.QCheckBox("Iterative")
        self.check_iterative.setToolTip("Iteratively filter on non detections.")

        self.button_advanced_options = QtWidgets.QPushButton("Advanced Options...")
        # self.button_advanced_options.pressed.connect(self.dialogAdvancedOptions)

        self.gaussian = GaussianOptions()
        self.poisson = PoissonOptions()
        self.compound = CompoundPoissonOptions()

        gaussian_collapse = CollapsableWidget("Gaussian Limit Options")
        gaussian_collapse.setWidget(self.gaussian)
        poisson_collapse = CollapsableWidget("Poisson Limit Options")
        poisson_collapse.setWidget(self.poisson)
        compound_collapse = CollapsableWidget("Compound Poisson Limit Options")
        compound_collapse.setWidget(self.compound)

        layout = QtWidgets.QFormLayout()
        # layout.addRow(buttons_layout)

        layout.addRow("Window size:", layout_window_size)
        layout.addRow("Method:", self.limit_method)

        layout.addRow(gaussian_collapse)
        layout.addRow(poisson_collapse)
        layout.addRow(compound_collapse)
        # layout.addStretch(1)
        layout.addRow(self.button_advanced_options)
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setWidget(widget)

    def asLimitOptions(self) -> SPCalLimitOptions:
        return SPCalLimitOptions(
            method=self.limit_method.currentText().lower(),
            gaussian_kws=self.gaussian.state(),
            poisson_kws=self.poisson.state(),
            compound_poisson_kws=self.compound.state(),
            window_size=int(self.window_size.value() or 0),
            max_iterations=100 if self.check_iterative.isChecked() else 1,
        )
