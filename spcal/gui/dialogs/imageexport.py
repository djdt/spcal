from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.detection import detection_maxima
from spcal.gui.graphs import color_schemes, symbols
from spcal.gui.graphs.particle import ParticleView
from spcal.result import SPCalResult


class ImageExportDialog(QtWidgets.QDialog):
    exportSettingsSelected = QtCore.Signal(Path, QtCore.QSize, float, dict)

    def __init__(
        self,
        path: Path | None = None,
        options: dict[str, bool] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Image Export")

        if path is None:
            path = Path("image.png")
        if options is None:
            options = {"transparent background": False}

        self.path = path

        settings = QtCore.QSettings()
        size_x = int(settings.value("ImageExport/SizeX", 800))
        size_y = int(settings.value("ImageExport/SizeY", 600))
        dpi = int(settings.value("ImageExport/DPI", 96))

        self.spinbox_size_x = QtWidgets.QSpinBox()
        self.spinbox_size_x.setRange(100, 10000)
        self.spinbox_size_x.setValue(size_x)

        self.spinbox_size_y = QtWidgets.QSpinBox()
        self.spinbox_size_y.setRange(100, 10000)
        self.spinbox_size_y.setValue(size_y)

        # self.spinbox_x0 = QtWidgets.QDoubleSpinBox()
        # self.spinbox_x0.setRange(xmin, xmax)
        # self.spinbox_x0.setValue(x0)
        #
        # self.spinbox_x1 = QtWidgets.QDoubleSpinBox()
        # self.spinbox_x1.setRange(xmin, xmax)
        # self.spinbox_x1.setValue(x1)

        self.spinbox_dpi = QtWidgets.QSpinBox()
        self.spinbox_dpi.setRange(96, 1200)
        self.spinbox_dpi.setValue(dpi)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save
            | QtWidgets.QDialogButtonBox.StandardButton.Close,
        )

        self.options = {}
        for option, on in options.items():
            self.options[option] = QtWidgets.QCheckBox(option)
            self.options[option].setChecked(on)
        # self.check_transparent = QtWidgets.QCheckBox("Transparent background")

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout_size = QtWidgets.QHBoxLayout()
        layout_size.addWidget(self.spinbox_size_x, 1)
        layout_size.addWidget(QtWidgets.QLabel("x"), 0)
        layout_size.addWidget(self.spinbox_size_y, 1)

        layout_form = QtWidgets.QFormLayout()
        layout_form.addRow("Size:", layout_size)
        layout_form.addRow("DPI:", self.spinbox_dpi)
        # layout_form.addRow(self.check_transparent)

        layout_options = QtWidgets.QVBoxLayout()
        layout_options.addStretch(1)
        for option in self.options.values():
            layout_options.addWidget(option, 0)

        layout_horz = QtWidgets.QHBoxLayout()
        layout_horz.addLayout(layout_form)
        layout_horz.addLayout(layout_options)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(layout_horz)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

    def accept(self) -> None:
        path, ok = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Image", str(self.path.absolute()), "PNG Images (*.png)"
        )
        if not ok:
            return

        size = QtCore.QSize(self.spinbox_size_x.value(), self.spinbox_size_y.value())

        options = {option: cbox.isChecked() for option, cbox in self.options.items()}

        # if self.check_transparent.isChecked():
        #     background = QtCore.Qt.GlobalColor.transparent
        # else:
        #     background = QtCore.Qt.GlobalColor.white
        self.exportSettingsSelected.emit(path, size, self.spinbox_dpi.value(), options)

        settings = QtCore.QSettings()
        settings.setValue("ImageExport/SizeX", size.width())
        settings.setValue("ImageExport/SizeY", size.height())
        settings.setValue("ImageExport/DPI", self.spinbox_dpi.value())

        super().accept()
