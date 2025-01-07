from PySide6 import QtCore, QtWidgets


class ImageExportDialog(QtWidgets.QDialog):
    exportSettingsSelected = QtCore.Signal(QtCore.QSize, int, dict)

    def __init__(
        self,
        size: QtCore.QSize | None = None,
        dpi: int = 96,
        options: dict[str, bool] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Image Export")

        if size is None:
            size = QtCore.QSize(800, 600)

        if options is None:
            options = {"transparent background": False}

        self.spinbox_size_x = QtWidgets.QSpinBox()
        self.spinbox_size_x.setRange(100, 10000)
        self.spinbox_size_x.setValue(size.width())

        self.spinbox_size_y = QtWidgets.QSpinBox()
        self.spinbox_size_y.setRange(100, 10000)
        self.spinbox_size_y.setValue(size.height())

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

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout_size = QtWidgets.QHBoxLayout()
        layout_size.addWidget(self.spinbox_size_x, 1)
        layout_size.addWidget(QtWidgets.QLabel("x"), 0)
        layout_size.addWidget(self.spinbox_size_y, 1)

        layout_form = QtWidgets.QFormLayout()
        layout_form.addRow("Size:", layout_size)
        layout_form.addRow("DPI:", self.spinbox_dpi)

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
        size = QtCore.QSize(self.spinbox_size_x.value(), self.spinbox_size_y.value())
        options = {option: cbox.isChecked() for option, cbox in self.options.items()}
        self.exportSettingsSelected.emit(size, self.spinbox_dpi.value(), options)

        super().accept()
