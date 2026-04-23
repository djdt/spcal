import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.datafile import SPCalDataFile
from spcal.isotope import SPCalIsotope
from spcal.processing.method import SPCalProcessingMethod

from spcal.gui.graphs.base import SinglePlotGraphicsView
from spcal.gui.widgets.periodictable import PeriodicTableSelector


class ScreeningOptionsDialog(QtWidgets.QDialog):
    screeningOptionsSelected = QtCore.Signal(float, int, bool)

    def __init__(
        self,
        ppm: int,
        size: int,
        replace_isotopes: bool = True,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Screening Options")
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

        self.screening_ppm = QtWidgets.QSpinBox()
        self.screening_ppm.setRange(1, 999999)
        self.screening_ppm.setValue(ppm)
        self.screening_ppm.setSingleStep(100)
        self.screening_ppm.setSuffix(" ppm")
        self.screening_ppm.setToolTip(
            "The number of detection (as particles per million events) required to pass the screen."
        )

        self.screening_size = QtWidgets.QSpinBox()
        self.screening_size.setRange(1, 100000000)
        self.screening_size.setValue(size)
        self.screening_size.setSingleStep(100000)
        self.screening_size.setToolTip(
            "Number of events to screen, a larger number takes longer."
        )

        self.checkbox_replace_isotopes = QtWidgets.QCheckBox("Replace isotopes.")
        self.checkbox_replace_isotopes.setChecked(replace_isotopes)
        self.checkbox_replace_isotopes.setToolTip(
            "Replace the current isotope selection. If not checked, adds to selection."
        )

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout()
        form_layout = QtWidgets.QFormLayout()
        form_layout.addRow("Screening Threshold", self.screening_ppm)
        form_layout.addRow("Screening Size", self.screening_size)
        form_layout.addRow(self.checkbox_replace_isotopes)
        layout.addLayout(form_layout, 1)
        layout.addWidget(self.button_box, 0)
        self.setLayout(layout)
        self.resize(300, 160)

    def accept(self):
        self.screeningOptionsSelected.emit(
            self.screening_ppm.value(),
            self.screening_size.value(),
            self.checkbox_replace_isotopes.isChecked(),
        )
        super().accept()


class SignalsPopup(QtWidgets.QDialog):
    def __init__(
        self,
        signals: dict[SPCalIsotope, np.ndarray],
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Signal Inspection")
        self.setWindowFlags(QtCore.Qt.WindowType.Popup)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(QtCore.QMargins(1, 1, 1, 1))

        for isotope, signal in signals.items():
            view = SinglePlotGraphicsView(f"{isotope}", ylabel="")
            view.plot.xaxis.setVisible(False)
            view.setInteractive(False)
            view.drawCurve(np.arange(signal.size), signal)
            layout.addWidget(view)

        self.nviews = len(signals)

        self.setLayout(layout)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(
            int(400 * self.devicePixelRatio()),
            int(100 * self.nviews * self.devicePixelRatio()),
        )


class SelectIsotopesDialog(QtWidgets.QDialog):
    isotopesSelected = QtCore.Signal(SPCalDataFile)

    def __init__(
        self,
        data_file: SPCalDataFile,
        screening_method: SPCalProcessingMethod,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Select Isotopes")
        self.setWhatsThis(
            "This dialog is used to select isotopes for processing. Select the preferred isotope (highest abundance without interferences) by left-clicking an element. Right-click allows selection of specific isotopes, while middle-click will preview the signals for an element. Holding shift while left-clicking will select all isotopes with a natural abundance over 10 %. Finally, non-target screening can be performed using the Screen button."
        )
        # self.setWindowFlag(QtCore.Qt.WindowType.WindowContextHelpButtonHint)

        self.data_file = data_file
        self.screening_method = screening_method
        self.screening_ppm = 100
        self.screening_size = 1000000
        self.screening_replace_isotopes = True

        self.table = PeriodicTableSelector(
            data_file.isotopes, data_file.selected_isotopes
        )
        self.table.requestShowIsotopes.connect(self.showIsotopes)
        self.table.isotopesChanged.connect(self.completeChanged)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
            | QtWidgets.QDialogButtonBox.StandardButton.Reset
        )
        button_screen = QtWidgets.QPushButton("Screen")
        button_screen.setIcon(QtGui.QIcon.fromTheme("edit-find"))
        self.button_box.addButton(
            button_screen, QtWidgets.QDialogButtonBox.ButtonRole.ActionRole
        )
        self.button_box.clicked.connect(self.onButtonClicked)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(
            QtWidgets.QLabel(str(data_file.path)),
            0,
            QtCore.Qt.AlignmentFlag.AlignHCenter,
        )
        layout.addWidget(self.table, 1)
        layout.addWidget(self.button_box, 0)
        self.setLayout(layout)

    def setScreeningParameters(self, ppm: int, size: int, replace_isotopes: bool):
        self.screening_ppm = ppm
        self.screening_size = size
        self.screening_replace_isotopes = replace_isotopes

    def screenDataFile(
        self, screening_target_ppm: int, screening_size: int, replace_isotopes: bool
    ):
        results = self.screening_method.processDataFile(
            self.data_file, self.data_file.preferred_isotopes, max_size=screening_size
        )
        selected_isotopes = []
        selected_numbers = []
        for isotope, result in results.items():
            if result.number > result.num_events * screening_target_ppm * 1e-6:
                selected_isotopes.append(isotope)
                selected_numbers.append(result.number)

        if len(selected_numbers) == 0:
            return
        nmax = max(selected_numbers)
        colors = [QtGui.QColor.fromRgbF(n / nmax, 0.0, 0.0) for n in selected_numbers]
        self.table.setIsotopeColors(selected_isotopes, colors)

        if not replace_isotopes:
            selected_isotopes = set(selected_isotopes)
            selected_isotopes.update(self.table.selectedIsotopes())
            selected_isotopes = list(selected_isotopes)
        self.table.setSelectedIsotopes(selected_isotopes)

    def showIsotopes(self, isotopes: list[SPCalIsotope]):
        popup = SignalsPopup(
            {isotope: self.data_file[isotope] for isotope in isotopes}, parent=self
        )
        popup.show()

    def onButtonClicked(self, button: QtWidgets.QAbstractButton):
        sb = self.button_box.standardButton(button)
        role = self.button_box.buttonRole(button)
        if role == QtWidgets.QDialogButtonBox.ButtonRole.ActionRole:  # screen
            self.dialogScreen()
        elif sb == QtWidgets.QDialogButtonBox.StandardButton.Reset:
            self.table.setSelectedIsotopes([])
            self.table.setIsotopeColors([], [])
        elif sb == QtWidgets.QDialogButtonBox.StandardButton.Ok:
            self.accept()
        else:  # Close
            self.reject()

    def dialogScreen(self):
        dlg = ScreeningOptionsDialog(
            self.screening_ppm,
            self.screening_size,
            self.screening_replace_isotopes,
            parent=self,
        )
        dlg.screeningOptionsSelected.connect(self.setScreeningParameters)
        dlg.screeningOptionsSelected.connect(self.screenDataFile)
        dlg.exec()

    def completeChanged(self):
        complete = self.isComplete()
        self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(
            complete
        )

    def isComplete(self) -> bool:
        return len(self.table.selectedIsotopes()) > 0

    def accept(self):
        self.data_file.selected_isotopes = self.table.selectedIsotopes()
        self.isotopesSelected.emit(self.data_file)
        super().accept()
