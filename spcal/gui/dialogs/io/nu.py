import json
import logging
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.calc import search_sorted_closest
from spcal.datafile import SPCalDataFile, SPCalNuDataFile
from spcal.gui.dialogs.io.base import ImportDialogBase
from spcal.gui.widgets.periodictable import PeriodicTableSelector
from spcal.io import nu
from spcal.isotope import ISOTOPE_TABLE
from spcal.processing.method import SPCalProcessingMethod

logger = logging.getLogger(__name__)


class NuIntegReadWorker(QtCore.QObject):
    started = QtCore.Signal(int)
    progress = QtCore.Signal(int)
    integRead = QtCore.Signal(object)
    finished = QtCore.Signal()

    def __init__(
        self,
        path: Path,
        index: list[dict],
        cyc_number: int | None,
        seg_number: int | None,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent)
        self.path = path
        self.index = index
        self.cyc_number = cyc_number
        self.seg_number = seg_number

    @QtCore.Slot()
    def read(self):
        self.started.emit(len(self.index))
        for i, idx in enumerate(self.index):
            if self.thread().isInterruptionRequested():
                return

            binary_path = self.path.joinpath(f"{idx['FileNum']}.integ")
            if binary_path.exists():
                data = nu.read_integ_binary(
                    binary_path,
                    idx["FirstCycNum"],
                    idx["FirstSegNum"],
                    idx["FirstAcqNum"],
                    memmap=False,
                )
                if self.cyc_number is not None:
                    data = data[data["cyc_number"] == self.cyc_number]
                if self.seg_number is not None:
                    data = data[data["seg_number"] == self.seg_number]
                self.integRead.emit(data)
                del data  # ensure cleared in thread
            else:
                logger.warning(  # pragma: no cover, missing files
                    f"collect_data_from_index: missing data file {idx['FileNum']}.integ, skipping"
                )
            self.progress.emit(i)

        self.finished.emit()


class NuImportDialog(ImportDialogBase):
    def __init__(
        self,
        path: str | Path,
        existing_file: SPCalDataFile | None = None,
        screening_method: SPCalProcessingMethod | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(path, "SPCal Nu Instruments Import", screening_method, parent)

        self.import_thread = QtCore.QThread(self)
        self.import_data: list[np.ndarray] = []

        with self.file_path.joinpath("run.info").open("r") as fp:
            self.info = json.load(fp)
        with self.file_path.joinpath("integrated.index").open("r") as fp:
            self.index = json.load(fp)
        with self.file_path.joinpath("autob.index").open("r") as fp:
            self.autob_index = json.load(fp)

        max_mass_diff = 0.05
        selected = []
        autoblanking = "regions"
        if isinstance(existing_file, SPCalNuDataFile):
            max_mass_diff = existing_file.max_mass_diff
            selected = existing_file.selected_isotopes
            autoblanking = existing_file.autoblanking

        # read first integ
        data: np.ndarray | None = None
        for idx in self.index:
            first_path = self.file_path.joinpath(f"{idx['FileNum']}.integ")
            if first_path.exists():
                data = nu.read_integ_binary(
                    first_path,
                    idx["FirstCycNum"],
                    idx["FirstSegNum"],
                    idx["FirstAcqNum"],
                    memmap=True,
                )
                break
        if data is None:
            raise ValueError("NuImportDialog: no valid integ files found.")

        self.signals = data["result"]["signal"] / self.info["AverageSingleIonArea"]
        self.masses = nu.masses_from_integ(data[0], self.info)[0]

        self.table = PeriodicTableSelector()
        self.progress = QtWidgets.QProgressBar()

        self.max_mass_diff = QtWidgets.QDoubleSpinBox()
        self.max_mass_diff.setRange(0.0, 1.0)
        self.max_mass_diff.setValue(max_mass_diff)
        self.max_mass_diff.valueChanged.connect(self.updateTableIsotopes)

        self.cycle_number = QtWidgets.QSpinBox()
        self.cycle_number.setRange(0, self.info["CyclesWritten"])
        self.cycle_number.setValue(0)
        self.cycle_number.setSpecialValueText("All")

        self.segment_number = QtWidgets.QSpinBox()
        self.segment_number.setRange(0, len(self.info["SegmentInfo"]))
        self.segment_number.setValue(0)
        self.segment_number.setSpecialValueText("All")

        self.first_integ = QtWidgets.QSpinBox()
        self.first_integ.setRange(1, len(self.index) + 1)
        self.first_integ.setValue(1)
        self.first_integ.valueChanged.connect(self.completeChanged)

        self.last_integ = QtWidgets.QSpinBox()
        self.last_integ.setRange(2, len(self.index) + 1)
        self.last_integ.setValue(len(self.index) + 1)
        self.last_integ.valueChanged.connect(self.completeChanged)

        layout_integ = QtWidgets.QHBoxLayout()
        layout_integ.addWidget(self.first_integ)
        layout_integ.addWidget(QtWidgets.QLabel("-"))
        layout_integ.addWidget(self.last_integ)

        # todo: option to remove blanked regions?
        self.combo_blanking = QtWidgets.QComboBox()
        self.combo_blanking.addItems(["Off", "Blank Regions", "Blank All Masses"])
        self.combo_blanking.setItemData(
            0,
            "Don't apply blanking, some data may be invalid.",
            role=QtCore.Qt.ItemDataRole.ToolTipRole,
        )
        self.combo_blanking.setItemData(
            1,
            "Remove mass regions recorded in the auto blanking files.",
            role=QtCore.Qt.ItemDataRole.ToolTipRole,
        )
        self.combo_blanking.setItemData(
            2,
            "Remove all masses when a region is blanked, regardless of their presence in the auto blanking files.",
            role=QtCore.Qt.ItemDataRole.ToolTipRole,
        )
        self.combo_blanking.setCurrentIndex(
            ["off", "regions", "all"].index(autoblanking)
        )

        self.updateTableIsotopes()

        self.table.setSelectedIsotopes(selected)
        self.table.isotopesChanged.connect(self.completeChanged)

        self.layout_body.addWidget(self.table, 1)
        self.layout_body.addWidget(self.progress, 0)

        # Set info and defaults
        method = self.info["MethodFile"]
        self.box_info_layout.addRow(
            "Method:", QtWidgets.QLabel(method[method.rfind("\\") + 1 :])
        )
        self.box_info_layout.addRow(
            "Events:",
            QtWidgets.QLabel(str(self.info["TotalAcquisitions"])),
        )
        self.box_info_layout.addRow(
            "Event time:",
            QtWidgets.QLabel(f"{nu.eventtime_from_info(self.info) * 1000} ms"),
        )
        self.box_info_layout.addRow(
            "Total time:",
            QtWidgets.QLabel(
                f"{nu.eventtime_from_info(self.info) * self.info['TotalAcquisitions']:.2f} s"
            ),
        )
        self.box_info_layout.addRow(
            "Integrations:",
            QtWidgets.QLabel(str(len(self.info["IntegrationRegions"]))),
        )

        self.box_options_layout.addRow("Cycle:", self.cycle_number)
        self.box_options_layout.addRow("Segment:", self.segment_number)
        self.box_options_layout.addRow("Integ files:", layout_integ)
        self.box_options_layout.addRow("Max diff m/z:", self.max_mass_diff)
        # self.box_options.layout().addRow("Max file:", self.file_number)
        self.box_options_layout.addRow("Auto blanking:", self.combo_blanking)

        self.table.setFocus()
        self.completeChanged()

    def updateTableIsotopes(self):
        natural_isotopes = [
            iso for iso in ISOTOPE_TABLE.values() if iso.composition is not None
        ]
        natural_masses = np.fromiter(
            (iso.mass for iso in natural_isotopes), dtype=float
        )
        indices = search_sorted_closest(self.masses, natural_masses)
        valid = (
            np.abs(self.masses[indices] - natural_masses) < self.max_mass_diff.value()
        )
        self.table.setEnabledIsotopes(
            [iso for iso, v in zip(natural_isotopes, valid) if v]
        )

    def screenDataFile(self, screening_target_ppm: int, screening_size: int):
        if self.screening_method is None:
            return

        first = self.first_integ.value() - 1
        while (
            first < len(self.index)
            and self.index[first]["FirstCycNum"] < self.cycle_number.value()
            and self.index[first]["FirstSegNum"] < self.segment_number.value()
        ):
            first += 1
        first_acq_number = self.index[first]["FirstAcqNum"]

        last = first + 1
        while (
            last < len(self.index)
            and self.index[last]["FirstAcqNum"] - first_acq_number < screening_size
        ):
            last += 1

        data_file = SPCalNuDataFile.load(
            self.file_path,
            self.max_mass_diff.value(),
            cycle_number=self.cycle_number.value() or None,
            segment_number=self.segment_number.value() or None,
            first_integ_file=first,
            last_integ_file=last,
        )

        results = self.screening_method.processDataFile(
            data_file, data_file.preferred_isotopes, max_size=screening_size
        )
        selected_isotopes = []
        selected_numbers = []
        for isotope, result in results.items():
            if result.number > result.num_events * screening_target_ppm * 1e-6:
                selected_isotopes.append(isotope)
                selected_numbers.append(result.number)

        if len(selected_numbers) == 0:
            return

        # vals = np.log(
        #     1.0 + np.array(selected_numbers) / np.amax(selected_numbers)
        # ) / np.log(2.0)

        nmax = np.amax(selected_numbers)
        colors = [QtGui.QColor.fromRgbF(n / nmax, 0.0, 0.0) for n in selected_numbers]
        self.table.setSelectedIsotopes(selected_isotopes)
        self.table.setIsotopeColors(selected_isotopes, colors)

    def isComplete(self) -> bool:
        if self.first_integ.value() >= self.last_integ.value():
            return False
        if len(self.table.selectedIsotopes()) == 0:
            return False
        return True

    def setControlsEnabled(self, enabled: bool):
        button = self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        button.setEnabled(enabled)
        self.table.setEnabled(enabled)

    @QtCore.Slot()
    def addData(self, data: np.ndarray):
        self.import_data.append(data)

    def closeEvent(self, event: QtGui.QCloseEvent):
        self.cleanup()

    def cleanup(self):
        self.import_data.clear()

    def accept(self):
        self.setControlsEnabled(False)
        self.progress.setValue(0)

        self.import_data.clear()

        cyc_number = self.cycle_number.value()
        if cyc_number == 0:
            cyc_number = None
        seg_number = self.segment_number.value()
        if seg_number == 0:
            seg_number = None

        selected_index = self.index[
            self.first_integ.value() - 1 : self.last_integ.value() - 1
        ]
        self.worker = NuIntegReadWorker(
            self.file_path, selected_index, cyc_number, seg_number
        )
        self.worker.moveToThread(self.import_thread)
        self.worker.started.connect(self.progress.setMaximum)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.integRead.connect(self.addData)
        self.worker.finished.connect(self.finalise)

        self.import_thread.started.connect(self.worker.read)
        self.import_thread.finished.connect(self.worker.deleteLater)
        self.import_thread.start()

    def reject(self):
        if self.import_thread.isRunning():
            self.import_thread.requestInterruption()
            self.import_thread.quit()
            self.import_thread.wait()

            self.setControlsEnabled(True)
            self.progress.reset()
        else:
            self.cleanup()
            super().reject()

    @QtCore.Slot()
    def finalise(self):
        self.import_thread.quit()
        self.import_thread.wait()

        if len(self.table.selectedIsotopes()) == 0:
            raise ValueError("no selected isotopes")

        cyc_number = self.cycle_number.value()
        if cyc_number == 0:
            cyc_number = None
        seg_number = self.segment_number.value()
        if seg_number == 0:
            seg_number = None

        masses = nu.masses_from_integ(self.import_data[0], self.info)[0]
        signals = nu.signals_from_integs(self.import_data, self.info)

        times = nu.times_from_integs(self.import_data, self.info) * 1e-9

        # if not raw:
        signals /= self.info["AverageSingleIonArea"]

        autoblanking = {
            "Off": "off",
            "Blank Regions": "regions",
            "Blank All Masses": "all",
        }[self.combo_blanking.currentText()]

        if autoblanking != "off":
            autobs = np.concatenate(
                nu.read_binaries_in_index(
                    self.file_path,
                    self.autob_index,
                    "autob",
                    nu.read_autob_binary,
                    cyc_number=cyc_number,
                    seg_number=seg_number,
                )
            )
            signals = nu.apply_autoblanking(
                autobs,
                signals,
                masses,
                self.info,
                blank_all_signals=autoblanking == "all",
            )

        data_file = SPCalNuDataFile(
            self.file_path,
            signals,
            times,
            masses,
            self.info,
            cycle_number=cyc_number,
            segment_number=seg_number,
            autoblanking=autoblanking,
            integ_files=(self.first_integ.value() - 1, self.last_integ.value() - 1),
        )
        data_file.selected_isotopes = self.table.selectedIsotopes()
        self.dataImported.emit(data_file)

        self.cleanup()
        super().accept()
