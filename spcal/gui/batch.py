import logging
from pathlib import Path
from types import TracebackType
from typing import TextIO

import h5py
import numpy as np
import numpy.lib.recfunctions as rfn
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.cluster import agglomerative_cluster, prepare_data_for_clustering
from spcal.detection import accumulate_detections, combine_detections
from spcal.gui.dialogs.calculator import CalculatorDialog
from spcal.gui.inputs import ReferenceWidget, SampleWidget
from spcal.gui.io import get_open_spcal_paths, is_spcal_path
from spcal.gui.options import OptionsWidget
from spcal.gui.results import ResultsWidget
from spcal.gui.util import Worker
from spcal.gui.widgets import AdjustingTextEdit, UnitsWidget
from spcal.io.nu import read_nu_directory, select_nu_signals
from spcal.io.text import export_single_particle_results, read_single_particle_file
from spcal.io.tofwerk import read_tofwerk_file
from spcal.limit import SPCalLimit
from spcal.result import ClusterFilter, Filter, SPCalResult
from spcal.siunits import mass_units, molar_concentration_units, size_units, time_units

logger = logging.getLogger(__name__)


def process_data(
    path: Path,
    data: np.ndarray,
    expr: dict[str, str],
    method: str,
    inputs: dict[str, dict[str, float | None]],
    filters: list[list[Filter]],
    cluster_filters: list[ClusterFilter],
    limit_method: str,
    acc_method: str,
    points_req: int,
    compositions_params: dict,
    limit_params: dict[str, dict],
    limit_window_size: int = 0,
    limit_iterations: int = 1,
) -> tuple[dict[str, SPCalResult], dict[str, np.ndarray], np.ndarray]:
    # === Add any valid expressions
    data = CalculatorDialog.reduceForData(data, expr)

    # === Calculate Limits ===
    limits: dict[str, SPCalLimit] = {}
    d, l, r = {}, {}, {}
    assert data.dtype.names is not None
    for name in data.dtype.names:
        if limit_method == "Manual Input":
            limits[name] = SPCalLimit(
                np.mean(data[name]),
                inputs[name]["limit"] or np.inf,
                name="Manual Input",
                params={},
            )
        else:
            limits[name] = SPCalLimit.fromMethodString(
                limit_method,
                data[name],
                compound_kws=limit_params["compound_kws"],
                gaussian_kws=limit_params["gaussian_kws"],
                poisson_kws=limit_params["poisson_kws"],
                window_size=limit_window_size,
                max_iters=limit_iterations,
            )

        # === Create detections ===
        d[name], l[name], r[name] = accumulate_detections(
            data[name],
            limits[name].accumulationLimit(acc_method),
            limits[name].detection_threshold,
            points_required=points_req,
            integrate=True,
        )

    detections, labels, regions = combine_detections(d, l, r)

    times = next(iter(inputs.values()))["dwelltime"] * (
        regions[:, 0] + (regions[:, 1] - regions[:, 0]) / 2.0
    )

    if method in ["Manual Input", "Reference Particle"]:
        calibration_mode = "efficiency"
    elif method == "Mass Response":
        calibration_mode = "mass response"
    else:
        raise ValueError("unknown calibration mode")

    results: dict[str, SPCalResult] = {}
    for name in detections.dtype.names:
        try:
            results[name] = SPCalResult(
                path,
                data[name],
                detections[name],
                labels,
                limits[name],
                calibration_mode=calibration_mode,
            )
        except ValueError:
            pass

    # === Calculate results ===
    for name, result in results.items():
        assert inputs[name]["dwelltime"] is not None
        inputs[name]["time"] = result.events * inputs[name]["dwelltime"]  # type: ignore
        # No None inputs
        result.inputs.update({k: v for k, v in inputs[name].items() if v is not None})

    # Filter results
    if len(filters) > 0:
        filter_indicies = Filter.filter_results(filters, results)
        for name in results:
            indicies = results[name].indicies
            results[name]._indicies = np.intersect1d(
                indicies, filter_indicies, assume_unique=True
            )

    # Cluster results
    clusters = {}
    valid = SPCalResult.all_valid_indicies(list(results.values()))
    for key in ["signal", "mass", "size", "cell_concentration"]:
        rdata = {}
        for name, result in results.items():
            if result.canCalibrate(key):
                rdata[name] = result.calibrated(key, use_indicies=False)[valid]
        if len(rdata) == 0:
            continue
        X = prepare_data_for_clustering(rdata)
        T = agglomerative_cluster(X, compositions_params["distance"])
        clusters[key] = T

    # Filter clusters
    if len(cluster_filters) > 0:
        filter_indicies = ClusterFilter.filter_clusters(cluster_filters, clusters)

        valid = SPCalResult.all_valid_indicies(list(results.values()))
        for name in results:
            indicies = results[name].indicies
            results[name]._indicies = np.intersect1d(
                indicies, valid[filter_indicies], assume_unique=True
            )

    return results, clusters, times


def process_text_file(
    path: Path,
    outpath: Path,
    import_options: dict,
    trim: tuple[int, int],
    process_kws: dict,
    output_kws: dict,
) -> None:
    data = read_single_particle_file(
        path,
        delimiter=import_options["delimiter"],
        columns=import_options["columns"],
        first_line=import_options["first line"],
        convert_cps=import_options["dwelltime"] if import_options["cps"] else None,
    )
    data = rfn.rename_fields(data, import_options["names"])

    data = data[trim[0] : data.size - trim[1]]
    if data.size == 0:
        raise ValueError("data size zero")

    results, clusters, times = process_data(path, data, **process_kws)

    # === Export to file ===
    export_single_particle_results(outpath, results, clusters, times, **output_kws)


def process_nu_file(
    path: Path,
    outpath: Path,
    import_options: dict,
    trim: tuple[int, int],
    process_kws: dict,
    output_kws: dict,
) -> None:
    masses, signals, info = read_nu_directory(
        path,
        autoblank=import_options["blanking"],
        cycle=import_options["cycle"],
        segment=import_options["segment"],
    )

    selected_masses = {
        f"{i['Symbol']}{i['Isotope']}": i["Mass"] for i in import_options["isotopes"]
    }
    data = select_nu_signals(masses, signals, selected_masses=selected_masses)
    data = rfn.rename_fields(data, import_options["names"])

    data = data[trim[0] : data.size - trim[1]]
    if data.size == 0:
        raise ValueError("data size zero")

    results, clusters, times = process_data(path, data, **process_kws)

    # === Export to file ===
    export_single_particle_results(outpath, results, clusters, times, **output_kws)


def process_tofwerk_file(
    path: Path,
    outpath: Path,
    import_options: dict,
    trim: tuple[int, int],
    process_kws: dict,
    output_kws: dict,
) -> None:
    with h5py.File(path, "r") as h5:
        peak_labels = h5["PeakData"]["PeakTable"]["label"].astype("U256")
    selected_labels = [
        f"[{i['Isotope']}{i['Symbol']}]+" for i in import_options["isotopes"]
    ]
    selected_labels.extend(import_options["other peaks"])
    selected_idx = np.flatnonzero(np.isin(peak_labels, selected_labels))

    data, info, dwell = read_tofwerk_file(path, idx=selected_idx)
    data = rfn.rename_fields(data, import_options["names"])

    data = data[trim[0] : data.size - trim[1]]
    if data.size == 0:
        raise ValueError("data size zero")

    results, clusters, times = process_data(path, data, **process_kws)

    # === Export to file ===
    export_single_particle_results(outpath, results, clusters, times, **output_kws)


class ImportOptionsWidget(QtWidgets.QGroupBox):
    def __init__(self, options: dict, parent: QtWidgets.QWidget | None = None):
        super().__init__("Import Options", parent)

        self.ignores = ["importer", "path", "old names", "masses"]
        self.options = options
        self.shown = False

        layout = QtWidgets.QFormLayout()

        for key in options.keys():
            if key in self.ignores:
                continue
            layout.addRow(f"{key}:", self.widgetForKey(key))

        self.setLayout(layout)

    def widgetForKey(self, key: str) -> QtWidgets.QWidget:
        value = self.options[key]
        if key == "dwelltime":
            widget = UnitsWidget(time_units, base_value=value)
            widget.setBestUnit()
        elif key == "isotopes":
            widget = AdjustingTextEdit(
                ", ".join(f"{x['Symbol']}{x['Number']}" for x in value)
            )
        elif isinstance(value, list) and len(value) > 1:
            widget = AdjustingTextEdit(", ".join(str(x) for x in value))
        else:
            widget = QtWidgets.QLineEdit(str(value).strip("[]"))
        widget.setReadOnly(True)
        return widget


class BatchProcessDialog(QtWidgets.QDialog):
    fileprocessed = QtCore.Signal()
    processingStarted = QtCore.Signal()
    processingFinshed = QtCore.Signal()

    def __init__(
        self,
        files: list[str],
        sample: SampleWidget,
        reference: ReferenceWidget,
        options: OptionsWidget,
        results: ResultsWidget,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Batch Process")
        self.setMinimumSize(640, 640)
        self.setAcceptDrops(True)

        self.sample = sample
        self.reference = reference
        self.options = options
        self.results = results

        self.button_files = QtWidgets.QPushButton("Open Files")
        self.button_files.pressed.connect(self.dialogLoadFiles)
        self.button_output = QtWidgets.QPushButton("Open Directory")
        self.button_output.pressed.connect(self.dialogOpenOutputDir)

        self.button_process = QtWidgets.QPushButton("Start Batch")
        self.button_process.setEnabled(len(files) > 0)
        self.button_process.pressed.connect(self.buttonProcess)

        self.trim_left = QtWidgets.QCheckBox("Use sample left trim.")
        self.trim_left.setChecked(True)
        self.trim_right = QtWidgets.QCheckBox("Use sample right trim.")
        self.trim_right.setChecked(True)

        self.check_single_file = QtWidgets.QCheckBox("Export to single file.")
        self.check_single_file.setToolTip("Output all results to a single file.")
        self.check_single_file.checkStateChanged.connect(self.singleFileModeChanged)

        self.progress = QtWidgets.QProgressBar()
        self.threadpool = QtCore.QThreadPool()
        self.threadpool.setMaxThreadCount(1)  # No advantage multi?
        self.aborted = False
        self.running = False

        self.files = QtWidgets.QListWidget()
        self.files.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.files.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.files.addItems(files)
        self.files.setTextElideMode(QtCore.Qt.ElideLeft)
        self.files.model().rowsInserted.connect(self.completeChanged)
        self.files.model().rowsRemoved.connect(self.completeChanged)

        self.inputs = QtWidgets.QGroupBox("Batch Options")
        self.inputs.setLayout(QtWidgets.QFormLayout())

        self.import_options = ImportOptionsWidget(self.sample.import_options)

        self.output_dir = QtWidgets.QLineEdit("")
        self.output_dir.setPlaceholderText("Same as input")
        self.output_dir.setToolTip("Leave blank to use the input directory.")
        self.output_dir.textChanged.connect(self.completeChanged)

        self.output_name = QtWidgets.QLineEdit("%_results.csv")
        self.output_name.setToolTip("Use '%' to represent the input file name.")
        self.output_name.textChanged.connect(self.completeChanged)

        self.inputs.layout().addRow("Output Name:", self.output_name)
        self.inputs.layout().addRow("Output Directory:", self.output_dir)
        self.inputs.layout().addWidget(self.button_output)
        self.inputs.layout().addRow(self.trim_left)
        self.inputs.layout().addRow(self.trim_right)
        self.inputs.layout().addRow(self.check_single_file)

        best_units = self.results.bestUnitsForResults()
        _units = {"mass": "kg", "size": "m", "cell_concentration": "mol/L"}
        _units.update({k: v[0] for k, v in best_units.items()})

        self.mass_units = QtWidgets.QComboBox()
        self.mass_units.addItems(mass_units.keys())
        self.mass_units.setCurrentText(_units["mass"])
        self.size_units = QtWidgets.QComboBox()
        self.size_units.addItems(size_units.keys())
        self.size_units.setCurrentText(_units["size"])
        self.conc_units = QtWidgets.QComboBox()
        self.conc_units.addItems(molar_concentration_units.keys())
        self.conc_units.setCurrentText(_units["cell_concentration"])

        self.check_export_inputs = QtWidgets.QCheckBox("Export options and inputs.")
        self.check_export_inputs.setChecked(True)
        self.check_export_arrays = QtWidgets.QCheckBox(
            "Export detected particle arrays."
        )
        self.check_export_arrays.setChecked(True)
        self.check_export_compositions = QtWidgets.QCheckBox(
            "Export peak compositions."
        )

        outputs = QtWidgets.QGroupBox("Output Options")
        outputs.setLayout(QtWidgets.QFormLayout())
        outputs.layout().addRow("Mass units", self.mass_units)
        outputs.layout().addRow("Size units", self.size_units)
        outputs.layout().addRow("Conc. units", self.conc_units)
        outputs.layout().addRow(self.check_export_inputs)
        outputs.layout().addRow(self.check_export_arrays)
        outputs.layout().addRow(self.check_export_compositions)

        layout = QtWidgets.QGridLayout()
        layout.addWidget(
            self.button_files, 0, 0, 1, 2, QtCore.Qt.AlignmentFlag.AlignLeft
        )

        layout.addWidget(self.files, 1, 0, 1, 1)
        layout.addWidget(self.inputs, 1, 1, 1, 1, QtCore.Qt.AlignmentFlag.AlignTop)

        layout.addWidget(outputs, 2, 0, 1, 1)
        layout.addWidget(
            self.import_options, 2, 1, 1, 1, QtCore.Qt.AlignmentFlag.AlignTop
        )

        layout.addWidget(self.progress, 3, 0, 1, 2)
        layout.addWidget(
            self.button_process, 4, 0, 1, 2, QtCore.Qt.AlignmentFlag.AlignRight
        )

        # layout.setRowStretch(1, 1)

        self.setLayout(layout)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:  # pragma: no cover
            super().dragEnterEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if is_spcal_path(url.toLocalFile()):
                    self.files.addItem(url.toLocalFile())
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() in [QtCore.Qt.Key_Backspace, QtCore.Qt.Key_Delete]:
            items = self.files.selectedIndexes()
            for item in reversed(sorted(items)):
                self.files.model().removeRow(item.row())
        else:
            super().keyPressEvent(event)

    def completeChanged(self) -> None:
        complete = self.isComplete()
        self.button_process.setEnabled(complete)

    def isComplete(self) -> bool:
        if self.files.count() == 0:
            return False
        if "%" not in self.output_name.text():
            return False
        if any(x in self.output_name.text() for x in "<>:/\\|?*"):
            return False
        if self.output_dir.text() != "" and not Path(self.output_dir.text()).is_dir():
            return False

        return True

    def singleFileModeChanged(self, state: QtCore.Qt.CheckState) -> None:
        is_single_file = state == QtCore.Qt.CheckState.Checked
        self.check_export_inputs.setEnabled(not is_single_file)
        self.check_export_arrays.setEnabled(not is_single_file)
        self.check_export_compositions.setEnabled(not is_single_file)

    def dialogLoadFiles(self) -> None:
        paths = get_open_spcal_paths(self, "Batch Process Files")
        if len(paths) > 0:
            self.files.addItems([str(p) for p in paths])

    def dialogOpenOutputDir(self) -> None:
        dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Output Directory", "", QtWidgets.QFileDialog.ShowDirsOnly
        )
        if dir != "":
            self.output_dir.setText(dir)

    def outputsForFiles(self, files: list[Path]) -> list[Path]:
        outdir = self.output_dir.text()

        outputs = []
        for file in files:
            outname = self.output_name.text().replace("%", file.stem)
            if outdir == "":
                outdir = file.parent
            outputs.append(Path(outdir, outname))

        return outputs

    def buttonProcess(self) -> None:
        if self.running:
            self.abort()
        else:
            self.start()

    def abort(self) -> None:
        self.threadpool.clear()
        self.threadpool.waitForDone()

        self.progress.reset()

        self.aborted = True
        self.button_process.setText("Start Batch")
        self.running = False

    def start(self) -> None:
        infiles = [Path(self.files.item(i).text()) for i in range(self.files.count())]
        outfiles = self.outputsForFiles(infiles)

        # Check for overwrites
        if any(of.exists() for of in outfiles):
            exist = ", ".join(of.name for of in outfiles if of.exists())
            button = QtWidgets.QMessageBox.warning(
                self,
                "Overwrite File(s)?",
                f"The file(s) '{exist}' already exist, do you want to overwrite them?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if button == QtWidgets.QMessageBox.StandardButton.No:
                return

        self.aborted = False
        self.button_process.setText("Cancel Batch")
        self.running = True

        self.progress.setValue(0)
        self.progress.setMaximum(len(infiles))

        method = self.options.efficiency_method.currentText()

        inputs: dict[str, dict[str, float | None]] = {}

        assert self.sample.responses.dtype.names is not None
        trim = self.sample.trimRegion(self.sample.responses.dtype.names[0])
        # trims converted to left, -right
        trim = trim[0], self.sample.responses.size - trim[1]
        if not self.trim_left.isChecked():
            trim = 0, trim[1]
        if not self.trim_right.isChecked():
            trim = trim[0], 0

        for name in self.sample.responses.dtype.names:
            inputs[name] = {
                "dwelltime": self.options.dwelltime.baseValue(),
                "uptake": self.options.uptake.baseValue(),
                "cell_diameter": self.options.celldiameter.baseValue(),
                "molar_mass": self.sample.io[name].molarmass.baseValue(),
                "density": self.sample.io[name].density.baseValue(),
                "response": self.sample.io[name].response.baseValue(),
            }
            # Read the limit if manual input
            if self.options.limit_method.currentText() == "Manual Input":
                inputs[name]["limit"] = self.sample.io[name].lod_count.value()

            inputs[name]["mass_fraction"] = self.sample.io[name].massfraction.value()
            try:
                if method == "Manual Input":
                    inputs[name]["efficiency"] = self.options.efficiency.value()
                elif method == "Reference Particle":
                    inputs[name]["efficiency"] = self.reference.getEfficiency(name)
                elif method == "Mass Response":
                    inputs[name]["mass_response"] = self.reference.io[
                        name
                    ].massresponse.baseValue()
            except ValueError:
                pass

        limit_params = {
            "compound_kws": self.options.compound_poisson.state(),
            "poisson_kws": self.options.poisson.state(),
            "gaussian_kws": self.options.gaussian.state(),
        }
        if not limit_params["compound_kws"]["simulate"]:
            limit_params["compound_kws"]["single ion"] = None
        units = {
            "mass": (
                self.mass_units.currentText(),
                mass_units[self.mass_units.currentText()],
            ),
            "size": (
                self.size_units.currentText(),
                size_units[self.size_units.currentText()],
            ),
            "conc": (
                self.conc_units.currentText(),
                molar_concentration_units[self.conc_units.currentText()],
            ),
        }
        match self.sample.import_options["importer"]:
            case "text":
                fn = process_text_file
            case "nu":
                fn = process_nu_file
            case "tofwerk":
                fn = process_tofwerk_file
            case default:
                raise ValueError(f"start: no exporter for importer '{default}'")

        for path, outpath in zip(infiles, outfiles):
            worker = Worker(
                fn,
                path,
                outpath,
                import_options=self.sample.import_options,
                trim=trim,
                process_kws={
                    "expr": self.sample.current_expr,
                    "method": method,
                    "inputs": inputs,
                    "filters": self.results.filters,
                    "cluster_filters": self.results.cluster_filters,
                    "limit_method": self.options.limit_method.currentText(),
                    "acc_method": self.options.limit_accumulation,
                    "points_req": self.options.points_required,
                    "limit_params": limit_params.copy(),
                    "limit_window_size": (
                        (self.options.window_size.value() or 0)
                        if self.options.window_size.isEnabled()
                        else 0
                    ),
                    "compositions_params": {
                        "distance": self.results.graph_options["composition"][
                            "distance"
                        ]
                    },
                },
                output_kws={
                    "units_for_results": units,
                    "single_file": self.check_single_file.isChecked(),
                    "output_inputs": self.check_export_inputs.isChecked(),
                    "output_compositions": self.check_export_compositions.isChecked(),
                    "output_arrays": self.check_export_arrays.isChecked(),
                },
            )
            worker.setAutoDelete(True)
            worker.signals.finished.connect(self.workerComplete)
            worker.signals.exception.connect(self.workerFailed)
            self.threadpool.start(worker)

        logger.info(f"Batch processing started for {len(infiles)} files.")
        self.processingStarted.emit()

    def finalise(self) -> None:
        self.threadpool.waitForDone()
        self.running = False

        self.progress.reset()
        self.button_process.setText("Start Batch")

        logger.info("Batch processing complete.")
        self.processingFinshed.emit()

    def workerComplete(self) -> None:
        if self.aborted:
            return

        self.progress.setValue(self.progress.value() + 1)
        if self.progress.value() == self.progress.maximum() and self.running:
            self.finalise()
        # if self.threadpool.activeThreadCount() == 0 and self.running:

    def workerFailed(
        self, etype: type, value: BaseException, tb: TracebackType | None = None
    ) -> None:
        if self.aborted:
            return

        self.abort()

        logger.exception("Batch exception", exc_info=(etype, value, tb))
        QtWidgets.QMessageBox.critical(self, "Batch Process Failed", str(value))
