import gzip
import shutil
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.io import NU_FILE_FILTER, most_recent_spcal_path
from spcal.io.nu import is_nu_directory, is_nu_run_info_file


class NuCompressWorker(QtCore.QObject):
    started = QtCore.Signal(int)
    progress = QtCore.Signal(int)
    finished = QtCore.Signal()

    def __init__(
        self, paths: list[Path], level: int = 6, parent: QtCore.QObject | None = None
    ):
        super().__init__(parent)
        self.paths = paths
        self.level = level

    def compressIntegFile(self, path: Path, level: int):
        tmp = path.with_name(path.name + ".gzip")
        with (
            path.open("rb") as fp,
            gzip.open(tmp, "wb", compresslevel=level) as gp,
        ):
            gp.write(fp.read())

        shutil.move(tmp, path)

    def process(self):
        self.started.emit(0)

        for i, path in enumerate(self.paths):
            if self.thread().isInterruptionRequested():
                break
            self.compressIntegFile(path, self.level)
            self.progress.emit(i)

        self.finished.emit()


class CompressSpinBox(QtWidgets.QSpinBox):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setRange(1, 9)
        self.setValue(6)

    def textFromValue(self, value: int):
        if value == 1:
            return "1 (fastest)"
        elif value == 6:
            return "6 (default)"
        elif value == 9:
            return "9 (best)"
        else:
            return super().textFromValue(value)


class NuBatchCompressorDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Nu Batch Compressor")
        self.setMinimumSize(600, 400)
        self.setAcceptDrops(True)

        self.compress_thread = QtCore.QThread(self)

        self.spinbox_level = CompressSpinBox()

        self.list = QtWidgets.QListWidget()

        self.progress = QtWidgets.QProgressBar()

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        self.open_button = QtWidgets.QPushButton("Open")
        self.open_button.setIcon(QtGui.QIcon.fromTheme("document-open"))
        self.open_button.pressed.connect(self.dialogOpenDirectory)
        # self.button_box.addButton(
        #     self.open_button, QtWidgets.QDialogButtonBox.ButtonRole.ResetRole
        # )

        gbox_options = QtWidgets.QGroupBox("Options")
        gbox_options_layout = QtWidgets.QFormLayout()
        gbox_options_layout.addRow("Compression level:", self.spinbox_level)
        gbox_options.setLayout(gbox_options_layout)

        gbox_files = QtWidgets.QGroupBox("Nu Directories")
        gbox_files_layout = QtWidgets.QVBoxLayout()
        gbox_files_layout.addWidget(self.list, 1)
        gbox_files_layout.addWidget(
            self.open_button, 0, QtCore.Qt.AlignmentFlag.AlignRight
        )
        gbox_files.setLayout(gbox_files_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(gbox_options, 0)
        layout.addWidget(gbox_files, 1)
        layout.addWidget(self.progress, 0)
        layout.addWidget(self.button_box, 0)
        self.setLayout(layout)

        self.completeChanged()

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if is_nu_run_info_file(path) or is_nu_directory(path):
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent):
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if is_nu_run_info_file(path) or is_nu_directory(path):
                self.addPath(path)
                event.accept()

    def integPaths(self) -> list[Path]:
        paths = []
        for row in range(self.list.count()):
            item = self.list.item(row)
            integs = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if integs is not None:
                paths.extend(integs)
        return paths

    def completeChanged(self):
        self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(
            self.isComplete()
        )

    def isComplete(self) -> bool:
        return len(self.integPaths()) > 0

    def isCompressed(self, integ: Path) -> bool:
        with integ.open("rb") as fp:
            return fp.read(2) == b"\x1f\x8b"

    def accept(self):
        paths = self.integPaths()

        self.worker = NuCompressWorker(paths, self.spinbox_level.value())
        self.progress.setRange(0, len(paths))

        self.worker.moveToThread(self.compress_thread)

        self.worker.started.connect(self.threadStarted)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self.threadFinished)

        self.compress_thread.started.connect(self.worker.process)
        self.compress_thread.finished.connect(self.worker.deleteLater)

        self.compress_thread.start()

    def reject(self):
        if self.compress_thread.isRunning():
            self.compress_thread.requestInterruption()
            self.compress_thread.quit()
            self.compress_thread.wait(1000)
            self.threadFinished()
            return
        super().reject()

    def threadStarted(self):
        self.progress.setValue(0)
        self.spinbox_level.setEnabled(False)
        self.open_button.setEnabled(False)
        self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(
            False
        )
        self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Close).setText(
            "Cancel"
        )

    def threadFinished(self):
        self.compress_thread.quit()
        self.compress_thread.wait(1000)

        self.progress.setValue(0)
        self.spinbox_level.setEnabled(True)
        self.open_button.setEnabled(True)
        self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(
            True
        )
        self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Close).setText(
            "Close"
        )

    def addPath(self, path: Path):
        if is_nu_run_info_file(path):
            path = path.parent

        if not is_nu_directory(path):
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Nu Directory",
                f"{path} is not a valid Nu Vitesse batch directory.",
            )
            return

        integs = sorted(path.glob("*.integ"), key=lambda p: int(p.stem))
        if len(integs) == 0:
            QtWidgets.QMessageBox.warning(
                self, "No .integ Files", f"Directory {path} has no .integ files."
            )
            return
        integs = list(filter(lambda p: not self.isCompressed(p), integs))
        if len(integs) == 0:
            QtWidgets.QMessageBox.warning(
                self,
                "Already Compressed",
                f"Nu Vitesse batch {path} has already been compressed.",
            )
            return

        item = QtWidgets.QListWidgetItem(str(path))
        item.setData(QtCore.Qt.ItemDataRole.UserRole, integs)
        item.setIcon(QtGui.QIcon.fromTheme("document-open-folder"))
        self.list.addItem(item)
        self.completeChanged()

    def dialogOpenDirectory(self):
        recent = most_recent_spcal_path() or ""
        recent = str(recent)

        file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Nu Vitesse Files", recent, NU_FILE_FILTER
        )
        if file == "":
            return
        self.addPath(Path(file).parent)


if __name__ == "__main__":
    app = QtWidgets.QApplication()
    dlg = NuBatchCompressorDialog()
    dlg.show()
    app.exec()
