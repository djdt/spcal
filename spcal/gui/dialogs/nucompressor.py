import gzip
import shutil
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.io import NU_FILE_FILTER, most_recent_spcal_path
from spcal.io.nu import is_nu_directory


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

    def process(self):
        self.started.emit(0)

        for i, path in enumerate(self.paths):
            if self.thread().isInterruptionRequested():
                break
            tmp = path.with_name(path.name + ".gzip")
            with (
                path.open("rb") as fp,
                gzip.open(tmp, "wb", compresslevel=self.level) as gp,
            ):
                gp.write(fp.read())

            shutil.move(tmp, path)
            self.progress.emit(i)

        self.finished.emit()


class NuBatchCompressor(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Nu Batch Compressor")
        self.setMinimumSize(600, 400)

        self.compress_thread = QtCore.QThread(self)

        self.spinbox_level = QtWidgets.QSpinBox()
        self.spinbox_level.setRange(1, 9)
        self.spinbox_level.setValue(6)

        self.list = QtWidgets.QListWidget()

        self.progress = QtWidgets.QProgressBar()

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        self.open_button = QtWidgets.QPushButton("Open...")
        self.open_button.setIcon(QtGui.QIcon.fromTheme("document-open"))
        self.open_button.pressed.connect(self.dialogOpenDirectory)
        self.button_box.addButton(
            self.open_button, QtWidgets.QDialogButtonBox.ButtonRole.ResetRole
        )

        gbox = QtWidgets.QGroupBox("Options")
        gbox_layout = QtWidgets.QFormLayout()
        gbox_layout.addRow("Compression level:", self.spinbox_level)
        gbox.setLayout(gbox_layout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(gbox, 0)
        layout.addWidget(self.list, 1)
        layout.addWidget(self.progress, 0)
        layout.addWidget(self.button_box, 0)
        self.setLayout(layout)

    def isCompressed(self, integ: Path) -> bool:
        with integ.open("rb") as fp:
            return fp.read(2) == b"\x1f\x8b"

    def accept(self):
        paths = []
        for row in range(self.list.count()):
            dir = Path(self.list.item(row).text())
            paths.extend(sorted(dir.glob("*.integ"), key=lambda p: int(p.stem)))

        paths = list(filter(lambda p: not self.isCompressed(p), paths))

        if len(paths) == 0:
            self.list.clear()
            return

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

    def dialogOpenDirectory(self):
        recent = most_recent_spcal_path() or ""
        recent = str(recent)

        file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Nu Vitesse Files", recent, NU_FILE_FILTER
        )
        if file == "":
            return
        parent = Path(file).parent
        if not is_nu_directory(parent):
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Nu Directory",
                f"{parent} is not a valid Nu Vitesse batch directory.",
            )
            return
        if all(self.isCompressed(x) for x in parent.glob("*.integ")):
            QtWidgets.QMessageBox.warning(
                self,
                "Already Compressed",
                f"Nu Vitesse batch {parent} has already been compressed.",
            )
            return

        item = QtWidgets.QListWidgetItem(str(Path(file).parent))
        self.list.addItem(item)


if __name__ == "__main__":
    app = QtWidgets.QApplication()
    dlg = NuBatchCompressor()
    dlg.show()
    app.exec()
