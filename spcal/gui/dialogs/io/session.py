"""Class to import SPCal sessions in a separate thread."""

import json
from pathlib import Path

from PySide6 import QtCore, QtWidgets


from spcal.gui.dialogs.missingpaths import MissingPathsDialog
from spcal.io.session import decode_json_method, decode_json_datafile


class SessionImportWorker(QtCore.QObject):
    started = QtCore.Signal(int)
    progress = QtCore.Signal(int, str)
    finished = QtCore.Signal()
    datafileImported = QtCore.Signal(object)

    def __init__(
        self, datafiles: list[tuple[dict, Path]], parent: QtCore.QObject | None = None
    ):
        super().__init__(parent)
        self.datafiles = datafiles

    def read(self):
        self.started.emit(len(self.datafiles))
        for i, (datafile, path) in enumerate(self.datafiles):
            if self.thread().isInterruptionRequested():
                return

            self.progress.emit(i, path.name)
            df = decode_json_datafile(datafile, path)
            if not self.thread().isInterruptionRequested():
                self.datafileImported.emit(df)

        self.finished.emit()


class SessionImportDialog(QtWidgets.QProgressDialog):
    methodImported = QtCore.Signal(object)
    datafileImported = QtCore.Signal(object)

    def __init__(
        self, path: Path | None = None, parent: QtWidgets.QWidget | None = None
    ):
        super().__init__("Import Session", "Cancel", 0, 0, parent)
        self.setWindowTitle("Import Session")
        self.setMinimumWidth(320)

        self.import_thread = QtCore.QThread(self)

        self.canceled.connect(self.stopThread)

        if path is not None:
            self.loadSession(path)

    def loadSession(self, path: Path):

        with path.open() as fp:
            session = json.load(fp)

        self.methodImported.emit(decode_json_method(session["method"]))

        datafiles = [
            (datafile, Path(datafile["path"])) for datafile in session["datafiles"]
        ]
        if len(datafiles) == 0:
            return

        missing = sum(not p.exists() for _, p in datafiles)

        if missing > 0:
            button = QtWidgets.QMessageBox.warning(
                self,
                "Datafile Not Found",
                f"Cannot find {missing} datafiles. Select new paths?",
                buttons=QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
            )
            if button == QtWidgets.QMessageBox.StandardButton.Yes:
                new_paths = MissingPathsDialog.getMissingPaths(
                    self, [p for _, p in datafiles]
                )
                for datafile, new in zip(datafiles, new_paths):
                    datafiles[datafile] = new

        self.worker = SessionImportWorker(
            [(df, path) for df, path in datafiles if path.exists()]
        )
        self.worker.moveToThread(self.import_thread)
        self.worker.started.connect(self.setMaximum)
        self.worker.progress.connect(self.updateProgress)
        self.worker.datafileImported.connect(self.datafileImported)
        self.worker.finished.connect(self.finalise)

        self.import_thread.started.connect(self.worker.read)
        self.import_thread.finished.connect(self.worker.deleteLater)
        self.import_thread.start()

    def updateProgress(self, value: int, file: str):
        self.setValue(value)
        self.setLabelText(f"Importing {file}...")

    def stopThread(self):
        if self.import_thread.isRunning():
            self.import_thread.requestInterruption()
            self.import_thread.quit()
            self.import_thread.wait(1000)

    def finalise(self):
        self.import_thread.quit()
        self.import_thread.wait()
        self.reset()
