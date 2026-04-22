from spcal.gui.io import NP_FILE_FILTERS

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets


class MissingPathsDialog(QtWidgets.QDialog):
    MAX_SEARCH_DEPTH = 5

    def __init__(
        self,
        missing_paths: list[Path],
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Update Missing Path(s)")
        self.setMinimumWidth(600)

        self.list = QtWidgets.QListWidget()
        for path in missing_paths:
            item = QtWidgets.QListWidgetItem()
            item.setData(QtCore.Qt.ItemDataRole.UserRole, path)
            item.setText(str(path))
            item.setIcon(self.iconForPath(path))
            self.list.addItem(item)

        self.list.itemDoubleClicked.connect(self.dialogForPath)
        self.list.itemChanged.connect(self.completeChanged)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )

        self.search_button = QtWidgets.QPushButton("Search...")
        self.search_button.pressed.connect(self.dialogSearch)
        self.button_box.addButton(
            self.search_button, QtWidgets.QDialogButtonBox.ButtonRole.ResetRole
        )

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel("Double-click to set a new path..."), 0)
        layout.addWidget(self.list, 1)
        layout.addWidget(self.button_box, 0)

        self.setLayout(layout)
        self.completeChanged()

    def completeChanged(self):
        self.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(
            self.isComplete()
        )

    def isComplete(self) -> bool:
        for path in self.newPaths():
            if path.exists():
                return True
        return False

    def iconForPath(self, path: Path) -> QtGui.QIcon:
        if path.exists():
            return QtGui.QIcon.fromTheme("dialog-ok")
        return QtGui.QIcon.fromTheme("dialog-warning")

    def originalPaths(self) -> list[Path]:
        return [
            Path(self.list.item(i).data(QtCore.Qt.ItemDataRole.UserRole))
            for i in range(self.list.count())
        ]

    def newPaths(self) -> list[Path]:
        return [Path(self.list.item(i).text()) for i in range(self.list.count())]

    def dialogForPath(self, item: QtWidgets.QListWidgetItem):
        path = Path(item.text())
        file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select New Path", str(path.parent), NP_FILE_FILTERS
        )
        if file == "":
            return

        item.setText(file)
        item.setIcon(self.iconForPath(Path(file)))

    def dialogSearch(self):
        dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Directory to search",
            str(self.originalPaths()[0].parent),
        )
        if dir != "":
            self.searchForPaths(Path(dir))

    def searchForPaths(self, root: Path):
        original_names = [p.name for p in self.originalPaths()]

        for path, dirs, filenames in root.walk():
            depth = len(path.relative_to(root).parents)
            if depth > MissingPathsDialog.MAX_SEARCH_DEPTH:
                dirs.clear()
            for filename in filenames:
                if filename in original_names:
                    idx = original_names.index(filename)
                    self.list.item(idx).setText(str(path.joinpath(filename)))
                    self.list.item(idx).setIcon(
                        self.iconForPath(path.joinpath(filename))
                    )
            for filename in dirs:
                if filename in original_names:
                    print(filename, original_names)
                    idx = original_names.index(filename)
                    self.list.item(idx).setText(str(path.joinpath(filename)))
                    self.list.item(idx).setIcon(
                        self.iconForPath(path.joinpath(filename))
                    )

    @staticmethod
    def getMissingPaths(parent: QtWidgets.QWidget, paths: list[Path]) -> list[Path]:
        dlg = MissingPathsDialog(paths, parent)
        dlg.exec()
        if dlg.result() == QtWidgets.QDialog.DialogCode.Accepted:
            return dlg.newPaths()
        return []


