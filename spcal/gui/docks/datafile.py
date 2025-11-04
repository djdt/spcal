from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from spcal.datafile import SPCalDataFile, SPCalTextDataFile


class DataFileDelegate(QtWidgets.QAbstractItemDelegate):
    margin = 5

    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(parent)
        style = QtWidgets.QApplication.style()
        self.close_align = QtCore.Qt.AlignmentFlag.AlignRight
        self.close_icon = style.standardIcon(
            QtWidgets.QStyle.StandardPixmap.SP_TabCloseButton
        )
        self.menu_align = (
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignBottom
        )
        self.menu_icon = style.standardIcon(
            QtWidgets.QStyle.StandardPixmap.SP_FileDialogListView
        )

    def editorEvent(
        self,
        event: QtCore.QEvent,
        model: QtCore.QAbstractItemModel,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ) -> bool:
        if event.type() == QtCore.QEvent.Type.MouseButtonRelease:
            assert isinstance(event, QtGui.QMouseEvent)
            style = (
                option.widget.style()  # type: ignore
                if option.widget is not None  # type: ignore
                else QtWidgets.QApplication.style()
            )
            frame: QtCore.QRect = option.rect.adjusted(  # type: ignore
                self.margin, self.margin, -self.margin, -self.margin
            )
            pixmap = self.close_icon.pixmap(
                style.pixelMetric(style.PixelMetric.PM_SmallIconSize)
            )
            close_rect = style.itemPixmapRect(frame, self.close_align, pixmap)
            if close_rect.contains(event.position().toPoint()):
                index.model().removeRow(index.row())
                return True
            menu_rect = style.itemPixmapRect(frame, self.menu_align, pixmap)
            if menu_rect.contains(event.position().toPoint()):
                model = index.model()
                assert isinstance(model, DataFileModel)
                model.editIsotopesRequested.emit(index)
                return True
        elif event.type() == QtCore.QEvent.Type.MouseMove:
            return True  # paint

        return super().editorEvent(event, model, option, index)

    def sizeHint(
        self,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ) -> QtCore.QSize:
        #
        return QtCore.QSize(200, 60)
        style = QtWidgets.QApplication.style()

        return style.sizeFromContents(
            style.ContentsType.CT_ItemViewItem, option, QtCore.QSize()
        )

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ):
        # data from the model
        path = index.data(DataFileModel.PathRole)
        nevents = index.data(DataFileModel.NumEventsRole)
        nisotopes = len(index.data(DataFileModel.IsotopesRole))
        nselected = len(index.data(DataFileModel.SelectedIsotopesRole))

        # get the style for drawing
        if option.widget is None:  # type: ignore , in QStyleOption
            style = QtWidgets.QApplication.style()
        else:
            style = option.widget.style()  # type: ignore , in QStyleOption

        # draw the basic item view
        style.drawControl(style.ControlElement.CE_ItemViewItem, option, painter)

        # frame for further drawing
        frame = QtCore.QRect(option.rect)  # type: ignore , in QStyleOption
        frame.adjust(self.margin, self.margin, -self.margin, -self.margin)

        # draw the close button
        pixmap_size = QtCore.QSize(
            style.pixelMetric(style.PixelMetric.PM_SmallIconSize),
            style.pixelMetric(style.PixelMetric.PM_SmallIconSize),
        )
        close_pixmap = self.close_icon.pixmap(pixmap_size, QtGui.QIcon.Mode.Disabled)
        close_rect = style.itemPixmapRect(frame, self.close_align, close_pixmap)
        if close_rect.contains(option.widget.mapFromGlobal(QtGui.QCursor.pos())):  # type: ignore
            close_pixmap = self.close_icon.pixmap(pixmap_size, QtGui.QIcon.Mode.Active)
        style.drawItemPixmap(painter, close_rect, self.close_align, close_pixmap)

        # draw the menu button
        if nisotopes > 1:
            menu_pixmap = self.menu_icon.pixmap(pixmap_size, QtGui.QIcon.Mode.Normal)
            menu_rect = style.itemPixmapRect(frame, self.menu_align, menu_pixmap)
            if menu_rect.contains(option.widget.mapFromGlobal(QtGui.QCursor.pos())):  # type: ignore
                menu_pixmap = self.menu_icon.pixmap(
                    pixmap_size, QtGui.QIcon.Mode.Selected
                )
            style.drawItemPixmap(painter, menu_rect, self.menu_align, menu_pixmap)

        frame.setRight(close_rect.left() - self.margin)

        # draw the texts
        enabled = bool(option.state & style.StateFlag.State_Enabled)  # type: ignore
        for text, alignment, bold in zip(
            [path.stem, f"Events: {nevents}", f"Isotopes: {nselected} / {nisotopes}"],
            [
                QtCore.Qt.AlignmentFlag.AlignLeft,
                QtCore.Qt.AlignmentFlag.AlignBottom,
                QtCore.Qt.AlignmentFlag.AlignRight
                | QtCore.Qt.AlignmentFlag.AlignBottom,
            ],
            [True, False, False],
        ):
            painter.save()
            if bold:
                font = painter.font()
                font.setBold(True)
                painter.setFont(font)

            rect = style.itemTextRect(
                painter.fontMetrics(),
                frame,
                alignment,
                enabled,
                text,  # type: ignore
            )
            style.drawItemText(
                painter,
                rect,
                alignment,
                option.palette,  # type: ignore
                enabled,
                text,
            )
            painter.restore()


class DataFileModel(QtCore.QAbstractListModel):
    DataFileRole = QtCore.Qt.ItemDataRole.UserRole
    PathRole = QtCore.Qt.ItemDataRole.UserRole + 1
    IsotopesRole = QtCore.Qt.ItemDataRole.UserRole + 2
    SelectedIsotopesRole = QtCore.Qt.ItemDataRole.UserRole + 3
    NumEventsRole = QtCore.Qt.ItemDataRole.UserRole + 4

    editIsotopesRequested = QtCore.Signal(QtCore.QModelIndex)

    def __init__(
        self,
        data_files: list[SPCalDataFile] | None = None,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent)
        if data_files is None:
            data_files = []

        self.data_files = data_files

    def removeRows(
        self,
        row: int,
        count: int,
        parent: QtCore.QModelIndex
        | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),
    ) -> bool:
        self.beginRemoveRows(parent, row, row + count)
        if row < 0 or row + count > self.rowCount():
            return False
        for i in range(row, row + count):
            self.data_files.pop(row)
        self.endRemoveRows()
        return True

    def rowCount(
        self,
        parent: QtCore.QModelIndex
        | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),
    ) -> int:
        return len(self.data_files)

    def data(
        self,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if not index.isValid():
            return None

        data_file = self.data_files[index.row()]

        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            return f"{data_file.path.stem} :: {data_file.num_events} events, {len(data_file.isotopes)} isotopes"
        elif role == DataFileModel.DataFileRole:
            return data_file
        elif role == DataFileModel.PathRole:
            return data_file.path
        elif role == DataFileModel.IsotopesRole:
            return data_file.isotopes
        elif role == DataFileModel.SelectedIsotopesRole:
            return data_file.selected_isotopes
        elif role == DataFileModel.NumEventsRole:
            return data_file.num_events


class SPCalDataFilesDock(QtWidgets.QDockWidget):
    dataFileChanged = QtCore.Signal(SPCalDataFile)
    dataFileAdded = QtCore.Signal(SPCalDataFile)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Open Data Files")

        self.list = QtWidgets.QListView()
        self.list.setMouseTracking(True)
        self.list.setSelectionMode(QtWidgets.QListView.SelectionMode.NoSelection)
        self.model = DataFileModel()
        self.list.setModel(self.model)
        self.list.setItemDelegate(DataFileDelegate())
        self.list.clicked.connect(
            lambda idx: self.dataFileChanged.emit(idx.data(DataFileModel.DataFileRole))
        )
        self.setWidget(self.list)

    def currentDataFile(self) -> SPCalDataFile:
        return self.list.currentIndex().data(DataFileModel.DataFileRole)

    def addDataFile(self, data_file: SPCalDataFile):
        self.model.beginInsertRows(
            QtCore.QModelIndex(), self.model.rowCount(), self.model.rowCount() + 1
        )
        self.model.data_files.append(data_file)
        self.model.endInsertRows()
        self.list.setCurrentIndex(self.model.index(self.model.rowCount(), 0))
        self.dataFileChanged.emit(data_file)

    def dataFiles(self) -> list[SPCalDataFile]:
        return self.model.data_files


if __name__ == "__main__":
    from pathlib import Path

    import numpy as np

    from spcal.isotope import ISOTOPE_TABLE

    datafiles = [
        SPCalTextDataFile(
            Path("/home/tom/Downloads/UTSCPO_430809_0.pdf"),
            np.ones(10, dtype=[("a", float), ("b", float)]),
            np.arange(100),
            [ISOTOPE_TABLE[("Fe", 56)], ISOTOPE_TABLE["Cu", 63]],
        ),
        SPCalTextDataFile(
            Path("/home/tom/Downloads/ASDUIH2323a.pdf"),
            np.ones(10, dtype=[("a", float)]),
            np.arange(100),
            [ISOTOPE_TABLE[("Fe", 56)]],
        ),
    ]
    app = QtWidgets.QApplication()
    view = SPCalDataFilesDock()
    for df in datafiles:
        df.selected_isotopes = df.isotopes
        view.addDataFile(df)
    # view = QtWidgets.QListView()
    # view.setMouseTracking(True)
    # model = DataFileModel(datafiles)
    # model.editIsotopesRequested.connect(lambda i: print(i))
    # view.setModel(model)
    # view.setItemDelegate(DataFileDelegate())
    view.show()
    app.exec()
