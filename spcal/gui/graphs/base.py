from pathlib import Path

import numpy as np
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.io import get_save_spcal_path, most_recent_spcal_path
from spcal.gui.util import create_action


class AxisRangeDialog(QtWidgets.QDialog):
    rangeSelected = QtCore.Signal(float, float)

    def __init__(
        self,
        view_range: tuple[float, float],
        view_limit: tuple[float, float],
        font: QtGui.QFont | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)

        self.spinbox_lo = QtWidgets.QDoubleSpinBox()
        self.spinbox_hi = QtWidgets.QDoubleSpinBox()

        for sb in [self.spinbox_lo, self.spinbox_hi]:
            sb.setRange(*view_limit)
            sb.setStepType(QtWidgets.QAbstractSpinBox.StepType.AdaptiveDecimalStepType)

        self.spinbox_lo.setValue(view_range[0])
        self.spinbox_hi.setValue(view_range[1])

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout = QtWidgets.QFormLayout()
        layout.addRow("min:", self.spinbox_lo)
        layout.addRow("max:", self.spinbox_hi)
        layout.addRow(self.button_box)
        self.setLayout(layout)

    def accept(self):
        self.rangeSelected.emit(self.spinbox_lo.value(), self.spinbox_hi.value())
        super().accept()


class SinglePlotItem(pyqtgraph.PlotItem):
    requestContextMenu = QtCore.Signal(QtCore.QPoint)

    def __init__(
        self,
        title: str,
        xlabel: str = "",
        ylabel: str = "",
        xunits: str | None = None,
        yunits: str | None = None,
        viewbox: pyqtgraph.ViewBox | None = None,
        font: QtGui.QFont | None = None,
        pen: QtGui.QPen | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        self.xaxis = pyqtgraph.AxisItem("bottom", pen=pen, textPen=pen, tick_pen=pen)
        self.xaxis.setLabel(xlabel, units=xunits)

        self.yaxis = pyqtgraph.AxisItem("left", pen=pen, textPen=pen, tick_pen=pen)
        self.yaxis.setLabel(ylabel, units=yunits)

        super().__init__(
            title=title,
            name="SPCal_SinglePlotItem",
            axisItems={"bottom": self.xaxis, "left": self.yaxis},
            viewBox=viewbox,
            parent=parent,
        )

        self.setMenuEnabled(False)
        self.hideButtons()
        self.addLegend(offset=(-5, 5), verSpacing=0, colCount=1, labelTextColor="black")

        if font is None:
            font = QtGui.QFont()
        self.setFont(font)

    def contextMenuEvent(self, event: QtWidgets.QGraphicsSceneContextMenuEvent):  # type: ignore
        self.requestContextMenu.emit(event.pos().toPoint())  # type: ignore

    def setFont(self, font: QtGui.QFont):  # type: ignore , pyqtgraph qt versions
        super().setFont(font)  # type: ignore

        fm = QtGui.QFontMetrics(font)
        pen: QtGui.QPen = self.xaxis.tickPen()  # type: ignore
        pen.setWidthF(fm.lineWidth())

        self.xaxis.setStyle(tickFont=font)
        # height calculation in pyqtgraph breaks for larger fonts
        self.xaxis.setHeight(fm.height() * 2)
        self.xaxis.setTickPen(pen)
        self.xaxis.label.setFont(font)  # type: ignore , not None

        self.yaxis.setStyle(tickFont=font)
        # estimate max width
        self.yaxis.setWidth(fm.tightBoundingRect("8888").width() * 2)
        self.yaxis.setTickPen(pen)
        self.yaxis.label.setFont(font)  # type: ignore , not None

        self.titleLabel.setText(
            self.titleLabel.text,
            family=font.family(),
            size=f"{font.pointSize()}pt",
        )

        if self.legend is not None:
            self.legend.setLabelTextSize(f"{font.pointSize()}pt")
            self.redrawLegend()

    def redrawLegend(self):
        if self.legend is None:
            return
        # store items
        items = []
        for item, label in self.legend.items:
            items.append((item, label.text))
        # clear
        self.legend.clear()
        # re-add
        for item, text in items:
            self.scene().addItem(item)  # type: ignore
            item.show()
            self.legend.addItem(item, text)
        # fix label heights
        height = 0.0
        for item, label in self.legend.items:
            height = height + label.itemRect().height()

        # fix broken geometry calculations
        geometry = self.legend.geometry()
        self.legend.setGeometry(geometry.x(), geometry.y(), geometry.width(), height)


class SinglePlotGraphicsView(pyqtgraph.GraphicsView):
    UNIT_LABELS = {
        "signal": ("Signal (cts)", None, 1.0),
        "mass": ("Mass", "g", 1e3),  # kg -> g
        "size": ("Size", "m", 1.0),
        "volume": ("Volume", "L", 1.0),
    }

    def __init__(
        self,
        title: str,
        xlabel: str = "",
        ylabel: str = "",
        xunits: str | None = None,
        yunits: str | None = None,
        viewbox: pyqtgraph.ViewBox | None = None,
        font: QtGui.QFont | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(background="white", parent=parent)

        self.data_for_export: dict[str, np.ndarray] = {}

        self.action_zoom_reset = create_action(
            "zoom-reset",
            "Reset zoom",
            "Show the full extent of the data.",
            self.zoomReset,
        )

        self.action_copy_image = create_action(
            "insert-image",
            "Copy Image to Clipboard",
            "Copy an image of the plot to the clipboard.",
            self.copyToClipboard,
        )

        self.action_show_legend = create_action(
            "view-visible",
            "Show Legend",
            "Toggle visibility of the legend.",
            self.setLegendVisible,
        )
        self.action_show_legend.setChecked(True)
        self.action_show_legend.setCheckable(True)

        self.action_export_data = create_action(
            "document-export",
            "Export Data",
            "Save currently loaded data to file.",
            lambda: self.exportData(None),
        )
        self.action_export_image = create_action(
            "viewimage",
            "Export Image",
            "Save the image to file, at a specified size and DPI.",
            lambda: self.exportImage(None),
        )

        self.context_menu_actions: list[QtGui.QAction] = []

        pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 1.0 * self.devicePixelRatio())
        pen.setCosmetic(True)

        self.plot = SinglePlotItem(
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xunits=xunits,
            yunits=yunits,
            viewbox=viewbox,
            font=font,
            pen=pen,
            parent=parent,
        )
        self.setCentralWidget(self.plot)

        self.plot.requestContextMenu.connect(self.customContextMenu)  # type: ignore

    def font(self) -> QtGui.QFont:  # type: ignore , weird pyqtgraph classes
        return self.plot.font()  # type: ignore

    def setFont(self, font: QtGui.QFont):  # type: ignore
        self.plot.setFont(font)

    def drawCurve(
        self,
        x: np.ndarray,
        y: np.ndarray,
        pen: QtGui.QPen | None = None,
        name: str | None = None,
        connect: str = "all",
    ) -> pyqtgraph.PlotCurveItem:
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 1.0)
            pen.setCosmetic(True)

        diffs = np.diff(y, n=2, append=0, prepend=0) != 0
        curve = pyqtgraph.PlotCurveItem(
            x=x[diffs],
            y=y[diffs],
            pen=pen,
            connect=connect,
            name=name,
            skipFiniteCheck=True,
        )
        self.plot.addItem(curve)
        return curve

    def drawHistogram(
        self,
        counts: np.ndarray,
        edges: np.ndarray,
        width: float = 0.5,
        offset: float = 0.0,
        pen: QtGui.QPen | None = None,
        brush: QtGui.QBrush | None = None,
    ) -> pyqtgraph.PlotCurveItem:
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 1.0)
            pen.setCosmetic(True)

        if brush is None:
            brush = QtGui.QBrush(QtCore.Qt.GlobalColor.black)

        assert width > 0.0 and width <= 1.0
        assert offset >= 0.0 and offset < 1.0

        widths = np.diff(edges)

        x = np.repeat(edges, 2)

        # Calculate bar start and end points for width / offset
        x[1:-1:2] += widths * ((1.0 - width) / 2.0 + offset)
        x[2::2] -= widths * ((1.0 - width) / 2.0 - offset)

        y = np.zeros(counts.size * 2 + 1, dtype=counts.dtype)
        y[1:-1:2] = counts

        curve = pyqtgraph.PlotCurveItem(
            x=x,
            y=y,
            stepMode="center",
            fillLevel=0,
            fillOutline=True,
            pen=pen,
            brush=brush,
            skipFiniteCheck=True,
        )

        self.plot.addItem(curve)
        return curve

    def drawLine(
        self,
        y: float,
        orientation: QtCore.Qt.Orientation,
        pen: QtGui.QPen | None = None,
        name: str | None = None,
        connect: str = "all",
    ) -> pyqtgraph.PlotCurveItem:
        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.GlobalColor.black, 1.0)
            pen.setCosmetic(True)

        x0, x1 = self.dataBounds()[:2]
        return self.drawCurve(
            x=np.array([x0, x1]), y=np.array([y, y]), pen=pen, name=name, connect="all"
        )

    def drawScatter(
        self,
        x: np.ndarray,
        y: np.ndarray,
        size: float = 6.0,
        symbol: str = "o",
        pen: QtGui.QPen | None = None,
        name: str | None = None,
        brush: QtGui.QBrush | None = None,
    ) -> pyqtgraph.ScatterPlotItem:
        if brush is None:
            brush = QtGui.QBrush(QtCore.Qt.GlobalColor.red)

        scatter = pyqtgraph.ScatterPlotItem(
            x=x,
            y=y,
            size=size,
            symbol=symbol,
            pen=pen,
            brush=brush,
            name=name,
        )
        self.plot.addItem(scatter)
        return scatter

    def axisMenu(self, axis: str = "x") -> QtWidgets.QMenu:
        assert self.plot.vb is not None

        (x1, x2), (y1, y2) = self.plot.vb.viewRange()
        min, max = self.plot.vb.state["limits"][f"{axis}Limits"]
        if axis == "x":
            scale = self.plot.xaxis.scale
            v1, v2 = x1, x2
        elif axis == "y":
            scale = self.plot.yaxis.scale
            v1, v2 = y1, y2
        else:
            raise ValueError("axis must be 'x' or 'y'")

        def open_range_dialog():
            dlg = AxisRangeDialog(
                (v1 * scale, v2 * scale),
                (min * scale, max * scale),
                self.font(),
                parent=self,  # type: ignore , pyqtgraph weirdness
            )
            dlg.rangeSelected.connect(
                lambda min, max: self.setAxisRange(axis, min, max)
            )
            dlg.open()

        menu = QtWidgets.QMenu(self)  # type: ignore

        autoscale_action = create_action(
            f"auto-scale-{axis}",
            "Auto Scale",
            "Automatically scale the axis with visible data.",
            lambda autoscale: self.setAxisAutoScale(axis, autoscale),
            checkable=True,
        )
        autoscale_action.setParent(self)  # type: ignore
        autoscale_action.setChecked(
            self.plot.vb.state["autoRange"][["x", "y"].index(axis)] is not False
        )
        range_action = create_action(
            f"panel-fit-{['width', 'height'][['x', 'y'].index(axis)]}",
            "Set Range...",
            "Set the view range of the axis.",
            open_range_dialog,
        )
        range_action.setParent(self)  # type: ignore

        menu.addAction(autoscale_action)
        menu.addAction(range_action)

        return menu

    def customContextMenu(self, pos: QtCore.QPoint):
        view_pos = self.plot.xaxis.mapFromView(pos)
        if view_pos is not None and self.plot.xaxis.contains(view_pos):
            menu = self.axisMenu("x")
            menu.popup(self.mapToGlobal(pos))  # type: ignore
            return
        view_pos = self.plot.yaxis.mapFromView(pos)
        if view_pos is not None and self.plot.yaxis.contains(view_pos):
            menu = self.axisMenu("y")
            menu.popup(self.mapToGlobal(pos))  # type: ignore
            return

        menu = QtWidgets.QMenu(self)  # type: ignore
        menu.addAction(self.action_copy_image)
        menu.addSeparator()

        if self.plot.legend is not None:
            menu.addAction(self.action_show_legend)

        menu.addAction(self.action_zoom_reset)

        menu.addSeparator()
        if len(self.data_for_export) > 0:
            menu.addAction(self.action_export_data)
        menu.addAction(self.action_export_image)

        if len(self.context_menu_actions) > 0:
            menu.addSeparator()
            for action in self.context_menu_actions:
                menu.addAction(action)

        menu.popup(self.mapToGlobal(pos))

    def setAxisRange(self, axis: str, min: float, max: float):
        if self.plot.vb is None:
            return
        if min > max:
            min, max = max, min
        if axis == "x":
            scale = self.plot.xaxis.scale
            self.plot.vb.setRange(xRange=(min / scale, max / scale))  # type: ignore , pyqtgraph bad names
        elif axis == "y":
            scale = self.plot.yaxis.scale
            self.plot.vb.setRange(yRange=(min / scale, max / scale))  # type: ignore , pyqtgraph bad names
        else:
            raise ValueError(f"bad axis '{axis}', must be x or y")

    def setAxisAutoScale(self, axis: str, auto_scale: bool):
        if self.plot.vb is None:
            return
        if axis == "x":
            self.plot.vb.setMouseEnabled(x=not auto_scale)
            self.plot.vb.setAutoVisible(x=auto_scale)
            self.plot.vb.enableAutoRange(x=auto_scale)
        else:
            self.plot.vb.setMouseEnabled(y=not auto_scale)
            self.plot.vb.setAutoVisible(y=auto_scale)
            self.plot.vb.enableAutoRange(y=auto_scale)

    def setLegendVisible(self, visible: bool):
        self.action_show_legend.setChecked(visible)
        if self.plot.legend is not None:
            self.plot.legend.setVisible(visible)

    def copyToClipboard(self):
        """Copy current view to system clipboard."""
        pixmap = QtGui.QPixmap(self.viewport().size())
        painter = QtGui.QPainter(pixmap)
        self.render(painter)
        painter.end()
        QtWidgets.QApplication.clipboard().setPixmap(pixmap)  # type: ignore

    def clear(self):
        self.data_for_export.clear()
        if self.plot.legend is not None:
            self.plot.legend.clear()
        self.plot.clear()
        self.scene().invalidate()

    def dataBounds(self) -> tuple[float, float, float, float]:
        items = [item for item in self.plot.listDataItems() if item.isVisible()]
        bx = np.array([item.dataBounds(0) for item in items], dtype=float)
        by = np.array([item.dataBounds(1) for item in items], dtype=float)
        bx, by = np.nan_to_num(bx), np.nan_to_num(by)
        # Just in case
        if len(bx) == 0 or len(by) == 0:
            return (0, 1, 0, 1)
        return (
            np.amin(bx[:, 0]),
            np.amax(bx[:, 1]),
            np.amin(by[:, 0]),
            np.amax(by[:, 1]),
        )

    def dataRect(self) -> QtCore.QRectF:
        x0, x1, y0, y1 = self.dataBounds()
        return QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)

    def exportData(self, path: Path | None):
        if path is None:
            path = most_recent_spcal_path()
            if path is not None:
                path = path.parent.joinpath("export.csv")
            path = get_save_spcal_path(
                self, [("CSV Documents", ".csv"), ("Numpy archives", ".npz")], path=path
            )
            if path is None:
                return

        names = list(self.data_for_export.keys())

        if path.suffix.lower() == ".csv":
            header = "\t".join(name for name in names)
            stack = np.full(
                (
                    max(d.size for d in self.data_for_export.values()),
                    len(self.data_for_export),
                ),
                np.nan,
                dtype=np.float32,
            )
            for i, x in enumerate(self.data_for_export.values()):
                stack[: x.size, i] = x
            np.savetxt(
                path, stack, delimiter="\t", comments="", header=header, fmt="%.16g"
            )
        elif path.suffix.lower() == ".npz":
            np.savez_compressed(
                path,
                **{k: v for k, v in self.data_for_export.items()},
                allow_pickle=False,
            )
        else:
            raise ValueError("file suffix must be '.npz' or '.csv'.")

    def exportImage(
        self,
        path: Path | str | None,
        background: QtGui.QColor | QtCore.Qt.GlobalColor | None = None,
    ):
        if path is None:
            dir = QtCore.QSettings().value("RecentFiles/1/path", None)
            dir = str(Path(str(dir)).parent) if dir is not None else ""
            path, filter = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Export Image",
                dir,
                "PNG Images(*.png);;All Files(*)",
            )
            if path == "":
                return
        path = Path(path)

        if background is None:
            background = QtGui.QColor(QtCore.Qt.GlobalColor.white)

        size = self.viewport().size()
        image = QtGui.QImage(size, QtGui.QImage.Format.Format_ARGB32)
        image.fill(background)

        painter = QtGui.QPainter(image)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)

        # Setup for export
        for item in self.plot.items:
            if hasattr(item, "_exportOpts"):
                item._exportOpts = {}

        self.scene().prepareForPaint()
        self.scene().render(
            painter,
            QtCore.QRectF(image.rect()),
            self.viewRect(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
        )
        painter.end()
        image.save(str(path.resolve()))

    def setDataLimits(
        self,
        xMin: float | None = None,
        xMax: float | None = None,
        yMin: float | None = None,
        yMax: float | None = None,
    ):
        """Set all plots limits in range 0.0 - 1.0."""
        bounds = self.dataBounds()
        dx = bounds[1] - bounds[0]
        dy = bounds[3] - bounds[2]
        limits = {}
        if xMin is not None:
            limits["xMin"] = bounds[0] + dx * xMin
        if xMax is not None:
            limits["xMax"] = bounds[0] + dx * xMax
        if yMin is not None:
            limits["yMin"] = bounds[2] + dy * yMin
        if yMax is not None:
            limits["yMax"] = bounds[2] + dy * yMax
        assert self.plot.vb is not None
        self.plot.vb.setLimits(**limits)

    def zoomReset(self):
        if self.plot.vb is not None:
            x, y = (
                self.plot.vb.state["autoRange"][0],
                self.plot.vb.state["autoRange"][1],
            )
            self.plot.vb.autoRange()
            self.plot.vb.enableAutoRange(x=x, y=y)

        # Reset the legend postion
        if self.plot.legend is not None:
            self.plot.legend.anchor(
                QtCore.QPointF(1, 0), QtCore.QPointF(1, 0), QtCore.QPointF(-10, 10)
            )
