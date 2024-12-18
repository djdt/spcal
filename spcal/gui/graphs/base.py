from pathlib import Path

import numpy as np
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.gui.util import create_action


class AxisEditDialog(QtWidgets.QDialog):
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

    def accept(self) -> None:
        self.rangeSelected.emit(self.spinbox_lo.value(), self.spinbox_hi.value())
        super().accept()


class SinglePlotGraphicsView(pyqtgraph.GraphicsView):
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

        if font is None:
            font = QtGui.QFont()
        self.font = font

        pen = QtGui.QPen(QtCore.Qt.black, 1.0)
        pen.setCosmetic(True)

        self.export_data: dict[str, np.ndarray] = {}

        self.xaxis = pyqtgraph.AxisItem("bottom", pen=pen, textPen=pen, tick_pen=pen)
        self.xaxis.setLabel(xlabel, units=xunits)

        self.yaxis = pyqtgraph.AxisItem("left", pen=pen, textPen=pen, tick_pen=pen)
        self.yaxis.setLabel(ylabel, units=yunits)

        self.plot = pyqtgraph.PlotItem(
            title=title,
            name="central_plot",
            axisItems={"bottom": self.xaxis, "left": self.yaxis},
            viewBox=viewbox,
            parent=parent,
        )
        # Common options
        self.plot.setMenuEnabled(False)
        self.plot.hideButtons()
        self.plot.addLegend(
            offset=(-5, 5), verSpacing=0, colCount=1, labelTextColor="black"
        )

        self.setFont(self.font)
        self.setCentralWidget(self.plot)

        self.action_auto_scale_y = create_action(
            "auto-scale-y",
            "Auto Scale Y",
            "Scale the y-axis to the maximum shown data point.",
            self.setAutoScaleY,
        )
        self.action_auto_scale_y.setCheckable(True)

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
            self.exportData,
        )

        self.context_menu_actions: list[QtGui.QAction] = []

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.xaxis.contains(self.xaxis.mapFromView(event.pos())):
            scale = self.xaxis.scale
            (v1, v2), (_, _) = self.plot.getViewBox().viewRange()
            min, max = self.plot.getViewBox().state["limits"]["xLimits"]
            dlg = AxisEditDialog(
                (v1 * scale, v2 * scale), (min * scale, max * scale), parent=self
            )
            dlg.rangeSelected.connect(self.setXAxisRange)
            dlg.open()
            event.accept()
        elif self.yaxis.contains(self.yaxis.mapFromView(event.pos())):
            scale = self.yaxis.scale
            (_, _), (v1, v2) = self.plot.getViewBox().viewRange()
            min, max = self.plot.getViewBox().state["limits"]["yLimits"]
            dlg = AxisEditDialog(
                (v1 * scale, v2 * scale), (min * scale, max * scale), parent=self
            )
            dlg.rangeSelected.connect(self.setYAxisRange)
            dlg.open()
            event.accept()
        else:
            from spcal.gui.dialogs.imageexport import ImageExportDialog

            dlg = ImageExportDialog(self, parent=self.parent())
            dlg.open()
            super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        if self.xaxis.contains(
            self.xaxis.mapFromView(event.pos())
        ) or self.yaxis.contains(self.yaxis.mapFromView(event.pos())):
            event.ignore()
            return

        menu = QtWidgets.QMenu(self)
        menu.addAction(self.action_copy_image)
        menu.addAction(self.action_auto_scale_y)

        if self.plot.legend is not None:
            menu.addAction(self.action_show_legend)

        if self.readyForExport():
            menu.addSeparator()
            menu.addAction(self.action_export_data)

        if len(self.context_menu_actions) > 0:
            menu.addSeparator()
            for action in self.context_menu_actions:
                menu.addAction(action)

        event.accept()
        menu.popup(event.globalPos())

    def setXAxisRange(self, min: float, max: float) -> None:
        scale = self.xaxis.scale
        if min > max:
            min, max = max, min
        self.plot.getViewBox().setRange(xRange=(min / scale, max / scale))

    def setYAxisRange(self, min: float, max: float) -> None:
        scale = self.yaxis.scale
        self.setAutoScaleY(False)
        if min > max:
            min, max = max, min
        self.plot.getViewBox().setRange(yRange=(min / scale, max / scale))

    def setAutoScaleY(self, auto_scale: bool) -> None:
        self.action_auto_scale_y.setChecked(auto_scale)
        self.plot.setMouseEnabled(y=not auto_scale)
        self.plot.setAutoVisible(y=auto_scale)
        self.plot.enableAutoRange(y=auto_scale)

    def setLegendVisible(self, visible: bool) -> None:
        self.action_show_legend.setChecked(visible)
        self.plot.legend.setVisible(visible)

    def copyToClipboard(self) -> None:
        """Copy current view to system clipboard."""
        pixmap = QtGui.QPixmap(self.viewport().size())
        painter = QtGui.QPainter(pixmap)
        self.render(painter)
        painter.end()
        QtWidgets.QApplication.clipboard().setPixmap(pixmap)  # type: ignore

    def clear(self) -> None:
        self.plot.legend.clear()
        self.plot.clear()
        self.export_data.clear()

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

    def redrawLegend(self) -> None:
        if self.plot.legend is None:
            return

        # store items
        items = []
        for item, label in self.plot.legend.items:
            items.append((item, label.text))
        # clear
        self.plot.legend.clear()
        # re-add
        for item, text in items:
            self.scene().addItem(item)
            item.show()
            self.plot.legend.addItem(item, text)
        # fix label heights
        for item, label in self.plot.legend.items:
            label.setGeometry(label.itemRect())

        self.plot.legend.updateSize()

    def setFont(self, font: QtGui.QFont) -> None:
        self.font = font

        self.xaxis.setStyle(tickFont=font, tickTextHeight=font.pointSize())
        self.xaxis.label.setFont(font)
        self.yaxis.setStyle(tickFont=font, tickTextHeight=font.pointSize())
        self.yaxis.label.setFont(font)
        self.plot.titleLabel.setText(
            self.plot.titleLabel.text,
            family=font.family(),
            size=f"{font.pointSize()}pt",
        )

        if self.plot.legend is not None:
            self.plot.legend.setLabelTextSize(f"{font.pointSize()}pt")
            self.redrawLegend()

    def exportData(self) -> None:
        dir = QtCore.QSettings().value("RecentFiles/1/path", None)
        dir = str(Path(dir).parent) if dir is not None else ""
        path, filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Data",
            dir,
            "CSV Documents(*.csv);;Numpy archives(*.npz);;All Files(*)",
        )
        if path == "":
            return

        path = Path(path)

        filter_suffix = filter[filter.rfind(".") : -1]
        if filter_suffix != "":  # append suffix if missing
            path = path.with_suffix(filter_suffix)

        data = self.dataForExport()
        names = list(data.keys())

        if path.suffix.lower() == ".csv":
            header = "\t".join(name for name in names)
            stack = np.full(
                (max(d.size for d in data.values()), len(data)),
                np.nan,
                dtype=np.float32,
            )
            for i, x in enumerate(data.values()):
                stack[: x.size, i] = x
            np.savetxt(
                path, stack, delimiter="\t", comments="", header=header, fmt="%.16g"
            )
        elif path.suffix.lower() == ".npz":
            np.savez_compressed(
                path,
                **{k: v for k, v in data.items()},
            )
        else:
            raise ValueError("dialogExportData: file suffix must be '.npz' or '.csv'.")

    def dataForExport(self) -> dict[str, np.ndarray]:
        return self.export_data

    def readyForExport(self) -> bool:
        return len(self.export_data) > 0

    def setLimits(self, **kwargs) -> None:
        self.plot.setLimits(**kwargs)

    def setDataLimits(
        self,
        xMin: float | None = None,
        xMax: float | None = None,
        yMin: float | None = None,
        yMax: float | None = None,
    ) -> None:
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
        self.setLimits(**limits)

    def zoomReset(self) -> None:
        x, y = self.plot.vb.state["autoRange"][0], self.plot.vb.state["autoRange"][1]
        self.plot.autoRange()
        self.plot.enableAutoRange(x=x, y=y)

        # Reset the legend postion
        if self.plot.legend is not None:
            self.plot.legend.anchor(
                QtCore.QPointF(1, 0), QtCore.QPointF(1, 0), QtCore.QPointF(-10, 10)
            )
