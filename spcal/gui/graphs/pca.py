from typing import List

import numpy as np
import pyqtgraph
from PySide6 import QtCore, QtGui, QtWidgets

from spcal.calc import pca
from spcal.gui.graphs.base import SinglePlotGraphicsView


class PCAArrow(QtWidgets.QGraphicsPathItem):
    def __init__(
        self,
        angle_r: float,
        length: float,
        name: str | None = None,
        pen: QtGui.QPen | None = None,
        brush: QtGui.QPen | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent=parent)
        self.setFlag(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations
        )

        # Angle from top
        if angle_r < 0.0:
            angle_r += 2.0 * np.pi
        angle_r = angle_r % (2.0 * np.pi)

        if pen is None:
            pen = QtGui.QPen(QtCore.Qt.red, 2.0)
            pen.setCosmetic(True)
        self.setPen(pen)

        if brush is None:
            brush = QtGui.QBrush(QtCore.Qt.red)
        self.setBrush(brush)

        path = QtGui.QPainterPath(QtCore.QPointF(0.0, 0.0))
        path.lineTo(0.0, -length)
        path.lineTo(3.0, -length + 3.0)
        path.lineTo(-3.0, -length + 3.0)
        path.lineTo(0.0, -length)

        tr = QtGui.QTransform().rotateRadians(angle_r)
        path = tr.map(path)
        self.setPath(path)

        if name is not None:
            self.text = QtWidgets.QGraphicsSimpleTextItem(name, self)
            rect = QtGui.QFontMetrics(self.text.font()).boundingRect(self.text.text())
            pos = tr.map(QtCore.QPointF(0.0, -(length * 1.05)))
            pos.setX(pos.x() - rect.width() * (np.sin(angle_r + np.pi) + 1.0) / 2.0)
            pos.setY(pos.y() - rect.height() * (np.cos(angle_r) + 1.0) / 2.0)

            self.text.setPos(pos)

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget | None = None,
    ) -> None:
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        super().paint(painter, option, widget)


class PCAView(SinglePlotGraphicsView):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__("PCA", xlabel="PC 1", ylabel="PC 2", parent=parent)

    def draw(
        self,
        X: np.ndarray,
        feature_names: List[str] | None = None,
        brush: QtGui.QBrush | None = None,
    ) -> None:
        a, v, _ = pca(X, 2)

        if brush is None:
            brush = QtGui.QBrush(QtCore.Qt.GlobalColor.black)

        scatter = pyqtgraph.ScatterPlotItem(x=a[:, 0], y=a[:, 1], pen=None, brush=brush)
        self.plot.addItem(scatter)

        if feature_names is not None:
            assert len(feature_names) == v.shape[1]

            angles = np.arctan2(v[0], v[1])
            for name, angle in zip(feature_names, angles):
                arrow = PCAArrow(angle, 100.0, name=name)
                self.plot.addItem(arrow)
