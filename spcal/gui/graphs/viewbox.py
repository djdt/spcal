from typing import List, Tuple

from pyqtgraph import ViewBox
from PySide6 import QtCore, QtWidgets


class ExtentsViewBox(ViewBox):
    """Viewbox that autoRanges to a set value."""

    def setExtents(self, rect: QtCore.QRectF) -> None:
        self.extent = rect

    def autoRange(self, *args, **kwargs) -> None:
        if self.extent is None:
            super().autoRange(*args, **kwargs)
        else:
            self.setRange(rect=self.extent)


class ViewBoxForceScaleAtZero(ExtentsViewBox):
    """Viewbox that forces the bottom to be 0."""

    def scaleBy(
        self,
        s: List[float] | None = None,
        center: QtCore.QPointF | None = None,
        x: float | None = None,
        y: float | None = None,
    ) -> None:
        if center is not None:
            center.setY(0.0)
        super().scaleBy(s, center, x, y)

    def translateBy(
        self,
        t: QtCore.QPointF | None = None,
        x: float | None = None,
        y: float | None = None,
    ) -> None:
        if t is not None:
            t.setY(0.0)
        if y is not None:
            y = 0.0
        super().translateBy(t, x, y)
