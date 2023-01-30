from typing import List

from PySide6 import QtCore

from spcal.gui.graphs.base import SPCalViewBox


class ViewBoxForceScaleAtZero(SPCalViewBox):
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
