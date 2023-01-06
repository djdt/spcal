from PySide6 import QtCore, QtGui, QtWidgets


class ElidedLabel(QtWidgets.QWidget):
    def __init__(
        self,
        text: str = "",
        elide: QtCore.Qt.TextElideMode = QtCore.Qt.TextElideMode.ElideLeft,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self._text = text
        self._elide = elide

    def elide(self) -> QtCore.Qt.TextElideMode:
        return self._elide

    def setElide(self, elide: QtCore.Qt.TextElideMode) -> None:
        self._elide = elide

    def text(self) -> str:
        return self._text

    def setText(self, text: str) -> None:
        self._text = text
        self.updateGeometry()

    def sizeHint(self) -> QtCore.QSize:
        fm = self.fontMetrics()
        return fm.boundingRect(self._text).size()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        fm = painter.fontMetrics()

        # width + 1 to prevent elide when text width = widget width
        elided = fm.elidedText(self._text, self._elide, self.width() + 1)
        painter.drawText(
            self.contentsRect(),
            QtCore.Qt.AlignVCenter
            | QtCore.Qt.TextSingleLine
            | QtCore.Qt.TextShowMnemonic,
            elided,
        )
