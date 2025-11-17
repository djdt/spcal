from PySide6 import QtWidgets


class AdjustingTextEdit(QtWidgets.QTextEdit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.textChanged.connect(self.onTextChanged)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.onTextChanged()

    def onTextChanged(self):
        self.setMaximumHeight(
            self.document().size().height()
            + self.contentsMargins().top()
            + self.contentsMargins().bottom()
        )
