from PySide6 import QtGui, QtWidgets


class SmallTextEdit(QtWidgets.QTextEdit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.textChanged.connect(self.onTextChanged)
        self.onTextChanged()
        # self.document().blockCountChanged.connect(lambda i: print("block count", i))
        # self.document().documentLayoutChanged.connect(lambda: print("doc layout"))
    #     self.show()
    #     self.resize(self.size())

    def onTextChanged(self) -> None:
        # super().showEvent(event)
        print("text changed")
        self.show()
        self.setMaximumHeight(
            self.document().size().height()
            + self.contentsMargins().top()
            + self.contentsMargins().bottom()
        )
