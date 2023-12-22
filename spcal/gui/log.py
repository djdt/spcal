import logging

from PySide6 import QtCore, QtGui, QtWidgets


class LogRecordSignaller(QtCore.QObject):
    new_record = QtCore.Signal(str, logging.LogRecord)


class QtListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()

        self.signal = LogRecordSignaller()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)
        self.signal.new_record.emit(self.format(record), record)


class LoggingTextEdit(QtWidgets.QPlainTextEdit):
    """Display the python log in a QTextEdit.

    Log is read only and levels are colored.
    """

    COLORS = {
        logging.DEBUG: "black",
        logging.INFO: "blue",
        logging.WARNING: "orange",
        logging.ERROR: "red",
        logging.CRITICAL: "purple",
    }

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setFont(QtGui.QFont("monospace"))
        self.setReadOnly(True)

    @QtCore.Slot(str, logging.LogRecord)
    def add_record(self, string: str, record: logging.LogRecord) -> None:
        color = self.COLORS.get(record.levelno, "black")
        string = f"<pre><font color={color}>{string}</font></pre>"
        self.appendHtml(string)

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(600, 400)


class LoggingDialog(QtWidgets.QDialog):
    """Display the python log in a dialog."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self.handler = QtListHandler()
        self.textedit = LoggingTextEdit()
        self.handler.signal.new_record.connect(self.textedit.add_record)
        self.handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)8s - %(name)s : %(message)s",
                datefmt="%H:%M:%S",
            )
        )

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.textedit)
        self.setLayout(layout)
