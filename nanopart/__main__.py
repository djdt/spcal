from PySide2 import QtWidgets
import sys

from nanopart.gui.main import NanoPartWindow

from typing import List


def main(argv: List[str] = []):
    app = QtWidgets.QApplication(argv)
    win = NanoPartWindow()
    win.show()
    app.exec_()


if __name__ == "__main__":
    main(sys.argv)
