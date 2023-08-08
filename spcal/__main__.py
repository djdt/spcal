import argparse
import logging
import sys
from importlib.resources import files
from pathlib import Path
from typing import List

import numpy
from PySide6 import QtCore, QtGui, QtWidgets

import spcal.resources
from spcal import __version__
from spcal.gui.main import SPCalWindow

import pyqtgraph  # isort:skip


logging.captureWarnings(True)
logger = logging.getLogger("spcal")
logger.setLevel(logging.INFO)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    def existing_file(file: str) -> Path:
        path = Path(file)
        if not path.exists():
            raise argparse.ArgumentTypeError(f"file {file} does not exist")
        return path

    parser = argparse.ArgumentParser(
        prog="spcal",
        description="spICP-MS toolkit and visulistion.",
    )

    parser.add_argument(
        "--sample", type=existing_file, help="open file as sample on startup"
    )
    parser.add_argument(
        "--reference", type=existing_file, help="open file as reference on startup"
    )
    parser.add_argument(
        "--auto-yes", action="store_true", help="answer yes to all prompts"
    )
    parser.add_argument(
        "--nohook", action="store_true", help="don't install the execption hook"
    )
    parser.add_argument(
        "qtargs", nargs=argparse.REMAINDER, help="arguments to pass to Qt"
    )
    args = parser.parse_args(argv)

    return args


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)

    app = QtWidgets.QApplication(args.qtargs)
    app.setApplicationName("SPCal")
    app.setOrganizationName("SPCal")
    app.setApplicationVersion(__version__)
    app.setWindowIcon(QtGui.QIcon(str(files("spcal.resources").joinpath("app.ico"))))

    window = SPCalWindow()

    if not args.nohook:
        sys.excepthook = window.exceptHook

    logger.addHandler(window.log.handler)
    logger.info(f"SPCal {__version__} started.")
    logger.info(f"using numpy {numpy.version.version}.")
    logger.info(f"using pyqtgraph {pyqtgraph.__version__}.")

    window.show()

    if args.sample:
        dlg = window.sample.dialogLoadFile(args.sample)
        if args.auto_yes:
            dlg.accept()
    if args.reference:
        dlg = window.reference.dialogLoadFile(args.reference)
        if args.auto_yes:
            dlg.accept()

    # Keep event loop active with timer
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(100)

    return app.exec_()


if __name__ == "__main__":
    main(sys.argv[1:])
