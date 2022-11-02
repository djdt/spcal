from PySide6 import QtCore, QtWidgets
import argparse
from pathlib import Path
import sys
import logging

import numpy
import pyqtgraph

from spcal import __version__
from spcal.resources import icons
from spcal.gui.main import NanoPartWindow

from typing import List

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
    app.setApplicationVersion(__version__)

    window = NanoPartWindow()

    if not args.nohook:
        sys.excepthook = window.exceptHook

    logger.addHandler(window.log.handler)
    logger.info(f"SPCal {__version__} started.")
    logger.info(f"using numpy {numpy.version.version}.")
    logger.info(f"using pyqtgraph {pyqtgraph.__version__}.")

    window.show()

    if args.sample:
        window.sample.dialogLoadFile(args.sample)
    if args.reference:
        window.reference.dialogLoadFile(args.reference)

    # Keep event loop active with timer
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(100)

    return app.exec_()


if __name__ == "__main__":
    main(sys.argv[1:])
