from PySide2 import QtCore, QtWidgets
import argparse
from pathlib import Path
import sys
import logging

from spcal import __version__
from spcal.gui.main import NanoPartWindow

from typing import List, Optional

logging.captureWarnings(True)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="spcal",
        description="spICP-MS toolkit and visulistion.",
    )

    parser.add_argument(
        "--sample", type=Path, help="Open file as sample on startup."
    )
    parser.add_argument(
        "--reference", type=Path, help="Open file as reference on startup."
    )
    parser.add_argument(
        "qtargs", nargs=argparse.REMAINDER, help="Arguments to pass to Qt."
    )
    args = parser.parse_args(argv)

    if args.sample is not None and not args.sample.exists():
        raise parser.error(f"[--sample]: File '{args.sample}' not found.")
    if args.reference is not None and not args.reference.exists():
        raise parser.error(f"[--reference]: File '{args.reference}' not found.")

    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    app = QtWidgets.QApplication(args.qtargs)
    app.setApplicationName("SPCal")
    app.setApplicationVersion(__version__)

    window = NanoPartWindow()

    sys.excepthook = window.exceptHook
    logger.addHandler(window.log.handler)
    logger.info(f"SPCal {__version__} started.")

    window.show()

    if args.sample:
        window.sample.loadFile(args.sample)
    if args.reference:
        window.reference.loadFile(args.reference)

    # Keep event loop active with timer
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(100)

    return app.exec_()


if __name__ == "__main__":
    main(sys.argv[1:])
