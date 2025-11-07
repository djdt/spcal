import argparse
import importlib.metadata
import logging
import sys
from importlib.resources import files
from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

import spcal.resources  # noqa: F401
from spcal.datafile import (
    SPCalNuDataFile,
    SPCalTextDataFile,
    SPCalTOFWERKDataFile,
)
from spcal.gui.mainwindow import SPCalMainWindow
from spcal.io.nu import is_nu_directory, is_nu_run_info_file
from spcal.io.text import guess_text_parameters, is_text_file
from spcal.io.tofwerk import is_tofwerk_file
from spcal.isotope import SPCalIsotope

logging.captureWarnings(True)
logger = logging.getLogger("spcal")
logger.setLevel(logging.INFO)

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    class DataFileAction(argparse.Action):
        def __call__(
            self,
            parser: argparse.ArgumentParser,
            namespace: argparse.Namespace,
            values: Any | list[Any],
            option_string: str | None = None,
        ):
            if isinstance(values, list):
                path = Path(values[0])
            else:
                path = Path(values)

            if not path.exists():
                parser.error(f"{path} does not exist")

            if is_nu_directory(path) or is_nu_run_info_file(path):
                data_file = SPCalNuDataFile.load(path)
            elif is_tofwerk_file(path):
                data_file = SPCalTOFWERKDataFile.load(path)
            elif is_text_file(path):
                delim, skip_rows, _ = guess_text_parameters(path.open().readlines(2048))
                data_file = SPCalTextDataFile.load(
                    path, delimiter=delim, skip_rows=skip_rows
                )
            else:
                parser.error(f"{path} is not a valid single particle file")

            if isinstance(values, list):
                isotopes = [SPCalIsotope.fromString(value) for value in values[1:]]
                data_file.selected_isotopes = isotopes

            data_files = getattr(namespace, "open") or []
            data_files.append(data_file)
            setattr(namespace, "open", data_files)

    parser = argparse.ArgumentParser(
        prog="spcal",
        description="spICP-MS toolkit and visulistion.",
    )

    parser.add_argument(
        "--open",
        nargs="+",
        metavar=("PATH", "ISOTOPES"),
        type=str,
        action=DataFileAction,
        help="open sample file and optionally select isotopes",
    )
    parser.add_argument(
        "--nohook", action="store_true", help="don't install the execption hook"
    )
    parser.add_argument(
        "qtargs", nargs=argparse.REMAINDER, help="arguments to pass to Qt"
    )
    args = parser.parse_args(argv)

    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # prevent print group separators, they are buggy in Qt6 validators
    locale = QtCore.QLocale.system()
    locale.setNumberOptions(
        locale.NumberOption.OmitGroupSeparator
        | locale.NumberOption.RejectGroupSeparator
    )
    QtCore.QLocale.setDefault(locale)

    app = QtWidgets.QApplication(args.qtargs)
    app.setApplicationName("SPCal")
    app.setOrganizationName("SPCal")
    app.setApplicationVersion(importlib.metadata.version("spcal"))
    app.setWindowIcon(QtGui.QIcon(str(files("spcal.resources").joinpath("app.ico"))))

    window = SPCalMainWindow()

    if not args.nohook:
        sys.excepthook = window.exceptHook

    logger.addHandler(window.log.handler)
    logger.info(f"SPCal {app.applicationVersion()} started.")
    logger.info(f"using numpy {importlib.metadata.version('numpy')}.")
    logger.info(f"using pyqtgraph {importlib.metadata.version('pyqtgraph')}.")

    window.show()

    if args.open:
        for data_file in args.open:
            window.files.addDataFile(data_file)

    # Keep event loop active with timer
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(100)

    return app.exec_()


if __name__ == "__main__":
    main(sys.argv[1:])
