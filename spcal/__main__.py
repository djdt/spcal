import argparse
import importlib.metadata
import logging
import sys
from importlib.resources import files
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from spcal.datafile import (
    SPCalDataFile,
    SPCalNuDataFile,
    SPCalTOFWERKDataFile,
    SPCalTextDataFile,
)
from spcal.io.nu import is_nu_directory, is_nu_run_info_file
from spcal.io.text import guess_text_parameters, is_text_file
from spcal.io.tofwerk import is_tofwerk_file
from spcal.isotope import SPCalIsotope
from spcal.gui.mainwindow import SPCalMainWindow
import spcal.resources  # noqa: F401

logging.captureWarnings(True)
logger = logging.getLogger("spcal")
logger.setLevel(logging.INFO)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    def existing_file(file: str) -> Path:
        path = Path(file)
        if not path.exists():
            raise argparse.ArgumentTypeError(f"file {file} does not exist")
        return path

    def isotope(text: str) -> SPCalIsotope:
        return SPCalIsotope.fromString(text)

    parser = argparse.ArgumentParser(
        prog="spcal",
        description="spICP-MS toolkit and visulistion.",
    )

    parser.add_argument(
        "--open", nargs="+", type=existing_file, help="open sample files"
    )
    parser.add_argument(
        "--isotopes",
        nargs="+",
        type=isotope,
        help="isotopes to select in opened files, defaults to first 5",
    )
    parser.add_argument(
        "--nohook", action="store_true", help="don't install the execption hook"
    )
    parser.add_argument("--old-gui", action="store_true")
    parser.add_argument(
        "qtargs", nargs=argparse.REMAINDER, help="arguments to pass to Qt"
    )
    args = parser.parse_args(argv)

    return args


def open_file(path: Path, isotopes: list[SPCalIsotope] | None = None) -> SPCalDataFile:
    if is_nu_directory(path) or is_nu_run_info_file(path):
        data_file = SPCalNuDataFile.load(path)
    elif is_tofwerk_file(path):
        data_file = SPCalTOFWERKDataFile.load(path)
    elif is_text_file(path):
        delim, skip_rows, _ = guess_text_parameters(path.open().readlines(2048))
        data_file = SPCalTextDataFile.load(path, delimiter=delim, skip_rows=skip_rows)
    else:
        raise FileNotFoundError("getImportDialogForPath: invalid directory.")

    data_file.selected_isotopes = isotopes or data_file.isotopes
    return data_file


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
        for path in args.open:
            data_file = open_file(path)
            window.addDataFile(data_file, args.isotopes or data_file.isotopes[:10])

    # Keep event loop active with timer
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(100)

    return app.exec_()


if __name__ == "__main__":
    main(sys.argv[1:])
