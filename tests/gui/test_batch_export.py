from pathlib import Path

import numpy as np
from pytestqt.qtbot import QtBot

from spcal.gui.main import SPCalWindow
from spcal.io.nu import read_nu_directory, select_nu_signals
from spcal.io.text import read_single_particle_file
from spcal.io.tofwerk import read_tofwerk_file


def test_batch_export(tmp_path: Path, qtbot: QtBot):
    window = SPCalWindow()
    qtbot.add_widget(window)
    with qtbot.wait_exposed(window):
        window.show()

    assert not window.action_open_batch.isEnabled()

    path = Path(__file__).parent.parent.joinpath("data/text/tofwerk_export_au.csv")

    data = read_single_particle_file(path, columns=(2,))

    with qtbot.wait_signal(window.sample.detectionsChanged):
        window.sample.loadData(
            data,
            {
                "path": path,
                "columns": [2],
                "ignores": [0, 1],
                "first line": 0,
                "names": {data.dtype.names[0]: data.dtype.names[0]},
                "cps": False,
                "delimiter": ",",
                "importer": "text",
                "dwelltime": 1e-4,
            },
        )

    assert window.action_open_batch.isEnabled()

    tmp = tmp_path.joinpath("batch_export.csv")
    dlg = window.dialogBatchProcess()
    qtbot.add_widget(dlg)
    dlg.files.addItems([str(path)])

    dlg.output_name.setText(tmp.name)
    dlg.output_dir.setText(str(tmp_path))

    assert not tmp.exists()

    with qtbot.wait_signal(dlg.processingFinshed):
        dlg.start()

    assert tmp.stat().st_size > 0


def test_batch_export_nu(tmp_path: Path, qtbot: QtBot):
    # Todo: need to make a better (more data) Nu test data file
    return
    window = SPCalWindow()
    qtbot.add_widget(window)
    with qtbot.wait_exposed(window):
        window.show()

    assert not window.action_open_batch.isEnabled()

    path = Path(__file__).parent.parent.joinpath("data/nu")

    masses, signals, info = read_nu_directory(path)
    data = select_nu_signals(masses, signals, selected_masses={"Ar40": 39.96238})

    window.options.poisson.alpha.setValue(0.1)

    with qtbot.wait_signal(window.sample.detectionsChanged):
        window.sample.loadData(
            data,
            {
                "path": path,
                "importer": "nu",
                "dwelltime": 2.8e-5,
                "isotopes": np.array(
                    [(18, "Ar", 40, 39.9623831237, 0.996035, 0)],
                    dtype=[
                        ("Number", np.uint16),
                        ("Symbol", "U2"),
                        ("Isotope", np.uint16),
                        ("Mass", float),
                        ("Composition", float),
                        ("Preffered", np.uint8),
                    ],
                ),
                "cycle": 1,
                "segment": 1,
                "blanking": True,
            },
        )

    assert window.action_open_batch.isEnabled()

    tmp = tmp_path.joinpath("batch_export_nu.csv")

    dlg = window.dialogBatchProcess()
    qtbot.add_widget(dlg)
    dlg.files.addItems([str(path)])

    dlg.output_name.setText(tmp.name)
    dlg.output_dir.setText(str(tmp_path))

    assert not tmp.exists()

    with qtbot.wait_signal(dlg.processingFinshed):
        dlg.start()

    assert tmp.stat().st_size > 0


def test_batch_export_tofwerk(tmp_path: Path, qtbot: QtBot):
    window = SPCalWindow()
    qtbot.add_widget(window)
    with qtbot.wait_exposed(window):
        window.show()

    assert not window.action_open_batch.isEnabled()

    path = Path(__file__).parent.parent.joinpath("data/tofwerk/tofwerk_au_50nm.h5")

    data, info, dwell = read_tofwerk_file(path, idx=np.array([293]))

    with qtbot.wait_signal(window.sample.detectionsChanged):
        window.sample.loadData(
            data,
            {
                "path": path,
                "importer": "tofwerk",
                "dwelltime": dwell,
                "isotopes": np.array(
                    [(79, "Au", 197, 196.96656879, 1.0, 1)],
                    dtype=[
                        ("Number", np.uint16),
                        ("Symbol", "U2"),
                        ("Isotope", np.uint16),
                        ("Mass", float),
                        ("Composition", float),
                        ("Preffered", np.uint8),
                    ],
                ),
                "other peaks": [],
            },
        )

    assert window.action_open_batch.isEnabled()

    tmp = tmp_path.joinpath("batch_export_tofwerk.csv")

    dlg = window.dialogBatchProcess()
    qtbot.add_widget(dlg)
    dlg.files.addItems([str(path)])

    dlg.output_name.setText(tmp.name)
    dlg.output_dir.setText(str(tmp_path))

    assert not tmp.exists()

    with qtbot.wait_signal(dlg.processingFinshed):
        dlg.start()

    assert tmp.stat().st_size > 0
