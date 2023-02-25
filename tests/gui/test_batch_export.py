import tempfile
from pathlib import Path

import numpy as np
from pytestqt.qtbot import QtBot

from spcal.gui.batch import BatchProcessDialog
from spcal.gui.main import SPCalWindow
from spcal.io.nu import read_nu_directory, select_nu_signals
from spcal.io.text import read_single_particle_file
from spcal.io.tofwerk import read_tofwerk_file


def test_batch_export(qtbot: QtBot):
    window = SPCalWindow()
    qtbot.add_widget(window)
    with qtbot.wait_exposed(window):
        window.show()

    assert not window.action_open_batch.isEnabled()

    path = Path(__file__).parent.parent.joinpath("data/text/tofwerk_export_au.csv")

    data, old_names = read_single_particle_file(path, columns=(2,), new_names=("Au",))

    with qtbot.wait_signal(window.sample.detectionsChanged):
        window.sample.loadData(
            data,
            {
                "path": path,
                "columns": [2],
                "ignores": [0, 1],
                "first line": 0,
                "names": ["Au"],
                "old names": old_names,
                "cps": False,
                "delimiter": ",",
                "importer": "text",
                "dwelltime": 1e-4,
            },
        )

    assert window.action_open_batch.isEnabled()

    with tempfile.NamedTemporaryFile(suffix=".csv") as tmp:
        opath = Path(tmp.name)

        dlg = window.dialogBatchProcess()
        qtbot.add_widget(dlg)
        dlg.files.addItems([str(path)])

        dlg.output_name.setText(opath.name)
        dlg.output_dir.setText(str(opath.parent))

        assert opath.stat().st_size == 0

        with qtbot.wait_signal(dlg.processingFinshed):
            dlg.start()

        assert opath.stat().st_size > 0


def test_batch_export_nu(qtbot: QtBot):
    window = SPCalWindow()
    qtbot.add_widget(window)
    with qtbot.wait_exposed(window):
        window.show()

    assert not window.action_open_batch.isEnabled()

    path = Path(__file__).parent.parent.joinpath("data/nu")

    masses, signals, info = read_nu_directory(path)
    data = select_nu_signals(masses, signals, selected_masses={"Ar40": 39.96238})

    window.options.error_rate_poisson.setValue(0.1)

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
            },
        )

    assert window.action_open_batch.isEnabled()

    with tempfile.NamedTemporaryFile(suffix=".csv") as tmp:
        opath = Path(tmp.name)

        dlg = window.dialogBatchProcess()
        qtbot.add_widget(dlg)
        dlg.files.addItems([str(path)])

        dlg.output_name.setText(opath.name)
        dlg.output_dir.setText(str(opath.parent))

        assert opath.stat().st_size == 0

        with qtbot.wait_signal(dlg.processingFinshed):
            dlg.start()

        assert opath.stat().st_size > 0
