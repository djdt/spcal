import tempfile
from pathlib import Path

import numpy as np
from pytestqt.qtbot import QtBot

from spcal.gui.main import SPCalWindow
from spcal.io.text import read_single_particle_file


def test_batch_export(qtbot: QtBot):
    window = SPCalWindow()
    qtbot.add_widget(window)
    with qtbot.wait_exposed(window):
        window.show()

    assert not window.action_open_batch.isEnabled()

    npz = np.load(Path(__file__).parent.parent.joinpath("data/agilent_au_data.npz"))
    data = np.array(npz["au50nm"], dtype=[("Au", float)])

    with tempfile.NamedTemporaryFile(suffix=".csv") as tmp:
        path = Path(tmp.name)
        with path.open("w") as fp:
            fp.write("Au\n")
            np.savetxt(fp, data, delimiter=",")

        data, old_names = read_single_particle_file(path)

        window.sample.loadData(
            data,
            {
                "path": Path(tmp.name),
                "columns": [0],
                "ignores": [],
                "first line": 0,
                "names": ["Au"],
                "old names": ["Au"],
                "cps": False,
                "delimiter": ",",
                "importer": "text",
                "dwelltime": 1e-4,
            },
        )

        assert window.action_open_batch.isEnabled()

        with tempfile.NamedTemporaryFile(suffix=".csv") as tmp_out:
            opath = Path(tmp_out.name)
            dlg = window.dialogBatchProcess()
            dlg.files.addItems([str(path)])
            dlg.output_name.setText(opath.name)
            dlg.output_dir.setText(str(opath.parent))
            assert opath.stat().st_size == 0
            with qtbot.wait_signal(dlg.processingFinshed):
                dlg.start()
            assert opath.stat().st_size > 0
