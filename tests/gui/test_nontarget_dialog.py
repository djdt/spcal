import numpy as np
import pytest
from PySide6 import QtCore
from pytestqt.qtbot import QtBot

from spcal.gui.dialogs.nontarget import NonTargetScreeningDialog

data = np.random.random((1000, 10))
for i, num in zip([0, 4, 5, 6], [10, 20, 50, 100]):
    data[:, i][np.random.choice(1000, num)] += 20.0


@pytest.mark.parametrize(
    "test_locales",
    [
        QtCore.QLocale.Language.English,
        QtCore.QLocale.Language.Spanish,
        QtCore.QLocale.Language.German,
    ],
    indirect=True,
)
def test_import_dialog_nu(qtbot: QtBot):
    def check_data(idx: np.ndarray, ppm: np.ndarray):
        if not np.all(np.isin([0, 4, 5, 6], idx)):
            return False
        return True

    def get_data_func(size: int) -> np.ndarray:
        return data[:size]

    dlg = NonTargetScreeningDialog(get_data_func, screening_ppm=100.0)
    qtbot.add_widget(dlg)
    with qtbot.wait_exposed(dlg):
        dlg.open()

    with qtbot.wait_signal(
        dlg.screeningComplete, check_params_cb=check_data, timeout=500
    ):
        dlg.accept()


def test_import_dialog_nu_high_screening_ppm(qtbot: QtBot):
    def check_data(idx: np.ndarray, ppm: np.ndarray):
        if not np.all(np.isin([4, 5, 6], idx)):
            return False
        return True

    def get_data_func(size: int) -> np.ndarray:
        return data[:size]

    dlg = NonTargetScreeningDialog(get_data_func, screening_ppm=10000.0)
    qtbot.add_widget(dlg)
    with qtbot.wait_exposed(dlg):
        dlg.open()

    with qtbot.wait_signal(
        dlg.screeningComplete, check_params_cb=check_data, timeout=500
    ):
        dlg.accept()
