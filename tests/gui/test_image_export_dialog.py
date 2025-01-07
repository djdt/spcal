from PySide6 import QtCore
from pytestqt.qtbot import QtBot

from spcal.gui.dialogs.imageexport import ImageExportDialog


def test_image_export_dialog(qtbot: QtBot):
    options = {"test option 1": False, "test option 2": False, "test option 3": True}
    dlg = ImageExportDialog(options=options)
    qtbot.add_widget(dlg)
    with qtbot.wait_exposed(dlg):
        dlg.show()

    for option, on in options.items():
        assert option in dlg.options
        assert dlg.options[option].isChecked() == on

    dlg.options["test option 2"].setChecked(True)
    dlg.spinbox_dpi.setValue(100)
    dlg.spinbox_size_x.setValue(400)
    dlg.spinbox_size_y.setValue(200)

    def check_settings(
        size: QtCore.QSize, dpi: float, options: dict[str, bool]
    ) -> bool:
        if size.width() != 400 or size.height() != 200:
            return False
        if dpi != 100.0:
            return False
        if options["test option 1"]:
            return False
        if not options["test option 2"]:
            return False
        if not options["test option 3"]:
            return False
        return True

    with qtbot.wait_signal(dlg.exportSettingsSelected, check_params_cb=check_settings):
        dlg.accept()
