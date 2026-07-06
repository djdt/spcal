import pytest
from PySide6 import QtCore, QtWidgets


@pytest.fixture(
    scope="module",
    autouse=True,
    params=[QtCore.QLocale.Language.C],
)
def test_locales(request):
    locale = QtCore.QLocale(request.param)
    locale.setNumberOptions(
        locale.NumberOption.OmitGroupSeparator
        | locale.NumberOption.RejectGroupSeparator
    )
    QtCore.QLocale.setDefault(locale)


@pytest.fixture(scope="session", autouse=True)
def app_config(qapp: QtWidgets.QApplication):
    qapp.setOrganizationName("PyTest")
    qapp.setOrganizationName("PyTest-SPCal")

    settings = QtCore.QSettings()
    settings.clear()
    settings.setValue("DisableCheckForUpdates", "0.0.0")
