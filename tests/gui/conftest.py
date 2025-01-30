import pytest
from PySide6 import QtCore


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
