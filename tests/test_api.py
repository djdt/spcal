from spcal.api import compare_version_strings


def test_compare_verison_strings():
    assert compare_version_strings("0.0.9", "0.1.0")
    assert compare_version_strings("0.1.0", "0.1.1")
    assert compare_version_strings("0.1.9", "0.1.10")

    assert not compare_version_strings("0.1.1", "0.1.0")
    assert not compare_version_strings("0.1.0", "0.0.9")
    assert not compare_version_strings("1.0.9", "0.1.9")

    assert compare_version_strings("0.1", "0.1.1")
    assert not compare_version_strings("0.1.1", "0.1")
