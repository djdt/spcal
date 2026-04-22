from spcal import isotope

import pytest


def test_isotope_base():
    iso = isotope.SPCalIsotopeBase("test")
    assert iso.name == "test"


def test_isotope():
    iso = isotope.SPCalIsotope("Tt", 1000, 123.456, composition=0.1)

    assert str(iso) == "1000Tt"
    assert iso.symbol == "Tt"

    assert iso == isotope.SPCalIsotope("Tt", 1000, 123, 456)
    assert not iso == isotope.SPCalIsotope("Tq", 1000, 123, 456)


def test_isotope_from_string():
    iso = isotope.SPCalIsotope.fromString("Au197")
    assert iso.name == "Au"
    assert iso.isotope == 197
    assert iso.composition == 1.0

    iso = isotope.SPCalIsotope.fromString("197Au")
    assert iso.name == "Au"
    assert iso.isotope == 197
    assert iso.composition == 1.0

    with pytest.raises(NameError):
        isotope.SPCalIsotope.fromString("197Bq")

    with pytest.raises(NameError):
        isotope.SPCalIsotope.fromString("Au")


def test_isotope_expression():
    expr = isotope.SPCalIsotopeExpression(
        "test expr", (isotope.ISOTOPE_TABLE[("H", 1)], isotope.ISOTOPE_TABLE[("H", 2)])
    )
    assert expr != isotope.SPCalIsotopeExpression(
        "test expr", (isotope.ISOTOPE_TABLE[("H", 1)], isotope.ISOTOPE_TABLE[("H", 3)])
    )
    assert str(expr) == "test expr"

    assert expr.validForIsotopes(
        [isotope.ISOTOPE_TABLE[("H", 1)], isotope.ISOTOPE_TABLE[("H", 2)]]
    )
    assert not expr.validForIsotopes(
        [isotope.ISOTOPE_TABLE[("H", 1)], isotope.ISOTOPE_TABLE[("H", 3)]]
    )


def test_isotope_expression_sum():
    expr = isotope.SPCalIsotopeExpression.sumIsotopes(
        [isotope.ISOTOPE_TABLE[("H", 1)], isotope.ISOTOPE_TABLE[("H", 2)]]
    )
    assert expr.name == "ΣH"

    expr = isotope.SPCalIsotopeExpression.sumIsotopes(
        [isotope.ISOTOPE_TABLE[("H", 1)], isotope.ISOTOPE_TABLE[("He", 3)]]
    )
    assert expr.name == "Σ1H3He"


def test_isotope_comparisions():

    assert isotope.SPCalIsotopeBase("1") < isotope.SPCalIsotopeBase("2")

    assert isotope.SPCalIsotope.fromString("6Li") < isotope.SPCalIsotope.fromString(
        "7Li"
    )

    assert isotope.SPCalIsotope.fromString(
        "197Au"
    ) < isotope.SPCalIsotopeExpression.fromString("test", "+ 6Li 7Li")

    assert not isotope.SPCalIsotopeExpression.fromString(
        "test", "+ 6Li 7Li"
    ) < isotope.SPCalIsotope.fromString("197Au")

    assert isotope.SPCalIsotopeExpression.fromString(
        "b", "+ 6Li 7Li"
    ) > isotope.SPCalIsotopeExpression.fromString(
        "a", "+ 6Li 7Li"
    )
