from spcal.isotope import ISOTOPE_TABLE


def text_for_mz(mz: float) -> str:
    text = f"{mz:.2f}"
    possible_isotopes = [
        iso
        for iso in ISOTOPE_TABLE.values()
        if abs(iso.mass - mz) < 0.1
        and iso.composition is not None
        and iso.composition > 0.05
    ]
    if len(possible_isotopes) > 0:
        text += "(" + ",".join(iso.symbol for iso in possible_isotopes) + ")"
    return text
