import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# REGEX_ISOTOPE = re.compile("(\\d+)?([A-Z][a-z]?)(\\d+)?")
REGEX_ISOTOPE = re.compile("(?:(\\d+)([A-Z][a-z]?))|(?:([A-Z][a-z]?)(\\d+))")


@dataclass(frozen=True)
class SPCalIsotopeBase:
    name: str


@dataclass(frozen=True)
class SPCalIsotope(SPCalIsotopeBase):
    name: str
    isotope: int
    mass: float
    composition: float | None = None

    def __str__(self) -> str:
        if self.isotope == 0:  # pragma: no cover , unused?
            return f'"{self.name}"'
        return f"{self.isotope}{self.name}"

    def __repr__(self) -> str:  # pragma: no cover
        return f"SPCalIsotope({self.__str__()})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SPCalIsotope):  # pragma: no cover
            return False
        return self.name == other.name and self.isotope == other.isotope

    @property
    def symbol(self) -> str:
        return self.name

    @classmethod
    def fromString(cls, text: str) -> "SPCalIsotope":
        m = REGEX_ISOTOPE.fullmatch(text.strip())
        if m is not None:
            # symbol = m.group(2)
            if m.group(1) is not None and m.group(2) is not None:
                symbol, isotope = m.group(2), int(m.group(1))
            elif m.group(3) is not None and m.group(4) is not None:
                symbol, isotope = m.group(3), int(m.group(4))
            else:
                raise NameError(f"'{text}' is not a valid isotope")

            if (symbol, isotope) in ISOTOPE_TABLE:
                return ISOTOPE_TABLE[(symbol, isotope)]

        raise NameError(f"'{text}' is not a valid isotope")


@dataclass(frozen=True)
class SPCalIsotopeExpression(SPCalIsotopeBase):
    name: str
    tokens: tuple[str | SPCalIsotope, ...]

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:  # pragma: no cover
        return f"SPCalIsotopeExpression({self.name}: {self.tokens})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SPCalIsotopeExpression):  # pragma: no cover
            return False
        return all(x == y for x, y in zip(self.tokens, other.tokens))

    def validForIsotopes(self, isotopes: list[SPCalIsotope]) -> bool:
        iso_tokens = [token for token in self.tokens if isinstance(token, SPCalIsotope)]
        return all(iso in isotopes for iso in iso_tokens)

    @classmethod
    def sumIsotopes(cls, isotopes: list[SPCalIsotope]) -> "SPCalIsotopeExpression":
        if all(isotope.symbol == isotopes[0].symbol for isotope in isotopes):
            name = "Σ" + isotopes[0].symbol
        else:
            name = "Σ" + "".join([str(iso) for iso in isotopes])
        tokens: list[str | SPCalIsotope] = ["+"] * (len(isotopes) - 1)
        tokens.extend(isotopes)
        return cls(name, tuple(tokens))

    @classmethod
    def fromString(cls, name: str, text: str) -> "SPCalIsotopeExpression":
        tokens: list[str | SPCalIsotope] = []
        for token in text.split(" "):
            try:
                tokens.append(SPCalIsotope.fromString(token))
            except NameError:
                tokens.append(token)
        return cls(name, tuple(tokens))


# class SPCalIon(object):
#     def __init__(self, isotopes: list[SPCalIsotope], charge: int = 1):
#         self.isotopes = isotopes
#         self.charge = charge
#
#     def __str__(self) -> str:
#         return f"[{iso}]" for iso in self.isotopes
#     @property
#     def mass(self) -> float | None:
#         return sum(iso.mass for iso in self.isotopes)


ISOTOPE_TABLE = {
    ("H", 1): SPCalIsotope("H", 1, 1.00782503223, 0.999885),
    ("H", 2): SPCalIsotope("H", 2, 2.01410177812, 0.000115),
    ("H", 3): SPCalIsotope("H", 3, 3.0160492779, None),
    ("He", 3): SPCalIsotope("He", 3, 3.0160293201, 0.00000134),
    ("He", 4): SPCalIsotope("He", 4, 4.00260325413, 0.99999866),
    ("Li", 6): SPCalIsotope("Li", 6, 6.0151228874, 0.0759),
    ("Li", 7): SPCalIsotope("Li", 7, 7.0160034366, 0.9241),
    ("Be", 9): SPCalIsotope("Be", 9, 9.012183065, 1),
    ("B", 10): SPCalIsotope("B", 10, 10.01293695, 0.199),
    ("B", 11): SPCalIsotope("B", 11, 11.00930536, 0.801),
    ("C", 12): SPCalIsotope("C", 12, 12, 0.9893),
    ("C", 13): SPCalIsotope("C", 13, 13.00335483507, 0.0107),
    ("C", 14): SPCalIsotope("C", 14, 14.0032419884, None),
    ("N", 14): SPCalIsotope("N", 14, 14.00307400443, 0.99636),
    ("N", 15): SPCalIsotope("N", 15, 15.00010889888, 0.00364),
    ("O", 16): SPCalIsotope("O", 16, 15.99491461957, 0.99757),
    ("O", 17): SPCalIsotope("O", 17, 16.9991317565, 0.00038),
    ("O", 18): SPCalIsotope("O", 18, 17.99915961286, 0.00205),
    ("F", 19): SPCalIsotope("F", 19, 18.99840316273, 1),
    ("Ne", 20): SPCalIsotope("Ne", 20, 19.9924401762, 0.9048),
    ("Ne", 21): SPCalIsotope("Ne", 21, 20.993846685, 0.0027),
    ("Ne", 22): SPCalIsotope("Ne", 22, 21.991385114, 0.0925),
    ("Na", 23): SPCalIsotope("Na", 23, 22.989769282, 1),
    ("Mg", 24): SPCalIsotope("Mg", 24, 23.985041697, 0.7899),
    ("Mg", 25): SPCalIsotope("Mg", 25, 24.985836976, 0.1),
    ("Mg", 26): SPCalIsotope("Mg", 26, 25.982592968, 0.1101),
    ("Al", 27): SPCalIsotope("Al", 27, 26.98153853, 1),
    ("Si", 28): SPCalIsotope("Si", 28, 27.97692653465, 0.92223),
    ("Si", 29): SPCalIsotope("Si", 29, 28.9764946649, 0.04685),
    ("Si", 30): SPCalIsotope("Si", 30, 29.973770136, 0.03092),
    ("P", 31): SPCalIsotope("P", 31, 30.97376199842, 1),
    ("S", 32): SPCalIsotope("S", 32, 31.9720711744, 0.9499),
    ("S", 33): SPCalIsotope("S", 33, 32.9714589098, 0.0075),
    ("S", 34): SPCalIsotope("S", 34, 33.967867004, 0.0425),
    ("S", 36): SPCalIsotope("S", 36, 35.96708071, 0.0001),
    ("Cl", 35): SPCalIsotope("Cl", 35, 34.968852682, 0.7576),
    ("Cl", 37): SPCalIsotope("Cl", 37, 36.965902602, 0.2424),
    ("Ar", 36): SPCalIsotope("Ar", 36, 35.967545105, 0.003336),
    ("Ar", 38): SPCalIsotope("Ar", 38, 37.96273211, 0.000629),
    ("Ar", 40): SPCalIsotope("Ar", 40, 39.9623831237, 0.996035),
    ("K", 39): SPCalIsotope("K", 39, 38.9637064864, 0.932581),
    ("K", 40): SPCalIsotope("K", 40, 39.963998166, 0.000117),
    ("K", 41): SPCalIsotope("K", 41, 40.9618252579, 0.067302),
    ("Ca", 40): SPCalIsotope("Ca", 40, 39.962590863, 0.96941),
    ("Ca", 42): SPCalIsotope("Ca", 42, 41.95861783, 0.00647),
    ("Ca", 43): SPCalIsotope("Ca", 43, 42.95876644, 0.00135),
    ("Ca", 44): SPCalIsotope("Ca", 44, 43.95548156, 0.02086),
    ("Ca", 46): SPCalIsotope("Ca", 46, 45.953689, 0.00004),
    ("Ca", 48): SPCalIsotope("Ca", 48, 47.95252276, 0.00187),
    ("Sc", 45): SPCalIsotope("Sc", 45, 44.95590828, 1),
    ("Ti", 46): SPCalIsotope("Ti", 46, 45.95262772, 0.0825),
    ("Ti", 47): SPCalIsotope("Ti", 47, 46.95175879, 0.0744),
    ("Ti", 48): SPCalIsotope("Ti", 48, 47.94794198, 0.7372),
    ("Ti", 49): SPCalIsotope("Ti", 49, 48.94786568, 0.0541),
    ("Ti", 50): SPCalIsotope("Ti", 50, 49.94478689, 0.0518),
    ("V", 50): SPCalIsotope("V", 50, 49.94715601, 0.0025),
    ("V", 51): SPCalIsotope("V", 51, 50.94395704, 0.9975),
    ("Cr", 50): SPCalIsotope("Cr", 50, 49.94604183, 0.04345),
    ("Cr", 52): SPCalIsotope("Cr", 52, 51.94050623, 0.83789),
    ("Cr", 53): SPCalIsotope("Cr", 53, 52.94064815, 0.09501),
    ("Cr", 54): SPCalIsotope("Cr", 54, 53.93887916, 0.02365),
    ("Mn", 55): SPCalIsotope("Mn", 55, 54.93804391, 1),
    ("Fe", 54): SPCalIsotope("Fe", 54, 53.93960899, 0.05845),
    ("Fe", 56): SPCalIsotope("Fe", 56, 55.93493633, 0.91754),
    ("Fe", 57): SPCalIsotope("Fe", 57, 56.93539284, 0.02119),
    ("Fe", 58): SPCalIsotope("Fe", 58, 57.93327443, 0.00282),
    ("Co", 59): SPCalIsotope("Co", 59, 58.93319429, 1),
    ("Ni", 58): SPCalIsotope("Ni", 58, 57.93534241, 0.68077),
    ("Ni", 60): SPCalIsotope("Ni", 60, 59.93078588, 0.26223),
    ("Ni", 61): SPCalIsotope("Ni", 61, 60.93105557, 0.011399),
    ("Ni", 62): SPCalIsotope("Ni", 62, 61.92834537, 0.036346),
    ("Ni", 64): SPCalIsotope("Ni", 64, 63.92796682, 0.009255),
    ("Cu", 63): SPCalIsotope("Cu", 63, 62.92959772, 0.6915),
    ("Cu", 65): SPCalIsotope("Cu", 65, 64.9277897, 0.3085),
    ("Zn", 64): SPCalIsotope("Zn", 64, 63.92914201, 0.4917),
    ("Zn", 66): SPCalIsotope("Zn", 66, 65.92603381, 0.2773),
    ("Zn", 67): SPCalIsotope("Zn", 67, 66.92712775, 0.0404),
    ("Zn", 68): SPCalIsotope("Zn", 68, 67.92484455, 0.1845),
    ("Zn", 70): SPCalIsotope("Zn", 70, 69.9253192, 0.0061),
    ("Ga", 69): SPCalIsotope("Ga", 69, 68.9255735, 0.60108),
    ("Ga", 71): SPCalIsotope("Ga", 71, 70.92470258, 0.39892),
    ("Ge", 70): SPCalIsotope("Ge", 70, 69.92424875, 0.2057),
    ("Ge", 72): SPCalIsotope("Ge", 72, 71.922075826, 0.2745),
    ("Ge", 73): SPCalIsotope("Ge", 73, 72.923458956, 0.0775),
    ("Ge", 74): SPCalIsotope("Ge", 74, 73.921177761, 0.365),
    ("Ge", 76): SPCalIsotope("Ge", 76, 75.921402726, 0.0773),
    ("As", 75): SPCalIsotope("As", 75, 74.92159457, 1),
    ("Se", 74): SPCalIsotope("Se", 74, 73.922475934, 0.0089),
    ("Se", 76): SPCalIsotope("Se", 76, 75.919213704, 0.0937),
    ("Se", 77): SPCalIsotope("Se", 77, 76.919914154, 0.0763),
    ("Se", 78): SPCalIsotope("Se", 78, 77.91730928, 0.2377),
    ("Se", 80): SPCalIsotope("Se", 80, 79.9165218, 0.4961),
    ("Se", 82): SPCalIsotope("Se", 82, 81.9166995, 0.0873),
    ("Br", 79): SPCalIsotope("Br", 79, 78.9183376, 0.5069),
    ("Br", 81): SPCalIsotope("Br", 81, 80.9162897, 0.4931),
    ("Kr", 78): SPCalIsotope("Kr", 78, 77.92036494, 0.00355),
    ("Kr", 80): SPCalIsotope("Kr", 80, 79.91637808, 0.02286),
    ("Kr", 82): SPCalIsotope("Kr", 82, 81.91348273, 0.11593),
    ("Kr", 83): SPCalIsotope("Kr", 83, 82.91412716, 0.115),
    ("Kr", 84): SPCalIsotope("Kr", 84, 83.9114977282, 0.56987),
    ("Kr", 86): SPCalIsotope("Kr", 86, 85.9106106269, 0.17279),
    ("Rb", 85): SPCalIsotope("Rb", 85, 84.9117897379, 0.7217),
    ("Rb", 87): SPCalIsotope("Rb", 87, 86.909180531, 0.2783),
    ("Sr", 84): SPCalIsotope("Sr", 84, 83.9134191, 0.0056),
    ("Sr", 86): SPCalIsotope("Sr", 86, 85.9092606, 0.0986),
    ("Sr", 87): SPCalIsotope("Sr", 87, 86.9088775, 0.07),
    ("Sr", 88): SPCalIsotope("Sr", 88, 87.9056125, 0.8258),
    ("Y", 89): SPCalIsotope("Y", 89, 88.9058403, 1),
    ("Zr", 90): SPCalIsotope("Zr", 90, 89.9046977, 0.5145),
    ("Zr", 91): SPCalIsotope("Zr", 91, 90.9056396, 0.1122),
    ("Zr", 92): SPCalIsotope("Zr", 92, 91.9050347, 0.1715),
    ("Zr", 94): SPCalIsotope("Zr", 94, 93.9063108, 0.1738),
    ("Zr", 96): SPCalIsotope("Zr", 96, 95.9082714, 0.028),
    ("Nb", 93): SPCalIsotope("Nb", 93, 92.906373, 1),
    ("Mo", 92): SPCalIsotope("Mo", 92, 91.90680796, 0.1453),
    ("Mo", 94): SPCalIsotope("Mo", 94, 93.9050849, 0.0915),
    ("Mo", 95): SPCalIsotope("Mo", 95, 94.90583877, 0.1584),
    ("Mo", 96): SPCalIsotope("Mo", 96, 95.90467612, 0.1667),
    ("Mo", 97): SPCalIsotope("Mo", 97, 96.90601812, 0.096),
    ("Mo", 98): SPCalIsotope("Mo", 98, 97.90540482, 0.2439),
    ("Mo", 100): SPCalIsotope("Mo", 100, 99.9074718, 0.0982),
    ("Tc", 97): SPCalIsotope("Tc", 97, 96.9063667, None),
    ("Tc", 98): SPCalIsotope("Tc", 98, 97.9072124, None),
    ("Tc", 99): SPCalIsotope("Tc", 99, 98.9062508, None),
    ("Ru", 96): SPCalIsotope("Ru", 96, 95.90759025, 0.0554),
    ("Ru", 98): SPCalIsotope("Ru", 98, 97.9052868, 0.0187),
    ("Ru", 99): SPCalIsotope("Ru", 99, 98.9059341, 0.1276),
    ("Ru", 100): SPCalIsotope("Ru", 100, 99.9042143, 0.126),
    ("Ru", 101): SPCalIsotope("Ru", 101, 100.9055769, 0.1706),
    ("Ru", 102): SPCalIsotope("Ru", 102, 101.9043441, 0.3155),
    ("Ru", 104): SPCalIsotope("Ru", 104, 103.9054275, 0.1862),
    ("Rh", 103): SPCalIsotope("Rh", 103, 102.905498, 1),
    ("Pd", 102): SPCalIsotope("Pd", 102, 101.9056022, 0.0102),
    ("Pd", 104): SPCalIsotope("Pd", 104, 103.9040305, 0.1114),
    ("Pd", 105): SPCalIsotope("Pd", 105, 104.9050796, 0.2233),
    ("Pd", 106): SPCalIsotope("Pd", 106, 105.9034804, 0.2733),
    ("Pd", 108): SPCalIsotope("Pd", 108, 107.9038916, 0.2646),
    ("Pd", 110): SPCalIsotope("Pd", 110, 109.9051722, 0.1172),
    ("Ag", 107): SPCalIsotope("Ag", 107, 106.9050916, 0.51839),
    ("Ag", 109): SPCalIsotope("Ag", 109, 108.9047553, 0.48161),
    ("Cd", 106): SPCalIsotope("Cd", 106, 105.9064599, 0.0125),
    ("Cd", 108): SPCalIsotope("Cd", 108, 107.9041834, 0.0089),
    ("Cd", 110): SPCalIsotope("Cd", 110, 109.90300661, 0.1249),
    ("Cd", 111): SPCalIsotope("Cd", 111, 110.90418287, 0.128),
    ("Cd", 112): SPCalIsotope("Cd", 112, 111.90276287, 0.2413),
    ("Cd", 113): SPCalIsotope("Cd", 113, 112.90440813, 0.1222),
    ("Cd", 114): SPCalIsotope("Cd", 114, 113.90336509, 0.2873),
    ("Cd", 116): SPCalIsotope("Cd", 116, 115.90476315, 0.0749),
    ("In", 113): SPCalIsotope("In", 113, 112.90406184, 0.0429),
    ("In", 115): SPCalIsotope("In", 115, 114.903878776, 0.9571),
    ("Sn", 112): SPCalIsotope("Sn", 112, 111.90482387, 0.0097),
    ("Sn", 114): SPCalIsotope("Sn", 114, 113.9027827, 0.0066),
    ("Sn", 115): SPCalIsotope("Sn", 115, 114.903344699, 0.0034),
    ("Sn", 116): SPCalIsotope("Sn", 116, 115.9017428, 0.1454),
    ("Sn", 117): SPCalIsotope("Sn", 117, 116.90295398, 0.0768),
    ("Sn", 118): SPCalIsotope("Sn", 118, 117.90160657, 0.2422),
    ("Sn", 119): SPCalIsotope("Sn", 119, 118.90331117, 0.0859),
    ("Sn", 120): SPCalIsotope("Sn", 120, 119.90220163, 0.3258),
    ("Sn", 122): SPCalIsotope("Sn", 122, 121.9034438, 0.0463),
    ("Sn", 124): SPCalIsotope("Sn", 124, 123.9052766, 0.0579),
    ("Sb", 121): SPCalIsotope("Sb", 121, 120.903812, 0.5721),
    ("Sb", 123): SPCalIsotope("Sb", 123, 122.9042132, 0.4279),
    ("Te", 120): SPCalIsotope("Te", 120, 119.9040593, 0.0009),
    ("Te", 122): SPCalIsotope("Te", 122, 121.9030435, 0.0255),
    ("Te", 123): SPCalIsotope("Te", 123, 122.9042698, 0.0089),
    ("Te", 124): SPCalIsotope("Te", 124, 123.9028171, 0.0474),
    ("Te", 125): SPCalIsotope("Te", 125, 124.9044299, 0.0707),
    ("Te", 126): SPCalIsotope("Te", 126, 125.9033109, 0.1884),
    ("Te", 128): SPCalIsotope("Te", 128, 127.90446128, 0.3174),
    ("Te", 130): SPCalIsotope("Te", 130, 129.906222748, 0.3408),
    ("I", 127): SPCalIsotope("I", 127, 126.9044719, 1),
    ("Xe", 124): SPCalIsotope("Xe", 124, 123.905892, 0.000952),
    ("Xe", 126): SPCalIsotope("Xe", 126, 125.9042983, 0.00089),
    ("Xe", 128): SPCalIsotope("Xe", 128, 127.903531, 0.019102),
    ("Xe", 129): SPCalIsotope("Xe", 129, 128.9047808611, 0.264006),
    ("Xe", 130): SPCalIsotope("Xe", 130, 129.903509349, 0.04071),
    ("Xe", 131): SPCalIsotope("Xe", 131, 130.90508406, 0.212324),
    ("Xe", 132): SPCalIsotope("Xe", 132, 131.9041550856, 0.269086),
    ("Xe", 134): SPCalIsotope("Xe", 134, 133.90539466, 0.104357),
    ("Xe", 136): SPCalIsotope("Xe", 136, 135.907214484, 0.088573),
    ("Cs", 133): SPCalIsotope("Cs", 133, 132.905451961, 1),
    ("Ba", 130): SPCalIsotope("Ba", 130, 129.9063207, 0.00106),
    ("Ba", 132): SPCalIsotope("Ba", 132, 131.9050611, 0.00101),
    ("Ba", 134): SPCalIsotope("Ba", 134, 133.90450818, 0.02417),
    ("Ba", 135): SPCalIsotope("Ba", 135, 134.90568838, 0.06592),
    ("Ba", 136): SPCalIsotope("Ba", 136, 135.90457573, 0.07854),
    ("Ba", 137): SPCalIsotope("Ba", 137, 136.90582714, 0.11232),
    ("Ba", 138): SPCalIsotope("Ba", 138, 137.905247, 0.71698),
    ("La", 138): SPCalIsotope("La", 138, 137.9071149, 0.0008881),
    ("La", 139): SPCalIsotope("La", 139, 138.9063563, 0.9991119),
    ("Ce", 136): SPCalIsotope("Ce", 136, 135.90712921, 0.00185),
    ("Ce", 138): SPCalIsotope("Ce", 138, 137.905991, 0.00251),
    ("Ce", 140): SPCalIsotope("Ce", 140, 139.9054431, 0.8845),
    ("Ce", 142): SPCalIsotope("Ce", 142, 141.9092504, 0.11114),
    ("Pr", 141): SPCalIsotope("Pr", 141, 140.9076576, 1),
    ("Nd", 142): SPCalIsotope("Nd", 142, 141.907729, 0.27152),
    ("Nd", 143): SPCalIsotope("Nd", 143, 142.90982, 0.12174),
    ("Nd", 144): SPCalIsotope("Nd", 144, 143.910093, 0.23798),
    ("Nd", 145): SPCalIsotope("Nd", 145, 144.9125793, 0.08293),
    ("Nd", 146): SPCalIsotope("Nd", 146, 145.9131226, 0.17189),
    ("Nd", 148): SPCalIsotope("Nd", 148, 147.9168993, 0.05756),
    ("Nd", 150): SPCalIsotope("Nd", 150, 149.9209022, 0.05638),
    ("Pm", 145): SPCalIsotope("Pm", 145, 144.9127559, None),
    ("Pm", 147): SPCalIsotope("Pm", 147, 146.915145, None),
    ("Sm", 144): SPCalIsotope("Sm", 144, 143.9120065, 0.0307),
    ("Sm", 147): SPCalIsotope("Sm", 147, 146.9149044, 0.1499),
    ("Sm", 148): SPCalIsotope("Sm", 148, 147.9148292, 0.1124),
    ("Sm", 149): SPCalIsotope("Sm", 149, 148.9171921, 0.1382),
    ("Sm", 150): SPCalIsotope("Sm", 150, 149.9172829, 0.0738),
    ("Sm", 152): SPCalIsotope("Sm", 152, 151.9197397, 0.2675),
    ("Sm", 154): SPCalIsotope("Sm", 154, 153.9222169, 0.2275),
    ("Eu", 151): SPCalIsotope("Eu", 151, 150.9198578, 0.4781),
    ("Eu", 153): SPCalIsotope("Eu", 153, 152.921238, 0.5219),
    ("Gd", 152): SPCalIsotope("Gd", 152, 151.9197995, 0.002),
    ("Gd", 154): SPCalIsotope("Gd", 154, 153.9208741, 0.0218),
    ("Gd", 155): SPCalIsotope("Gd", 155, 154.9226305, 0.148),
    ("Gd", 156): SPCalIsotope("Gd", 156, 155.9221312, 0.2047),
    ("Gd", 157): SPCalIsotope("Gd", 157, 156.9239686, 0.1565),
    ("Gd", 158): SPCalIsotope("Gd", 158, 157.9241123, 0.2484),
    ("Gd", 160): SPCalIsotope("Gd", 160, 159.9270624, 0.2186),
    ("Tb", 159): SPCalIsotope("Tb", 159, 158.9253547, 1),
    ("Dy", 156): SPCalIsotope("Dy", 156, 155.9242847, 0.00056),
    ("Dy", 158): SPCalIsotope("Dy", 158, 157.9244159, 0.00095),
    ("Dy", 160): SPCalIsotope("Dy", 160, 159.9252046, 0.02329),
    ("Dy", 161): SPCalIsotope("Dy", 161, 160.9269405, 0.18889),
    ("Dy", 162): SPCalIsotope("Dy", 162, 161.9268056, 0.25475),
    ("Dy", 163): SPCalIsotope("Dy", 163, 162.9287383, 0.24896),
    ("Dy", 164): SPCalIsotope("Dy", 164, 163.9291819, 0.2826),
    ("Ho", 165): SPCalIsotope("Ho", 165, 164.9303288, 1),
    ("Er", 162): SPCalIsotope("Er", 162, 161.9287884, 0.00139),
    ("Er", 164): SPCalIsotope("Er", 164, 163.9292088, 0.01601),
    ("Er", 166): SPCalIsotope("Er", 166, 165.9302995, 0.33503),
    ("Er", 167): SPCalIsotope("Er", 167, 166.9320546, 0.22869),
    ("Er", 168): SPCalIsotope("Er", 168, 167.9323767, 0.26978),
    ("Er", 170): SPCalIsotope("Er", 170, 169.9354702, 0.1491),
    ("Tm", 169): SPCalIsotope("Tm", 169, 168.9342179, 1),
    ("Yb", 168): SPCalIsotope("Yb", 168, 167.9338896, 0.00123),
    ("Yb", 170): SPCalIsotope("Yb", 170, 169.9347664, 0.02982),
    ("Yb", 171): SPCalIsotope("Yb", 171, 170.9363302, 0.1409),
    ("Yb", 172): SPCalIsotope("Yb", 172, 171.9363859, 0.2168),
    ("Yb", 173): SPCalIsotope("Yb", 173, 172.9382151, 0.16103),
    ("Yb", 174): SPCalIsotope("Yb", 174, 173.9388664, 0.32026),
    ("Yb", 176): SPCalIsotope("Yb", 176, 175.9425764, 0.12996),
    ("Lu", 175): SPCalIsotope("Lu", 175, 174.9407752, 0.97401),
    ("Lu", 176): SPCalIsotope("Lu", 176, 175.9426897, 0.02599),
    ("Hf", 174): SPCalIsotope("Hf", 174, 173.9400461, 0.0016),
    ("Hf", 176): SPCalIsotope("Hf", 176, 175.9414076, 0.0526),
    ("Hf", 177): SPCalIsotope("Hf", 177, 176.9432277, 0.186),
    ("Hf", 178): SPCalIsotope("Hf", 178, 177.9437058, 0.2728),
    ("Hf", 179): SPCalIsotope("Hf", 179, 178.9458232, 0.1362),
    ("Hf", 180): SPCalIsotope("Hf", 180, 179.946557, 0.3508),
    ("Ta", 180): SPCalIsotope("Ta", 180, 179.9474648, 0.0001201),
    ("Ta", 181): SPCalIsotope("Ta", 181, 180.9479958, 0.9998799),
    ("W", 180): SPCalIsotope("W", 180, 179.9467108, 0.0012),
    ("W", 182): SPCalIsotope("W", 182, 181.94820394, 0.265),
    ("W", 183): SPCalIsotope("W", 183, 182.95022275, 0.1431),
    ("W", 184): SPCalIsotope("W", 184, 183.95093092, 0.3064),
    ("W", 186): SPCalIsotope("W", 186, 185.9543628, 0.2843),
    ("Re", 185): SPCalIsotope("Re", 185, 184.9529545, 0.374),
    ("Re", 187): SPCalIsotope("Re", 187, 186.9557501, 0.626),
    ("Os", 184): SPCalIsotope("Os", 184, 183.9524885, 0.0002),
    ("Os", 186): SPCalIsotope("Os", 186, 185.953835, 0.0159),
    ("Os", 187): SPCalIsotope("Os", 187, 186.9557474, 0.0196),
    ("Os", 188): SPCalIsotope("Os", 188, 187.9558352, 0.1324),
    ("Os", 189): SPCalIsotope("Os", 189, 188.9581442, 0.1615),
    ("Os", 190): SPCalIsotope("Os", 190, 189.9584437, 0.2626),
    ("Os", 192): SPCalIsotope("Os", 192, 191.961477, 0.4078),
    ("Ir", 191): SPCalIsotope("Ir", 191, 190.9605893, 0.373),
    ("Ir", 193): SPCalIsotope("Ir", 193, 192.9629216, 0.627),
    ("Pt", 190): SPCalIsotope("Pt", 190, 189.9599297, 0.00012),
    ("Pt", 192): SPCalIsotope("Pt", 192, 191.9610387, 0.00782),
    ("Pt", 194): SPCalIsotope("Pt", 194, 193.9626809, 0.3286),
    ("Pt", 195): SPCalIsotope("Pt", 195, 194.9647917, 0.3378),
    ("Pt", 196): SPCalIsotope("Pt", 196, 195.96495209, 0.2521),
    ("Pt", 198): SPCalIsotope("Pt", 198, 197.9678949, 0.07356),
    ("Au", 197): SPCalIsotope("Au", 197, 196.96656879, 1),
    ("Hg", 196): SPCalIsotope("Hg", 196, 195.9658326, 0.0015),
    ("Hg", 198): SPCalIsotope("Hg", 198, 197.9667686, 0.0997),
    ("Hg", 199): SPCalIsotope("Hg", 199, 198.96828064, 0.1687),
    ("Hg", 200): SPCalIsotope("Hg", 200, 199.96832659, 0.231),
    ("Hg", 201): SPCalIsotope("Hg", 201, 200.97030284, 0.1318),
    ("Hg", 202): SPCalIsotope("Hg", 202, 201.9706434, 0.2986),
    ("Hg", 204): SPCalIsotope("Hg", 204, 203.97349398, 0.0687),
    ("Tl", 203): SPCalIsotope("Tl", 203, 202.9723446, 0.2952),
    ("Tl", 205): SPCalIsotope("Tl", 205, 204.9744278, 0.7048),
    ("Pb", 204): SPCalIsotope("Pb", 204, 203.973044, 0.014),
    ("Pb", 206): SPCalIsotope("Pb", 206, 205.9744657, 0.241),
    ("Pb", 207): SPCalIsotope("Pb", 207, 206.9758973, 0.221),
    ("Pb", 208): SPCalIsotope("Pb", 208, 207.9766525, 0.524),
    ("Bi", 209): SPCalIsotope("Bi", 209, 208.9803991, 1),
    ("Po", 209): SPCalIsotope("Po", 209, 208.9824308, None),
    ("Po", 210): SPCalIsotope("Po", 210, 209.9828741, None),
    ("At", 210): SPCalIsotope("At", 210, 209.9871479, None),
    ("At", 211): SPCalIsotope("At", 211, 210.9874966, None),
    ("Rn", 211): SPCalIsotope("Rn", 211, 210.9906011, None),
    ("Rn", 220): SPCalIsotope("Rn", 220, 220.0113941, None),
    ("Rn", 222): SPCalIsotope("Rn", 222, 222.0175782, None),
    ("Fr", 223): SPCalIsotope("Fr", 223, 223.019736, None),
    ("Ra", 223): SPCalIsotope("Ra", 223, 223.0185023, None),
    ("Ra", 224): SPCalIsotope("Ra", 224, 224.020212, None),
    ("Ra", 226): SPCalIsotope("Ra", 226, 226.0254103, None),
    ("Ra", 228): SPCalIsotope("Ra", 228, 228.0310707, None),
    ("Ac", 227): SPCalIsotope("Ac", 227, 227.0277523, None),
    ("Ra", 230): SPCalIsotope("Ra", 230, 230.0331341, None),
    ("Th", 232): SPCalIsotope("Th", 232, 232.0380558, 1),
    ("Pa", 231): SPCalIsotope("Pa", 231, 231.0358842, 1),
    ("U", 233): SPCalIsotope("U", 233, 233.0396355, None),
    ("U", 234): SPCalIsotope("U", 234, 234.0409523, 0.000054),
    ("U", 235): SPCalIsotope("U", 235, 235.0439301, 0.007204),
    ("U", 236): SPCalIsotope("U", 236, 236.0455682, None),
    ("U", 238): SPCalIsotope("U", 238, 238.0507884, 0.992742),
    ("Np", 236): SPCalIsotope("Np", 236, 236.04657, None),
    ("Np", 237): SPCalIsotope("Np", 237, 237.0481736, None),
    ("Pu", 238): SPCalIsotope("Pu", 238, 238.0495601, None),
    ("Pu", 239): SPCalIsotope("Pu", 239, 239.0521636, None),
    ("Pu", 240): SPCalIsotope("Pu", 240, 240.0538138, None),
    ("Pu", 241): SPCalIsotope("Pu", 241, 241.0568517, None),
    ("Pu", 242): SPCalIsotope("Pu", 242, 242.0587428, None),
    ("Pu", 244): SPCalIsotope("Pu", 244, 244.0642053, None),
    ("Am", 241): SPCalIsotope("Am", 241, 241.0568293, None),
    ("Am", 243): SPCalIsotope("Am", 243, 243.0613813, None),
    ("Cm", 243): SPCalIsotope("Cm", 243, 243.0613893, None),
    ("Cm", 244): SPCalIsotope("Cm", 244, 244.0627528, None),
    ("Cm", 245): SPCalIsotope("Cm", 245, 245.0654915, None),
    ("Cm", 246): SPCalIsotope("Cm", 246, 246.0672238, None),
    ("Cm", 247): SPCalIsotope("Cm", 247, 247.0703541, None),
    ("Cm", 248): SPCalIsotope("Cm", 248, 248.0723499, None),
    ("Bk", 247): SPCalIsotope("Bk", 247, 247.0703073, None),
    ("Bk", 249): SPCalIsotope("Bk", 249, 249.0749877, None),
    ("Cf", 249): SPCalIsotope("Cf", 249, 249.0748539, None),
    ("Cf", 250): SPCalIsotope("Cf", 250, 250.0764062, None),
    ("Cf", 251): SPCalIsotope("Cf", 251, 251.0795886, None),
    ("Cf", 252): SPCalIsotope("Cf", 252, 252.0816272, None),
    ("Es", 252): SPCalIsotope("Es", 252, 252.08298, None),
    ("Fm", 257): SPCalIsotope("Fm", 257, 257.0951061, None),
    ("Md", 258): SPCalIsotope("Md", 258, 258.0984315, None),
    ("Md", 260): SPCalIsotope("Md", 260, 260.10365, None),
    ("No", 259): SPCalIsotope("No", 259, 259.10103, None),
    ("Lr", 262): SPCalIsotope("Lr", 262, 262.10961, None),
    ("Rf", 267): SPCalIsotope("Rf", 267, 267.12179, None),
    ("Db", 268): SPCalIsotope("Db", 268, 268.12567, None),
    ("Sg", 271): SPCalIsotope("Sg", 271, 271.13393, None),
    ("Bh", 272): SPCalIsotope("Bh", 272, 272.13826, None),
    ("Hs", 270): SPCalIsotope("Hs", 270, 270.13429, None),
    ("Mt", 276): SPCalIsotope("Mt", 276, 276.15159, None),
    ("Ds", 281): SPCalIsotope("Ds", 281, 281.16451, None),
    ("Rg", 280): SPCalIsotope("Rg", 280, 280.16514, None),
    ("Cn", 285): SPCalIsotope("Cn", 285, 285.17712, None),
    ("Nh", 284): SPCalIsotope("Nh", 284, 284.17873, None),
    ("Fl", 289): SPCalIsotope("Fl", 289, 289.19042, None),
    ("Mc", 288): SPCalIsotope("Mc", 288, 288.19274, None),
    ("Lv", 293): SPCalIsotope("Lv", 293, 293.20449, None),
    ("Ts", 292): SPCalIsotope("Ts", 292, 292.20746, None),
    ("Og", 294): SPCalIsotope("Og", 294, 294.21392, None),
}

RECOMMENDED_ISOTOPES = {
    # isotopes recommeneded by Agilent for analysis using He reaction gas
    "Li": 7,
    "Be": 9,
    "B": 11,
    "Na": 23,
    "Mg": 24,
    "Al": 27,
    "Si": 28,
    "P": 31,
    "Cl": 35,
    "K": 39,
    "Ca": 44,
    "Sc": 45,
    "Ti": 47,
    "V": 51,
    "Cr": 52,
    "Mn": 55,
    "Fe": 56,
    "Co": 59,
    "Ni": 60,
    "Cu": 63,
    "Zn": 66,
    "Ga": 71,
    "Ge": 72,
    "As": 75,
    "Se": 78,
    "Br": 79,
    "Rb": 85,
    "Sr": 88,
    "Y": 89,
    "Zr": 90,
    "Nb": 93,
    "Mo": 95,
    "Ru": 101,
    "Rh": 103,
    "Pd": 105,
    "Ag": 107,
    "Cd": 111,
    "In": 115,
    "Sn": 118,
    "Sb": 121,
    "Te": 125,
    "I": 127,
    "Cs": 133,
    "Ba": 137,
    "La": 139,
    "Ce": 140,
    "Pr": 141,
    "Nd": 146,
    "Sm": 147,
    "Eu": 153,
    "Gd": 157,
    "Tb": 159,
    "Dy": 163,
    "Ho": 165,
    "Er": 166,
    "Tm": 169,
    "Yb": 172,
    "Lu": 175,
    "Hf": 178,
    "Ta": 181,
    "W": 182,
    "Re": 185,
    "Os": 189,
    "Ir": 193,
    "Pt": 195,
    "Au": 197,
    "Hg": 202,
    "Tl": 205,
    "Pb": 208,
    "Bi": 209,
    "Th": 232,
    "U": 238,
}
