from spcal.processing import SPCalProcessingResult
from spcal.isotope import SPCalIsotope

from spcal.pratt import Parser, Reducer


class SPCalExpression(object):
    def __init__(self, tokens: list[str | SPCalIsotope]):
        self.tokens = tokens

    def validForResults(
        self, results: dict[SPCalIsotope, SPCalProcessingResult]
    ) -> bool:
        return all(
            token in results for token in self.tokens if isinstance(token, SPCalIsotope)
        )


class SPCalCalculator(object):
    def __init__(self):
        self.expressions: list[SPCalExpression] = []
        self.equations: list[SPCalEquation] = []
        self.processing_results: dict[SPCalIsotope,]

    def reduceForResults(
        self, results: dict[SPCalIsotope, SPCalProcessingResult]
    ) -> list[SPCalProcessingResult]:
        reducer = Reducer(
            variables={isotope: result.signals for isotope, result in results.items()}
        )
        calculated_results = []
        for expr in self.expressions:
            if expr.validForResults(results):
                calculated_results.append()
