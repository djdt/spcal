"""Pratt parsing.

Based on https://www.engr.mun.ca/~theo/Misc/pratt_parsing.htm
"""
import numpy as np
import re

from typing import Dict, List, Union


class ParserException(Exception):
    pass


class ReducerException(Exception):
    pass


class Expr(object):
    """Stores expressions for conversion to string."""

    def __init__(self, value: str, children: List["Expr"] = None):
        self.value = value
        self.children = children

    def __str__(self) -> str:
        if self.children is None:
            return self.value
        else:
            return f"{self.value} {' '.join([str(c) for c in self.children])}"


# Null Commands
class Null(object):
    """Base for non binding tokens."""

    rbp = -1

    def nud(self, parser: "Parser", tokens: List[str]) -> Expr:
        """Null denotation."""
        raise ParserException("Invalid token.")


class Parens(Null):
    """Parse input within parenthesis."""

    def nud(self, parser: "Parser", tokens: List[str]) -> Expr:
        expr = parser.parseExpr(tokens)
        if len(tokens) == 0 or tokens.pop(0) != ")":
            raise ParserException("Mismatched parenthesis.")
        return expr


class Value(Null):
    """Any value."""

    def __init__(self, value: str):
        self.value = value

    def nud(self, parser: "Parser", tokens: List[str]) -> Expr:
        return Expr(self.value)


class NaN(Null):
    """NaN value."""

    def nud(self, parser: "Parser", tokens: List[str]) -> Expr:
        return Expr("nan")


class Unary(Null):
    """Unary input."""

    def __init__(self, value: str, rbp: int):
        self.value = value
        self.rbp = rbp

    def nud(self, parser: "Parser", tokens: List[str]) -> Expr:
        expr = parser.parseExpr(tokens, self.rbp)
        return Expr(self.value, children=[expr])


class Binary(Null):
    """Binary input separated by `div`."""

    def __init__(self, value: str, div: str, rbp: int):
        self.value = value
        self.div = div
        self.rbp = rbp

    def nud(self, parser: "Parser", tokens: List[str]) -> Expr:
        expr = parser.parseExpr(tokens)
        if len(tokens) == 0 or tokens.pop(0) != self.div:
            raise ParserException(f"Missing '{self.div}' statement.")
        rexpr = parser.parseExpr(tokens, self.rbp)
        return Expr(self.value, children=[expr, rexpr])


class Ternary(Null):
    """Ternary input separated by `div` and `div2`."""

    def __init__(self, value: str, div: str, div2: str, rbp: int):
        self.value = value
        self.div = div
        self.div2 = div2
        self.rbp = rbp

    def nud(self, parser: "Parser", tokens: List[str]) -> Expr:
        lexpr = parser.parseExpr(tokens)
        if len(tokens) == 0 or tokens.pop(0) != self.div:
            raise ParserException(f"Missing '{self.div}' statement.")
        expr = parser.parseExpr(tokens)
        if len(tokens) == 0 or tokens.pop(0) != self.div2:
            raise ParserException(f"Missing '{self.div2}' statement.")
        rexpr = parser.parseExpr(tokens, self.rbp)
        return Expr(self.value, children=[lexpr, expr, rexpr])


class UnaryFunction(Unary):
    """Function with format 'func(<x>)'."""

    def __init__(self, value: str):
        super().__init__(value, 0)

    def nud(self, parser: "Parser", tokens: List[str]) -> Expr:
        if len(tokens) == 0 or tokens.pop(0) != "(":
            raise ParserException("Missing opening parenthesis.")
        result = super().nud(parser, tokens)
        if len(tokens) == 0 or tokens.pop(0) != ")":
            raise ParserException("Missing closing parenthesis.")
        return result


class BinaryFunction(Binary):
    """Function with format 'func(<x>, <y>)'."""

    def __init__(self, value: str):
        super().__init__(value, ",", 0)

    def nud(self, parser: "Parser", tokens: List[str]) -> Expr:
        if len(tokens) == 0 or tokens.pop(0) != "(":
            raise ParserException("Missing opening parenthesis.")
        result = super().nud(parser, tokens)
        if len(tokens) == 0 or tokens.pop(0) != ")":
            raise ParserException("Missing closing parenthesis.")
        return result


class TernaryFunction(Ternary):
    """Function with format 'func(<x>, <y>, <z>)'."""

    def __init__(self, value: str):
        super().__init__(value, ",", ",", 0)

    def nud(self, parser: "Parser", tokens: List[str]) -> Expr:
        if len(tokens) == 0 or tokens.pop(0) != "(":
            raise ParserException("Missing opening parenthesis.")
        result = super().nud(parser, tokens)
        if len(tokens) == 0 or tokens.pop(0) != ")":
            raise ParserException("Missing closing parenthesis.")
        return result


# Left Commands
class Left(object):
    """Base class for left binding operations."""

    lbp = -1

    @property
    def rbp(self):
        """The right binding power fo the token."""
        return self.lbp + 1

    def led(self, parser: "Parser", tokens: List[str], expr: Expr) -> Expr:
        """Left denotation, uses the current `expr`."""
        raise ParserException("Invalid token.")  # pragma: no cover


class LeftBinary(Left):
    """Left binary operation.

    Right binding if `right`.
    """

    def __init__(self, value: str, lbp: int, right: bool = False):
        self.value = value
        self.lbp = lbp
        self.right = right

    @property
    def rbp(self):
        return self.lbp + (0 if self.right else 1)

    def led(self, parser: "Parser", tokens: List[str], expr: Expr) -> Expr:
        rexpr = parser.parseExpr(tokens, self.rbp)
        return Expr(self.value, children=[expr, rexpr])


class LeftTernary(Left):
    """Left ternary input separted by `div`."""

    def __init__(self, value: str, div: str, lbp: int):
        self.value = value
        self.div = div
        self.lbp = lbp

    def led(self, parser: "Parser", tokens: List[str], lexpr: Expr) -> Expr:
        expr = parser.parseExpr(tokens)
        if len(tokens) == 0 or tokens.pop(0) != self.div:
            raise ParserException(f"Missing '{self.div}' statement.")
        rexpr = parser.parseExpr(tokens, self.rbp)
        return Expr(value=self.value, children=[lexpr, expr, rexpr])


class LeftIndex(Left):
    """Index value of left with '[]'."""

    def __init__(self, value: str, lbp: int):
        self.value = value
        self.lbp = lbp

    def led(self, parser: "Parser", tokens: List[str], expr: Expr) -> Expr:
        rexpr = parser.parseExpr(tokens, 0)
        if len(tokens) == 0 or tokens.pop(0) != "]":
            raise ParserException("Mismatched bracket ']'.")
        return Expr(self.value, children=[expr, rexpr])


class Parser(object):
    """Class for parsing inputs to an easily reduced string.

    Uses a series of regular expressions to convert input into tokens.
    These are then parsed using the tokens in `nulls` and `lefts`.
    To add functionality add tokens to these variables.

    Args:
        variables: tokens to consider as values

    Parameters:
        variables: list of value tokens
        nulls: dict of non-binding tokens
        lefts: dict of left-binding tokens

    See Also:
        `:func:pewpew.lib.pratt.Reducer`
    """

    function_token = "[a-z]+[a-zA-Z0-9_]*"
    null_token = "[\\[\\]\\(\\)\\,]|if|then|else"
    number_token = "\\d*\\.?\\d+(?:[eE][+\\-]?\\d+)?|nan"
    operator_token = "[+\\-\\*/^!=<>?:]+"
    base_tokens = "|".join([function_token, null_token, number_token, operator_token])

    def __init__(self, variables: List[str] = None):
        self.regexp_number = re.compile(Parser.number_token)
        self.regexp_tokenise = re.compile(f"\\s*({Parser.base_tokens})\\s*")

        self._variables: List[str] = []
        if variables is not None:
            self.variables = variables

        self.nulls: Dict[str, Null] = {
            "(": Parens(),
            "if": Ternary("?", "then", "else", 11),
            "nan": NaN(),
            "-": Unary("u-", 30),
        }
        self.lefts: Dict[str, Left] = {
            "?": LeftTernary("?", ":", 10),
            "<": LeftBinary("<", 10),
            "<=": LeftBinary("<=", 10),
            ">": LeftBinary(">", 10),
            ">=": LeftBinary(">=", 10),
            "=": LeftBinary("=", 10),
            "==": LeftBinary("=", 10),
            "!=": LeftBinary("!=", 10),
            "+": LeftBinary("+", 20),
            "-": LeftBinary("-", 20),
            "*": LeftBinary("*", 40),
            "/": LeftBinary("/", 40),
            "^": LeftBinary("^", 50, right=True),
            "[": LeftIndex("[", 80),
        }

    @property
    def variables(self) -> List[str]:
        return self._variables

    @variables.setter
    def variables(self, variables: List[str]) -> None:
        variable_token = "|".join(re.escape(v) for v in variables)
        self.regexp_tokenise = re.compile(
            f"\\s*({variable_token}|{Parser.base_tokens})\\s*"
        )
        self._variables = variables

    def getNull(self, token: str) -> Null:
        if token in self.nulls:
            return self.nulls[token]
        if token in self.variables or self.regexp_number.fullmatch(token) is not None:
            return Value(token)
        return Null()

    def getLeft(self, token: str) -> Left:
        if token in self.lefts:
            return self.lefts[token]
        return Left()

    def parseExpr(self, tokens: List[str], prec: int = 0) -> Expr:
        if len(tokens) == 0:
            raise ParserException("Unexpected end of input.")

        token = tokens.pop(0)
        cmd = self.getNull(token)
        expr = cmd.nud(self, tokens)
        while len(tokens) > 0:
            lcmd = self.getLeft(tokens[0])
            if prec > lcmd.lbp:
                break
            tokens.pop(0)
            expr = lcmd.led(self, tokens, expr)
        return expr

    def parse(self, string: str) -> str:
        """Parse the input string."""
        tokens = self.regexp_tokenise.findall(string)
        result = self.parseExpr(tokens)
        if len(tokens) != 0:
            raise ParserException(f"Unexpected input '{tokens[0]}'.")
        return str(result)


class Reducer(object):
    """Class for reducing preivously parsed inputs.

    Common operations are mapped to numpy ufuncs.

    Args:
        variables: dict mapping tokens to values

    Parameters:
        variables: dict of tokens and values
        operations: dict of (operation, number of inputs)

    See Also:
        `:func:pewpew.lib.pratt.Reducer`
    """

    def __init__(self, variables: dict = None):
        self._variables: Dict[str, Union[float, np.ndarray]] = {}

        if variables is not None:
            self.variables = variables

        self.operations = {
            "u-": (np.negative, 1),
            "+": (np.add, 2),
            "-": (np.subtract, 2),
            "*": (np.multiply, 2),
            "/": (np.divide, 2),
            "^": (np.power, 2),
            ">": (np.greater, 2),
            ">=": (np.greater_equal, 2),
            "<": (np.less, 2),
            "<=": (np.less_equal, 2),
            "=": (np.equal, 2),
            "!=": (np.not_equal, 2),
            "?": (np.where, 3),
            "[": (lambda x, i: x[int(i)], 2),
        }

    @property
    def variables(self) -> Dict[str, Union[float, np.ndarray]]:
        return self._variables

    @variables.setter
    def variables(self, variables: Dict[str, Union[float, np.ndarray]]) -> None:
        if any(" " in v for v in variables.keys()):
            raise ValueError("Spaces are not allowed in variable names!")
        self._variables = variables

    def reduceExpr(self, tokens: List[str]) -> Union[float, int, np.ndarray]:
        if len(tokens) == 0:
            raise ReducerException("Unexpected end of input.")
        token = tokens.pop(0)
        if token in self.operations:
            try:
                op, nargs = self.operations[token]
                args = [self.reduceExpr(tokens) for _ in range(nargs)]
                return op(*args)
            except (IndexError, TypeError):
                raise ReducerException(f"Unable to index '{token}'.")
            except (AttributeError, KeyError, ValueError):
                raise ReducerException(f"Invalid args for '{token}'.")
        elif token in self.variables:
            return self.variables[token]
        else:  # is a number
            try:
                if any(t in token for t in [".", "e", "E", "n"]):
                    return float(token)
                else:
                    return int(token)
            except ValueError:
                raise ReducerException(f"Unexpected input '{token}'.")

    def reduce(self, string: str) -> Union[float, np.ndarray]:
        """Reduce a parsed string to a value."""
        tokens = string.split(" ")
        result = self.reduceExpr(tokens)
        if len(tokens) != 0:
            raise ReducerException(f"Unexpected input '{tokens[0]}'.")
        return result
