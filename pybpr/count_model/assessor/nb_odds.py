# from functools import singledispatchmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class NBOdds:
    numerator: float
    denominator: float

    def __add__(self, other: "NBOdds") -> "NBOdds":
        return NBOdds(
            self.numerator + other.numerator,
            self.denominator + other.denominator,
        )

    def __mul__(self, other: float) -> "NBOdds":
        return NBOdds(
            self.numerator * other,
            self.denominator * other,
        )

    # @singledispatchmethod
    # def __mul__(self, other) -> "NBOdds":
    #     raise NotImplementedError()

    def __repr__(self):
        return self.__str__()

    def __str__(self) -> str:
        return f"({self.numerator} / {self.denominator} = {self.odds})"

    # def invert(self) -> "NBElement":
    #     return NBElement(self.denominator - self.numerator, self.denominator)

    def rescale(self, factor: float) -> "NBOdds":

        log_scaler = np.log(factor) - np.log(self.numerator + self.denominator)

        # equivalent to:
        # new numerator = (numerator / (numerator + denominator)) * factor
        # new denominator = (denominator / (numerator + denominator)) * factor
        return NBOdds(
            np.exp(np.log(self.numerator) + log_scaler),
            np.exp(np.log(self.denominator) + log_scaler),
        )

    @property
    def odds(self) -> float:
        return self.numerator / self.denominator

    @property
    def probability(self) -> float:
        return self.numerator / (self.denominator + self.numerator)

    @property
    def log_odds(self) -> float:
        return np.log(self.numerator) - np.log(self.denominator)


# @NBOdds.__mul__.register(float)
# def _(self, other: float) -> NBOdds:
#     return NBOdds(
#         self.numerator * other,
#         self.denominator * other,
#     )


# @NBOdds.__mul__.register(NBOdds)
# def _(self, other: NBOdds) -> NBOdds:
#     return NBOdds(
#         self.numerator * other.numerator,
#         self.denominator * other.denominator,
#     )
