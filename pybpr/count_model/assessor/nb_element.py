from dataclasses import dataclass
import numpy as np


@dataclass
class NBElement:
    numerator: float
    denominator: float

    def __add__(self, other: "NBElement") -> "NBElement":
        return NBElement(
            self.numerator + other.numerator,
            self.denominator + other.denominator,
        )

    def __mul__(self, other: float) -> "NBElement":
        return NBElement(
            self.numerator * other,
            self.denominator * other,
        )

    def __repr__(self):
        return self.__str__()

    def __str__(self) -> str:
        return f"({self.numerator} / {self.denominator} = {self.probability})"

    # def invert(self) -> "NBElement":
    #     return NBElement(self.denominator - self.numerator, self.denominator)

    @property
    def probability(self) -> float:
        return self.numerator / self.denominator

    @property
    def odds(self) -> float:
        return self.numerator / (self.denominator - self.numerator)

    @property
    def log_odds(self) -> float:
        return np.log(self.numerator) - np.log(self.denominator - self.numerator)

    def rescaled(self, weight: float) -> "NBElement":
        return NBElement((self.numerator * weight) / self.denominator, weight)
