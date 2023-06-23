from dataclasses import dataclass


@dataclass
class NBElement:
    numerator: float  # conditional odds numerator
    denominator: float  # conditional odds denominator

    def __add__(self, other: "NBElement") -> "NBElement":
        return NBElement(
            self.numerator + other.numerator,
            self.denominator + other.denominator,
        )

    def __repr__(self):
        return self.__str__()

    def __str__(self) -> str:
        return f"({self.numerator} / {self.denominator})"

    def invert(self) -> "NBElement":
        return NBElement(self.denominator - self.numerator, self.denominator)

    @property
    def probability(self) -> float:
        return self.numerator / (self.denominator)
