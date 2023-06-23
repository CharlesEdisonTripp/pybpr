from dataclasses import dataclass

from count_model.assessor.nb_element import NBElement


@dataclass
class NBFactor:
    positive_element: NBElement  # P(evidence | event)
    negative_element: NBElement  # P(evidence | !event)

    def __add__(self, other: "NBFactor") -> "NBFactor":
        return NBFactor(
            self.positive_element + other.positive_element,
            self.negative_element + other.negative_element,
        )

    def __repr__(self):
        return self.__str__()

    def __str__(self) -> str:
        return f"[{self.positive_element} : {self.negative_element}]"

    @property
    def likelihood(self) -> float:
        return self.positive_element.probability

    @property
    def negative_likelihood(self) -> float:
        return self.negative_element.probability

    def invert(self) -> "NBFactor":
        return NBFactor(self.negative_element.invert(), self.positive_element.invert())
