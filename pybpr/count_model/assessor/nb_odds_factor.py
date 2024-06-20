from dataclasses import dataclass


from pybpr.count_model.assessor.nb_odds import NBOdds


@dataclass
class NBOddsFactor:
    positive_element: NBOdds  # O(evidence | event)
    negative_element: NBOdds  # O(evidence | !event)

    def __add__(self, other: "NBOddsFactor") -> "NBOddsFactor":
        return NBOddsFactor(
            self.positive_element + other.positive_element,
            self.negative_element + other.negative_element,
        )

    def __mul__(self, other) -> "NBOddsFactor":
        return NBOddsFactor(
            self.positive_element * other,
            self.negative_element * other,
        )

    def __repr__(self):
        return self.__str__()

    def __str__(self) -> str:
        return (
            f"[{self.positive_element} : {self.negative_element} = {self.odds_ratio}]"
        )

    def invert(self) -> "NBOddsFactor":
        return NBOddsFactor(self.positive_element, self.negative_element)

    @property
    def odds_ratio(self) -> float:
        return self.positive_element.odds / self.negative_element.odds

    @property
    def log_odds_ratio(self) -> float:
        return self.positive_element.log_odds - self.negative_element.log_odds
