from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence
from pybpr.count_model.assessor.nb_factor import NBElement, NBFactor
from pybpr.count_model.event_counter import EventCounter
from pybpr.count_model.interaction import Interaction

import numpy


def compute_naive_bayes(
    prior_probability: float,
    factors: Iterable[NBFactor],
    bound: float = 1e-9,
):
    log_odds = numpy.log(prior_probability) - numpy.log(1.0 - prior_probability)
    # print(f"prior log_odds: {log_odds}")
    for factor in factors:
        # print(f"factor: {factor}")
        # log_odds += numpy.log(factor.likelihood) - numpy.log(factor.negative_likelihood)
        log_odds += (
            numpy.log(factor.positive_element.numerator)
            - numpy.log(factor.positive_element.denominator)
        ) - (
            numpy.log(factor.negative_element.numerator)
            - numpy.log(factor.negative_element.denominator)
        )
        # mathematically the same as: odds *= P(+A | +B) / P(+A | -B) = (P(+A & +B) / P(A & + B)) / (P(+A & -B) / P(A & -B))
    # print(f"log_odds: {log_odds}")
    odds = numpy.exp(log_odds)
    p = odds / (1 + odds)  # P(+B | +A)
    return max(bound, min(1.0 - bound, p))
