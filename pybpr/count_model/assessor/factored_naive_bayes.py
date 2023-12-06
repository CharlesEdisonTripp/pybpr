from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence
from pybpr.count_model.assessor.nb_factor import NBElement, NBFactor
from pybpr.count_model.event_counter import EventCounter
from pybpr.count_model.interaction import Interaction

import numpy


def compute_naive_bayes(
    prior_probability: float,
    factors: Iterable[NBFactor],
    bound: float = 1e-6,
):
    log_odds = numpy.log(prior_probability) - numpy.log(1.0 - prior_probability)

    for factor in factors:
        log_odds += numpy.log(factor.likelihood) - numpy.log(factor.negative_likelihood)
        # mathematically the same as: odds *= P(+A | +B) / P(+A | -B) = (P(+A & +B) / P(A & + B)) / (P(+A & -B) / P(A & -B))
    odds = numpy.exp(log_odds)
    p = odds / (1 + odds)  # P(+B | +A)
    return max(bound, min(1.0 - bound, p))
