from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence
from pybpr.count_model.assessor.nb_factor import NBElement, NBFactor
from pybpr.count_model.assessor.nb_odds import NBOdds
from pybpr.count_model.assessor.nb_odds_factor import NBOddsFactor
from pybpr.count_model.event_counter import EventCounter
from pybpr.count_model.interaction import Interaction

import numpy


def compute_naive_bayes(
    prior_probability: NBElement,
    factors: Iterable[NBFactor],
    bound: float = 1e-9,
):
    log_odds = prior_probability.log_odds
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


def compute_naive_bayes_using_odds(
    prior: NBOdds,
    event_odds: NBOdds,
    factors: Iterable[NBOdds],
    bound: float = 1e-9,
):
    """
    O(f1 | click, view)
        = P(f1 | view, click) / P(f1 | view, pass)
        = (P(f1, click | view) / P(click | view)) / (P(f1, pass | view) / P(pass | view))
            ~= (#(feature clicked) / #(clicks)) / (#(feature passed) / #(passes))
        = (P(f1, click | view) / P(f1, pass | view)) / (P(click | view) / P(pass | view))
        ~= (#(f1, click) / #(f1, pass)) / (#(click) / #(pass))
    """

    log_odds = prior.log_odds
    event_log_odds = event_odds.log_odds
    for factor_odds in factors:
        log_odds += factor_odds.log_odds - event_log_odds
    
    posterior_odds = numpy.exp(log_odds)
    probability = posterior_odds / (1.0 + posterior_odds)
    return max(bound, min(1.0 - bound, probability))
