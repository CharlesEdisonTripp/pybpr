from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence
from count_model.assessor.nb_factor import NBElement, NBFactor
from count_model.event_counter import EventCounter
from count_model.interaction import Interaction

import numpy

# class FactoredNaiveBayes():
#     def __call__(
#         self,
#         sources : List[Interaction], #
#         dst : Interaction, # user, action, product
#     ):


def compute_naive_bayes(
    prior: NBElement,
    factors: Iterable[NBFactor],
):
    prior_probability = prior.probability
    acc = numpy.log(numpy.array([prior_probability, 1.0 - prior_probability]))

    for factor in factors:
        acc += numpy.log(
            numpy.array(
                [
                    factor.likelihood,
                    factor.negative_likelihood,
                ]
            )
        )

    acc = numpy.exp(acc)
    p = acc[0] / numpy.sum(acc)
    print(f'nb: {acc} {p}')
    return max(1e-9, min(1.0 - 1e-9, p))
    


def compute_naive_bayes_with_evidence(
    prior: NBElement,
    likelihood_prior: NBFactor,
    event, # user A positively rating movie Y
    negative_event,
    evidence: Iterable, # user A positively rated movie X, and other ratings... 
    event_counter: EventCounter,
):
    return compute_naive_bayes(
        prior,
        (
            NBFactor(
                NBElement(
                    # event & evidence (e.g. view and click A and view and click B)
                    event_counter.get_count((event, e)),
                    # event (e.g. view and click A)
                    event_counter.get_count(event),
                ),
                NBElement(
                    # ~event & evidence (e.g. view A without click and view and click B)
                    event_counter.get_count((negative_event, e)),
                    # ~event (e.g. view A without click)
                    event_counter.get_count(negative_event),
                ),
            )
            for e in evidence
        ),
    )
