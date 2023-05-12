from dataclasses import dataclass

import numpy

from count_model.assessor.binary_assessor import BinaryAssessor


@dataclass(slots=True)
class BayesPosterior(BinaryAssessor):
    prior_numerator: float
    prior_denominator: float

    def assess(
        self,
        dst,
        dest_count,
        source_counts,
    ):
        numerator = self.prior_numerator + dest_count.count
        denominator = self.prior_denominator + dest_count.total
        return numerator / denominator
