from dataclasses import dataclass

import numpy

from pybpr.count_model.assessor.binary_assessor import BinaryAssessor


@dataclass(slots=True)
class CombinedPosterior(BinaryAssessor):
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
        for (
            src,
            source_to_pos_dest,
            source_to_neg_dest,
            neg_source_to_pos_dest,
            neg_source_to_neg_dest,
        ) in source_counts:
            numerator += source_to_pos_dest.count
            denominator += source_to_pos_dest.count + source_to_neg_dest.count
        return numerator / denominator
