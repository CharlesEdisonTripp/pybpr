from dataclasses import dataclass

import numpy

from pybpr.count_model.assessor.binary_assessor import BinaryAssessor


@dataclass(slots=True)
class AvgPosterior(BinaryAssessor):
    prior_numerator: float
    prior_denominator: float
    feature_prior_numerator: float
    feature_prior_denomenator: float

    def assess(
        self,
        dst,
        dest_count,
        source_counts,
    ):
        p = (self.prior_numerator + dest_count.count) / (
            self.prior_denominator + dest_count.total
        )
        n = 1
        for (
            src,
            source_to_pos_dest,
            source_to_neg_dest,
            neg_source_to_pos_dest,
            neg_source_to_neg_dest,
        ) in source_counts:
            fp = (self.feature_prior_numerator + source_to_pos_dest.count) / (
                self.feature_prior_denomenator
                + source_to_pos_dest.count
                + source_to_neg_dest.count
            )
            p += fp
            n += 1
        return p / n
