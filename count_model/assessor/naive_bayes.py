
from dataclasses import dataclass

import numpy

from count_model.assessor.binary_assessor import BinaryAssessor


@dataclass(slots=True)
class NaiveBayes(BinaryAssessor):
    prior_numerator:float
    prior_denominator:float
    pos_feature_prior_numerator:float
    pos_feature_prior_denominator:float
    neg_feature_prior_numerator:float
    neg_feature_prior_denominator:float

    def assess(self,
        dst,
        dest_count,
        source_counts,
    ):
        dest_prior = (self.prior_numerator + dest_count.count) / (
            self.prior_denominator + dest_count.total
        )
        pos_acc = numpy.log(dest_prior)
        neg_acc = numpy.log(1.0 - dest_prior)
        for source_count in source_counts:
            (
                src,
                src_to_dst,
                src_to_ndst,
                nsrc_to_dest,
                nsrc_to_ndst,
            ) = source_count
            # P(src | dst) -> # src to dst / (# total both src to dst)
            #  link_count(src, dst) / get_source_data(dst).total
            cond_prob = (self.pos_feature_prior_numerator + src_to_dst.count) / (
                self.pos_feature_prior_denominator
                + src_to_dst.count
                + nsrc_to_dest.count
            )
            pos_acc += numpy.log(cond_prob)

            # src to -dst / total +/- src to -dst
            neg_cond_prob = (self.neg_feature_prior_numerator + src_to_ndst.count) / (
                self.neg_feature_prior_denominator
                + src_to_ndst.count
                + nsrc_to_ndst.count
            )
            neg_acc += numpy.log(neg_cond_prob)
            # evidence = (feature_prior_numerator source_count.source_count.count / source_count.source_count.total)
            # ep += dest_prior * cond_prob
            # s = source_count.source_count
            # ep += np.log((s.count + feature_prior_numerator) / (feature_prior_denominator + s.total))
        pos = numpy.exp(pos_acc)
        neg = numpy.exp(neg_acc)
        p = pos / (pos + neg)
        # print(f'{p} {pos} {neg}')
        return max(1e-100, min(1.0, p))