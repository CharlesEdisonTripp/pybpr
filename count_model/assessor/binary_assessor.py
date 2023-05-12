from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy
from count_model.interaction import Interaction, Interactions

from count_model.link_count_data import LinkCountData
from count_model.link_counter import LinkCounter


@dataclass(slots=True)
class BinaryAssessor(ABC):
    event_counter: LinkCounter
    link_counter: LinkCounter

    
    def __call__(
        self,
        sources : Interactions,
        dst : Interaction,
    ):
        # dest_all_actors = dst.make_anonymous()
        ndst = dst.negative()

        event_count = self.event_counter.get_link_count(dst.object, dst)
        link_counter = self.link_counter
        source_counts = []
        for src in sources:
            nsrc = src.negative()

            source_counts.append(
                (
                    src,
                    link_counter.get_link_count(src, dst),
                    link_counter.get_link_count(src, ndst),
                    link_counter.get_link_count(nsrc, dst),
                    link_counter.get_link_count(nsrc, ndst),
                )
            )

        p = self.assess(
            dst,
            LinkCountData(event_count.count, event_count.total),
            source_counts,
        )
        # if not dst[1]:
        #     p = 1.0 - p
        # return numpy.log(p)
        return p
    
    @abstractmethod
    def assess(
        self,
        dst,
        dest_count,
        source_counts,
    ) -> float:
        pass


