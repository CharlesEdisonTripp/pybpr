from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Tuple, Union
from pybpr.count_model.count_model import CountModel
from pybpr.count_model.link_count_data import LinkCountData
from pybpr.count_model.link_counter import LinkCounter
from pybpr.count_model.source_data import SourceData
from pybpr.count_model.sequence_counter import SequenceCounter


@dataclass(slots=True)
class PermutationCounter(SequenceCounter):
    def observe_sequence(
        self,
        sequence: Union[Iterable, Iterator],
    ) -> None:
        link_counter = self.link_counter
        elements = tuple(sequence)
        for src in elements:
            for dst in elements:
                link_counter.observe_link(src, dst)

    def get_sequence_weights(
        self,
        sequence: Iterator,
    ) -> Any:
        link_counter = self.link_counter
        dests = {}
        denomenator = 0
        for src in sequence:
            source_data = link_counter.get_source_data(src)
            denomenator += source_data.total
            for dst, num in source_data.destination_counts:
                dests[dst] = dests.get(dst, 0) + num
        return dests, denomenator

    def get_link_counts(
        self,
        sequence: Iterator,
        dst,
    ) -> Any:
        return (self.link_counter.get_link_count(src, dst) for src in sequence)
