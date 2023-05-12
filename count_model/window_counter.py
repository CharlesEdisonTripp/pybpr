from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Tuple, Union
from count_model.count_model import CountModel
from count_model.link_counter import LinkCounter
from count_model.source_data import SourceData
from count_model.sequence_counter import SequenceCounter


@dataclass(slots=True)
class WindowCounter(SequenceCounter):
    window_size: int

    def observe_sequence(
        self,
        sequence: Union[Iterable, Iterator],
    ) -> None:
        link_counter = self.link_counter
        window = deque()
        for dst, dst_num in sequence:
            window.append((dst, dst_num))
            self.observe_window(window)
            while len(window) > self.window_size:
                window.popleft()

    @abstractmethod
    def observe_window(
        self,
        window: deque,
    ) -> None:
        pass

    def get_sequence_weights(
        self,
        sequence: Iterator,
    ) -> Any:
        link_counter = self.link_counter
        dests = {}
        denomenator = 0
        for src, num in sequence:
            source_data = link_counter.get_source_data(src)
            denomenator += source_data.total
            for dst, num in source_data.destination_counts:
                dests[dst] = dests.get(dst, 0) + num
        return dests, denomenator
