from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
import itertools
from typing import Any, Iterator, Tuple
from count_model.count_model import CountModel
from count_model.link_counter import LinkCounter
from count_model.source_data import SourceData
from count_model.sequence_counter import SequenceCounter
from count_model.window_counter import WindowCounter


@dataclass(slots=True)
class DestinationCounter(WindowCounter):

    def observe_window(
        self,
        window: deque,
    ) -> None:
        dst, dst_num = window[-1]
        for src, src_num in itertools.islice(window, 0, len(window)-1):
            self.link_counter.observe_link(src, dst, dst_num * src_num)
            
        