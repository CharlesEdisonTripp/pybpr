from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Tuple, Union
from pybpr.count_model.count_model import CountModel
from pybpr.count_model.link_counter import LinkCounter
from pybpr.count_model.source_data import SourceData


@dataclass(slots=True)
class SequenceCounter(ABC):
    link_counter: LinkCounter

    @abstractmethod
    def observe_sequence(
        self,
        sequence: Union[Iterable, Iterator],
    ) -> None:
        pass
