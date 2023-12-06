from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Tuple
from pybpr.count_model.link_count_data import LinkCountData

from pybpr.count_model.source_data import SourceData


@dataclass(slots=True)
class EventCounter:
    _map: Dict[Any, int] = field(default_factory=lambda: {})

    def observe(self, *event) -> None:
        self._map[event] = self.get_count(*event) + 1

    def get_count(self, *event) -> int:
        return self._map.get(event, 0)
