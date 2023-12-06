from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Tuple
from pybpr.count_model.link_count_data import LinkCountData

from pybpr.count_model.source_data import SourceData


@dataclass(slots=True)
class LinkCounter:
    __link_map: Dict[Any, SourceData] = field(default_factory=lambda: {})

    def __init__(self) -> None:
        self.__link_map = {}

    def observe_link(
        self,
        src,
        dst,
        num: int = 1,
    ) -> None:
        self.observe_destination(
            dst,
            self.get_source_data(src),
            num,
        )

    def get_source_data(
        self,
        src,
    ) -> SourceData:
        # TODO: if this is called a lot to query unregistered sources we should add a method that does not add it to the link map
        link_map = self.__link_map
        source_data = link_map.get(src, None)
        if source_data is None:
            source_data = SourceData()
            link_map[src] = source_data
        return source_data

    def get_link_count(
        self,
        src,
        dst,
    ) -> LinkCountData:
        source_data = self.__link_map.get(
            src,
            SourceData(0, {}),
        )
        return LinkCountData(
            source_data.destination_counts.get(
                dst,
                0,
            ),
            source_data.total,
        )

    def observe_destination(
        self,
        dst,
        link_data: SourceData,
        num: int,
    ) -> None:
        link_data.total += num
        destination_counts = link_data.destination_counts
        destination_counts[dst] = num + destination_counts.get(dst, 0)

    def get_sequence_weights(
        self,
        sequence: Iterator,
    ) -> Any:
        dests = {}
        denomenator = 0
        for src, num in sequence:
            source_data = self.get_source_data(src)
            denomenator += source_data.total
            for dst, num in source_data.destination_counts:
                dests[dst] = dests.get(dst, 0) + num
        return dests, denomenator
