from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Tuple, Union
from count_model.count_model import CountModel
from count_model.link_count_data import LinkCountData
from count_model.link_counter import LinkCounter
from count_model.source_data import SourceData
from count_model.sequence_counter import SequenceCounter


@dataclass(slots=True)
class FeatureCounter(SequenceCounter):
    # feature_counter: LinkCounter
    get_action: Callable[[Any], Any]
    get_features: Callable[[Any], Union[Iterable, Iterator]]

    def observe_sequence(
        self,
        sequence: Union[Iterable, Iterator],
    ) -> None:
        link_counter = self.link_counter
        # feature_counter = self.feature_counter
        for element in sequence:
            features = [self.get_features(element)]
            action = self.get_action(element)
            for feature in features:
                # count feature -> action links
                link_counter.observe_link(feature, action)

                # # feature co-occourances
                # for second_feature in features:
                #     feature_counter.observe_link(feature, second_feature)

                # # TODO: what about feature->feature counts? (same element and element-to-element)
