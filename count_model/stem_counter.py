from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Optional, Union
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from count_model.feature_counter import FeatureCounter
from count_model.link_counter import LinkCounter


class StemCounter(FeatureCounter):
    get_string: Callable[[Any], str]
    tokenize: Callable[[str], Union[Iterable[str], Iterator[str]]]
    stem: Callable[[str], str]

    def __init__(
        self,
        link_counter: LinkCounter,
        get_action: Callable[[Any], Any],
        get_string: Callable[[Any], str],
        tokenize: Callable[[str], Union[Iterable[str], Iterator[str]]],
        stem: Callable[[str], str],
    ) -> None:
        self.get_string = get_string
        self.tokenize = tokenize
        self.stem = stem
        super().__init__(
            link_counter,
            get_action,
            lambda element: self._get_features(element),
        )

    def _get_features(
        self,
        element,
    ) -> Iterator:
        return (self.stem(token) for token in self.tokenize(self.get_string(element)))
