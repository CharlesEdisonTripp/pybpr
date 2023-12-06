from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Optional, Union
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from pybpr.count_model.feature_counter import FeatureCounter
from pybpr.count_model.link_counter import LinkCounter
from pybpr.count_model.stem_counter import StemCounter


class PorterStemCounter(StemCounter):
    def __init__(
        self,
        link_counter: LinkCounter,
        get_action: Callable[[Any], Any],
        get_string: Callable[[Any], str],
    ) -> None:
        stemmer = PorterStemmer()
        super().__init__(
            link_counter,
            get_action,
            get_string,
            nltk.tokenize.word_tokenize,
            lambda token: stemmer.stem(token),
        )
