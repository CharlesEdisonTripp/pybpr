import pandas as pd
import functools
import os

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
import sys
import numpy as np


def load_click_segment(filename: str):  # , product_id_map: dict):
    print(f"Loading {filename}...")

    # segment_offset = segment_number << (32 - 8)

    _stemmer = PorterStemmer()

    _url_prefix_exp = re.compile(r"https://((\w|\.)+)/(:?(\w+)/)(.*)")
    _null_result = (-1, tuple())

    _category_map = {
        None: 0,
        "s": 1,
        "c": 2,
    }

    @functools.lru_cache(maxsize=8)
    def tokenize_and_stem(s: str):
        return tuple(
            (
                sys.intern(s)
                for s in (
                    _stemmer.stem(token)
                    for token in nltk.tokenize.word_tokenize(s.replace("+", " "))
                )
                if len(s) > 0
            )
        )

    @functools.lru_cache(maxsize=8)
    def decompose_url(s: str):
        if s is None or len(s) == 0:
            return _null_result
        url_match = _url_prefix_exp.fullmatch(s)
        if url_match is None:
            return _null_result

        url_groups = url_match.groups()
        # domain = url_groups[0]
        category = _category_map.get(url_groups[3], -1)
        keywords = url_groups[4]
        keyword_tokens = tokenize_and_stem(str(keywords))
        return (category, keyword_tokens)

    segment = pd.read_parquet(filename)
    # print(segment.columns.to_list())
    # segment.set_index(["timestamp", "user_id", "cleaned_url"], inplace=True, drop=True)
    # segment["seq"] = np.arange(0, 0 + len(segment.index), dtype=np.uint32)
    # product_numbers = (
    #     segment["product_id"].map(product_id_map.get).astype(np.uint32).to_numpy()
    # )

    segment = (
        segment.groupby(["date_created", "user_id", "cleaned_url"])["product_id"]
        .apply(lambda g: np.array(g, dtype=np.int64))
        .to_frame()
    )
    segment.reset_index(inplace=True)
    # print(query_groups.head(4))

    # query_groups = segment.groupby(["timestamp", "user_id", "cleaned_url"])[
    #     "seq"
    # ].apply(lambda g: np.copy(product_numbers[g.iloc[0] : g.iloc[-1] + 1]))
    # print(query_groups.sample(10))

    # segment["is_click"] = segment["is_click"].astype(np.int8)
    # segment["user_id"] = segment["user_id"].astype("category")

    segment["category"], segment["search_terms"] = zip(
        *segment["cleaned_url"].map(decompose_url)
    )
    segment.drop(["cleaned_url"], axis=1, inplace=True)

    # # query_groups.

    # segment["category"] = segment["category"].astype("category")
    # segment["search_terms"] = segment["search_terms"].astype("category")

    # segment["product_number"] = (
    #     segment["product_id"].map(lambda pid: product_id_map[pid]).astype(np.uint32)
    # )

    # # segment.set_index(["date_created", "user_id", "product_id"], inplace=True)
    # # segment.sort_index(inplace=True)

    print(f'"{filename}" loaded.')
    # return segment
    return segment


def postprocess_click_segment(future, keyword_map):
    segment = future.get()
    # cat = segment["search_terms"].cat
    # cat.rename_categories(
    #     [intern_search_stems(kws, keyword_map) for kws in cat.categories]
    # )
    return segment


def load_clicks(data_path, pool):
    import os

    # import pathos.multiprocessing as mp
    # import pathos

    # with pathos.pools.ParallelPool(max(1, mp.cpu_count() - 1)) as pool:
    print(f"Submitting file load tasks...")

    click_futures = [
        pool.apipe(
            load_click_segment,
            file,
        )
        for file in (
            os.path.join(data_path, f"Clicks_{str(ix).zfill(4)}_part_00.parquet")
            for ix in range(80)
        )
    ]
    print(f"Submitted file load tasks.")

    keyword_map = {}
    # import pandas as pd

    clicks = pd.concat(
        (postprocess_click_segment(f, keyword_map) for f in click_futures),
        axis=0,
        ignore_index=False,
        copy=False,
    )
    print(f"Concatenated.")
    del click_futures
    # for column in ["category", "search_terms"]:
    #     clicks[column] = clicks[column].astype("category")
    # clicks.sort_index(inplace=True)
    print(f"Loaded {len(clicks)} clicks.")
    return clicks
