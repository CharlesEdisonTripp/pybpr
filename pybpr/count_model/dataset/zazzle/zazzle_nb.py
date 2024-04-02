import functools
import os
import re
import sys
from collections import deque
from uuid import UUID
import numpy as np
import pandas as pd
from subprocess import call
import matplotlib.pyplot as plt
from functools import partial
from dataclasses import dataclass
from pprint import pprint
from typing import (
    Any,
    Generator,
    List,
    MutableMapping,
    Set,
    Tuple,
    Union,
    Optional,
    Dict,
)

import pathos.multiprocessing as mp
import pathos
import gc

import polars as pl

# import dask.dataframe as dd

# import swifter

# swifter.set_defaults(
#     npartitions=32,
#     dask_threshold=1,
#     scheduler="processes",
#     progress_bar=True,
#     progress_bar_desc=None,
#     allow_dask_on_strings=True,
#     force_parallel=True,
# )


def intern(mapping: MutableMapping, value) -> int:
    return mapping.setdefault(value, len(mapping))


def intern_values(
    ds: pd.Series,
    mapping: dict[Any, int],
    dtype=np.int64,
) -> pd.Series:
    return ds.map(lambda value: intern(mapping, value)).astype(dtype)


def intern_user_id(
    df: pd.DataFrame,
    user_id_map: dict[str, int],
):
    df["user_id"] = intern_values(df["user_id"], user_id_map)


import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

_stemmer = PorterStemmer()


def stem_token(token):
    return _stemmer.stem(token)


# @functools.lru_cache(maxsize=128)
def stem(tokens):
    return [_stemmer.stem(token) for token in tokens]


@functools.lru_cache(maxsize=8)
def tokenize_and_stem(s: str) -> Tuple[str, ...]:
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


_url_prefix_exp = re.compile(r"https://((\w|\.)+)/(:?(\w+)/)(.*)")
_null_result = (-1, tuple())

_category_map = {
    None: 0,
    "s": 1,
    "c": 2,
}


@functools.lru_cache(maxsize=8)
def decompose_url(s: str) -> Tuple[int, Tuple[str, ...]]:
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


def map_orderitems_df(orderitems, user_id_map):
    intern_user_id(orderitems, user_id_map)
    return orderitems


def collect():
    for i in range(3):
        for gen in range(3):
            gc.collect(generation=gen)


def intern_stems(stems):
    return tuple((sys.intern(s) for s in stems))


# @functools.lru_cache(maxsize=32)
def intern_search_stems(kws, keyword_map):
    # return tuple((sys.intern(kw) for kw in kws))
    res = keyword_map.get(kws, None)
    if res is not None:
        return res
    kws = intern_stems(kws)
    keyword_map[kws] = kws
    return kws


def prepare_orderitem_segment(filename: str, product_id_map: dict):
    segment = pd.read_parquet(filename)
    segment["product_number"] = (
        segment["product_id"].map(lambda pid: product_id_map[pid]).astype(np.uint32)
    )
    collect()
    print(f'"{filename}" loaded.')
    return segment


def prepare_click_segment(filename: str, product_id_map: dict):
    # segment_offset = segment_number << (32 - 8)

    segment = pd.read_parquet(filename)
    segment.set_index(["timestamp", "user_id", "cleaned_url"], inplace=True, drop=True)
    segment["seq"] = np.arange(0, 0 + len(segment.index), dtype=np.uint32)

    segment.groupby(["timestamp", "user_id", "cleaned_url"])

    query_groups = segment.groupby(["timestamp", "user_id", "cleaned_url"])[
        "seq"
    ].apply(lambda g: (g.iloc[0], g.iloc[-1]))
    print(query_groups.sample(10))

    segment["is_click"] = segment["is_click"].astype(np.int8)
    segment["user_id"] = segment["user_id"].astype("category")

    segment["category"], segment["search_terms"] = zip(
        *segment["cleaned_url"].map(decompose_url)
    )
    segment.drop(["cleaned_url"], axis=1, inplace=True)

    # query_groups.

    segment["category"] = segment["category"].astype("category")
    segment["search_terms"] = segment["search_terms"].astype("category")

    segment["product_number"] = (
        segment["product_id"].map(lambda pid: product_id_map[pid]).astype(np.uint32)
    )

    # segment.set_index(["date_created", "user_id", "product_id"], inplace=True)
    # segment.sort_index(inplace=True)

    collect()
    print(f'"{filename}" loaded.')
    return segment


def postprocess_click_segment(future, keyword_map):
    segment = future.get()
    cat = segment["search_terms"].cat
    cat.rename_categories(
        [intern_search_stems(kws, keyword_map) for kws in cat.categories]
    )
    return segment


def prepare_product_segment(filename):
    segment = pd.read_parquet(filename)

    segment["product_type"] = segment["product_type"].astype("category")
    segment["subject"] = segment["subject"].astype("category")
    segment["title_terms"] = segment["title"].map(tokenize_and_stem)
    segment["is_customizable"] = segment["is_customizable"].astype(np.int8)
    # segment["description_stems"] = segment["long_description"].map(tokenize_and_stem)

    # segment.set_index("product_id", inplace=True)

    collect()
    print(f'"{filename}" loaded.')
    return segment


def postprocess_product_segment(future, product_id_map):
    segment = future.get()
    segment["title_terms"] = segment["title_terms"].map(intern_stems)
    # print(segment.sample(10))
    segment["product_number"] = (
        segment["product_id"]
        .map(lambda pid: product_id_map.setdefault(pid, len(product_id_map)))
        .astype(np.uint32)
    )
    return segment


def load_products(data_path, pool):
    # with pathos.pools.ParallelPool(max(1, mp.cpu_count() - 1)) as pool:
    product_futures = [
        pool.apipe(prepare_product_segment, file)
        for file in (
            os.path.join(data_path, f"Products_{str(ix).zfill(4)}_part_00.parquet")
            for ix in range(80)
        )
    ]

    product_id_map = {}

    products = pd.concat(
        (postprocess_product_segment(f, product_id_map) for f in product_futures),
        axis=0,
        ignore_index=True,
        copy=False,
    )
    del product_futures
    for column in ["product_type", "subject"]:
        products[column] = products[column].astype("category")
    collect()
    print(f"Loaded {len(products)} products.")
    return products


# def load_clicks(data_path, product_id_map):
#     with pathos.pools.ParallelPool(max(1, mp.cpu_count() - 1)) as pool:
#         from pybpr.count_model.dataset.zazzle.load_click_segment import (
#             load_click_segment,
#         )

#         print(f"Submitting file load tasks...")

#         click_futures = [
#             pool.apipe(
#                 lambda filename: load_click_segment(filename, product_id_map),
#                 file,
#             )
#             for file in (
#                 os.path.join(data_path, f"Clicks_{str(ix).zfill(4)}_part_00.parquet")
#                 for ix in range(80)
#             )
#         ]
#         print(f"Submitted file load tasks.")

#         keyword_map = {}
#         clicks = pd.concat(
#             (postprocess_click_segment(f, keyword_map) for f in click_futures),
#             axis=0,
#             ignore_index=False,
#             copy=False,
#         )
#         print(f"Concatenated.")
#         del click_futures
#         for column in ["category", "search_terms"]:
#             clicks[column] = clicks[column].astype("category")
#         clicks.sort_index(inplace=True)
#         collect()
#         print(f"Loaded {len(clicks)} clicks.")


def load_zazzle_data(data_path):
    with pathos.pools.ParallelPool(max(1, mp.cpu_count() - 1)) as pool:
        product_futures = [
            pool.apipe(prepare_product_segment, file)
            for file in (
                os.path.join(data_path, f"Products_{str(ix).zfill(4)}_part_00.parquet")
                for ix in range(80)
            )
        ]

        product_id_map = {}

        products = pd.concat(
            (postprocess_product_segment(f, product_id_map) for f in product_futures),
            axis=0,
            ignore_index=True,
            copy=False,
        )
        del product_futures
        for column in ["product_type", "subject"]:
            products[column] = products[column].astype("category")
        collect()
        print(f"Loaded {len(products)} products.")

        click_futures = [
            pool.apipe(
                lambda filename: prepare_click_segment(filename, product_id_map),
                file,
            )
            for file in (
                os.path.join(data_path, f"Clicks_{str(ix).zfill(4)}_part_00.parquet")
                for ix in range(80)
            )
        ]

        keyword_map = {}
        clicks = pd.concat(
            (postprocess_click_segment(f, keyword_map) for f in click_futures),
            axis=0,
            ignore_index=False,
            copy=False,
        )
        del click_futures
        for column in ["category", "search_terms"]:
            clicks[column] = clicks[column].astype("category")
        clicks.sort_index(inplace=True)
        collect()
        print(f"Loaded {len(clicks)} clicks.")

        orderitem_futures = [
            pool.apipe(
                lambda filename: prepare_orderitem_segment(filename, product_id_map),
                file,
            )
            for file in (
                os.path.join(
                    data_path, f"OrderItems_{str(ix).zfill(4)}_part_00.parquet"
                )
                for ix in range(80)
            )
        ]
        orderitems = pd.concat(
            (f.get() for f in orderitem_futures),
            axis=0,
            ignore_index=True,
            copy=False,
        )
        del orderitem_futures
        collect()
        print(f"Loaded {len(orderitems)} orderitems.")

        # user_ids = set(clicks["user_id"].unique())
        # user_ids.update(orderitems["user_id"].unique())
        # user_ids_dtype = pd.api.types.CategoricalDtype(user_ids)
        # del user_ids
        # for df in [clicks, orderitems]:
        #     df["user_id"] = df["user_id"].astype(user_ids_dtype)
        # collect()

    # files = [
    #     os.path.join(data_path, f"OrderItems_{str(ix).zfill(4)}_part_00.parquet")
    #     for ix in range(80)
    # ]
    # orderitems = pd.concat(
    #     (map_orderitems_df(pd.read_parquet(ifile), user_id_map) for ifile in files),
    #     axis=0,
    #     ignore_index=True,
    #     copy=False,
    # )
    # # orderitems.reset_index(inplace=True, drop=True)
    # # intern_user_id(orderitems, user_id_map)

    # files = [
    #     os.path.join(data_path, f"Products_{str(ix).zfill(4)}_part_00.parquet")
    #     for ix in range(80)
    # ]
    # products = pd.concat(
    #     (pd.read_parquet(ifile) for ifile in files),
    #     axis=0,
    #     ignore_index=True,
    #     copy=False,
    # )
    # # products.reset_index(inplace=True, drop=True)

    return clicks, orderitems, products


def stem_column(token_map, df, row_id, column):
    stems = (
        df.select(pl.col(row_id), pl.col(column).alias("token"))
        .explode("token")
        .join(token_map, "token")
        .group_by(row_id)
        .agg(
            pl.col("stem").alias(column),
        )
    )
    return df.drop(column).join(
        stems,
        row_id,
    )


def simple_tokenize_column(col):
    return col.str.extract_all(
        r"[!?.,\"'/\\\(\)\{\}\[\]*+-_=&^%$#@~<>;:|]|(\w|\d|['])+"
    )


def load_products_pl(data_path):
    products = pl.scan_parquet(
        [
            os.path.join(data_path, f"Products_{str(ix).zfill(4)}_part_00.parquet")
            for ix in range(80)
        ]
    ).lazy()

    products = products.with_columns(
        pl.col("product_type").cast(pl.Categorical),
        pl.col("subject_score").cast(pl.Float32),
        pl.col("product_design_image_url_en_us").str.slice(
            len("https://rlv.zcache.com/")
        ),
        pl.col("product_image_url_en_us").str.slice(len("https://rlv.zcache.com/")),
        pl.col("product_url_en_us").str.slice(len("https://www.zazzle.com/")),
        simple_tokenize_column(pl.col("title")).alias("title_stems"),
        simple_tokenize_column(pl.col("long_description")).alias(
            "long_description_stems"
        ),
        pl.col("vision_embedding2")
        .str.extract_all(r"[+-]?([0-9]*[.])?[0-9]+")
        .cast(pl.List(pl.Float32))
        .alias("vision_embedding2"),
    )

    # products = products.sort("product_id")
    products = products.with_row_count(name="product_number")

    clicks = pl.scan_parquet(
        [
            os.path.join(data_path, f"Clicks_{str(ix).zfill(4)}_part_00.parquet")
            for ix in range(80)
        ]
    ).lazy()

    clicks = clicks.with_columns(
        pl.col("user_id").cast(pl.Categorical),
        pl.col("cleaned_url"),
    )

    clicks = clicks.join(
        products.select(["product_id", "product_number"]), "product_id"
    )
    clicks = clicks.drop("product_id")

    clicks = clicks.sort(
        [
            "user_id",
            "cleaned_url",
            "date_created",
            "product_number",
        ]
    )

    clicks = clicks.with_columns(
        (
            (pl.col("user_id") != pl.col("user_id").shift(-1))
            | (pl.col("cleaned_url") != pl.col("cleaned_url").shift(-1))
            | (pl.col("date_created").diff(-1) > (5 * 60 * 1e3))
        )
        .cum_sum()
        .alias("query_id")
    )

    queries = clicks.group_by("query_id").agg(
        pl.col("user_id").first().alias("user_id"),
        pl.col("cleaned_url").first().alias("cleaned_url"),
        pl.col("date_created").min().alias("time"),
        pl.col("product_number")
        .filter(pl.col("is_click") == 0)
        .list.sort()
        .alias("pass_ids"),
        pl.col("product_number")
        .filter(pl.col("is_click") != 0)
        .list.sort()
        .alias("click_ids"),
    )

    queries = queries.with_columns(
        pl.col("cleaned_url")
        .str.extract_groups(r"https://([^/]+)/(:?(\w+)/)(.*)")
        .alias("cleaned_url"),
    )

    queries = queries.with_columns(
        pl.col("cleaned_url").struct["3"].cast(pl.Categorical).alias("category"),
        simple_tokenize_column(
            pl.col("cleaned_url").struct["4"].str.replace_all(r"\+", " ")
        ).alias("query"),
    ).drop("cleaned_url")

    # token_map = queries.select(pl.col("query").explode().unique().alias("token"))

    # def add_to_token_map(token_map, df, col):
    #     return pl.concat(
    #         [
    #             token_map,
    #             df.select(pl.col(col).explode().unique().alias("token")).join(
    #                 token_map, "token", "anti"
    #             ),
    #         ]
    #     )

    # token_map = add_to_token_map(token_map, products, "title_stems")
    # token_map = add_to_token_map(token_map, products, "long_description_stems")

    token_map = (
        pl.concat(
            [
                queries.select(pl.col("query").explode().alias("token")),
                products.select(pl.col("title_stems").explode().alias("token")),
                products.select(
                    pl.col("long_description_stems").explode().alias("token")
                ),
            ]
        ).unique("token")
        .collect()
        .with_columns(pl.col("token").map_elements(stem_token).alias("stem"))
        .lazy()
    )

    # def add_to_token_map(token_map, df, col):
    #     return pl.concat(
    #         [
    #             token_map,
    #             df.select(pl.col(col).explode().unique().alias("token")).join(
    #                 token_map, "token", "anti"
    #             ),
    #         ]
    #     )

    # token_map = add_to_token_map(token_map, products, "title_stems")
    # token_map = add_to_token_map(token_map, products, "long_description_stems")

    # token_map = token_map.lazy().with_columns(
    #     pl.col("token").map_elements(stem_token).alias("stem")
    # )

    queries = stem_column(token_map, queries, "query_id", "query")
    products = stem_column(token_map, products, "product_number", "title_stems")
    products = stem_column(
        token_map, products, "product_number", "long_description_stems"
    )

    products = products.sort("product_number")
    queries = queries.sort(["time", "user_id"])

    # return products.collect(), queries.collect()
    return tuple(pl.collect_all([products, queries, token_map]))
