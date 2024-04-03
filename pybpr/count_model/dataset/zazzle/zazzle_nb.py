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

import polars as pl


import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

_stemmer = PorterStemmer()


def stem_token(token):
    return _stemmer.stem(token)


def stem_column(token_map, df, row_id, column, stem_id_column):
    stems = (
        df.select(pl.col(row_id), pl.col(column).alias("token"))
        .explode("token")
        .join(token_map, "token")
        .group_by(row_id)
        .agg(
            pl.col(stem_id_column).alias(column),
        )
    )
    return df.drop(column).join(
        stems,
        row_id,
    )


def simple_tokenize_column(col):
    return col.str.extract_all(
        r"[!?.,\"'\/\\\(\)\{\}\[\]*+-_=&^%$#@~<>;:|]|([^\W_]|['])+"
    )


def load_source_data(data_path, regenerate=False):
    output_path = os.path.join(data_path, "output")
    os.makedirs(output_path, exist_ok=True)

    outputs = (
        "products",
        "queries",
        "token_map",
        "feature_map",
        "orderitems",
        "users",
    )
    output_paths = [os.path.join(output_path, name + ".pq") for name in outputs]

    if regenerate or any(
        (
            not os.path.isfile(os.path.join(output_path, output + ".pq"))
            for output in outputs
        )
    ):
        dfs = etl_source_data(data_path)
        for path, df in zip(output_paths, dfs):
            df.write_parquet(
                path,
                compression="lz4",
                compression_level=9,
                pyarrow_options={},
            )
    else:
        dfs = tuple((pl.scan_parquet(path) for path in output_paths))

    return dfs


def etl_source_data(data_path):
    products = pl.scan_parquet(os.path.join(data_path, f"Products_*.parquet")).lazy()
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
    products = products.sort("product_id")
    products = products.with_row_count(name="product_number")
    # products = products.collect_async(streaming=True)

    orderitems = pl.scan_parquet(
        os.path.join(data_path, f"OrderItems_*.parquet")
    ).lazy()

    clicks = pl.scan_parquet(os.path.join(data_path, f"Clicks_*.parquet")).lazy()

    users = (
        pl.concat(
            [
                orderitems.select("user_id").unique(),
                clicks.select("user_id").unique(),
            ],
            rechunk=True,
        )
        .unique()
        .sort("user_id")
    )
    users = users.with_row_count(name="user_number")
    # users = users.collect_async(streaming=True)

    # products = (await products).lazy()

    orderitems = orderitems.join(
        products.select(["product_id", "product_number"]), "product_id"
    )
    orderitems = orderitems.drop("product_id")

    clicks = clicks.join(
        products.select(["product_id", "product_number"]), "product_id"
    )
    clicks = clicks.drop("product_id")

    # users = (await users).lazy()
    orderitems = orderitems.join(users, "user_id").drop("user_id")
    orderitems = orderitems.sort("date_created")

    clicks = clicks.join(users, "user_id").drop("user_id")

    # products = products.with_columns(pl.col("product_id").alias("product_number")).drop(
    #     "product_id"
    # )
    # products = products.sort("product_id")
    # products = products.with_row_count(name="product_number")
    # products = products.set_sorted("product_number")

    # orderitems = orderitems.with_columns(
    #     pl.col("product_id").alias("product_number")
    # ).drop("product_id")

    # clicks = clicks.with_columns(pl.col("product_id").alias("product_number")).drop(
    #     "product_id"
    # )

    # products = products.collect(streaming=True).lazy()

    clicks = clicks.sort(
        [
            "user_number",
            "cleaned_url",
            "date_created",
            "product_number",
        ]
    )

    clicks = clicks.with_columns(
        (
            (pl.col("user_number") != pl.col("user_number").shift(-1))
            | (pl.col("cleaned_url") != pl.col("cleaned_url").shift(-1))
            | (pl.col("date_created").diff(-1) > (5 * 60 * 1e3))
        )
        .cum_sum()
        .alias("query_id")
    )

    queries = clicks.group_by("query_id").agg(
        pl.col("user_number").first(),
        pl.col("cleaned_url").first(),
        pl.col("date_created").min().alias("time"),
        pl.col("product_number").filter(pl.col("is_click") == 0).alias("pass_numbers"),
        pl.col("product_number").filter(pl.col("is_click") != 0).alias("click_numbers"),
    )
    del clicks

    queries = queries.with_columns(
        pl.col("cleaned_url")
        .str.extract_groups(r"https://([^/]+)/(:?(\w+)/)(.*)")
        .alias("cleaned_url"),
        pl.col("pass_numbers").list.sort(),
        pl.col("click_numbers").list.sort(),
    )

    queries = queries.with_columns(
        pl.col("cleaned_url").struct["3"].cast(pl.Categorical).alias("category"),
        simple_tokenize_column(
            pl.col("cleaned_url").struct["4"].str.replace_all(r"\+", " ")
        ).alias("query"),
    ).drop("cleaned_url")
    queries = queries.sort("time")

    orderitems, queries, products, users = tuple(
        (
            df.lazy()
            for df in pl.collect_all(
                [orderitems, queries, products, users],
                streaming=True,
            )
        )
    )

    token_map = (
        pl.concat(
            [
                products.select(pl.col("title_stems").explode().alias("token")),
                products.select(
                    pl.col("long_description_stems").explode().alias("token")
                ),
                queries.select(pl.col("query").explode().alias("token")),
            ],
            rechunk=False,
        )
        .group_by("token")
        .len()
        .sort("token")
    ).collect(streaming=True)

    # token_map = token_map.collect()
    token_map = token_map.with_columns(
        pl.col("token").map_elements(stem_token).alias("stem")
    )
    token_map = token_map.lazy()
    # products = products.lazy()
    # queries = queries.lazy()
    # orderitems = orderitems.lazy()
    # users = users.lazy()

    feature_map = (
        token_map.select(pl.col("stem"), pl.col("len"))
        .group_by("stem")
        .agg(pl.col("len").sum().alias("count"))
        .sort("count", descending=True)
    )
    feature_map = feature_map.with_row_count(name="feature_id")
    feature_map = feature_map.collect(streaming=True).lazy()
    # feature_map = feature_map.set_sorted("feature_id")

    token_map = token_map.drop("len")
    token_map = token_map.join(feature_map.select("stem", "feature_id"), "stem")

    queries = stem_column(token_map, queries, "query_id", "query", "feature_id")
    products = stem_column(
        token_map, products, "product_number", "title_stems", "feature_id"
    )
    products = stem_column(
        token_map, products, "product_number", "long_description_stems", "feature_id"
    )

    # return products.collect(), queries.collect()
    # feature_map = feature_map.set_sorted("feature_id")

    return tuple(
        pl.collect_all(
            [products, queries, token_map, feature_map, orderitems, users],
            streaming=True,
        )
    )
