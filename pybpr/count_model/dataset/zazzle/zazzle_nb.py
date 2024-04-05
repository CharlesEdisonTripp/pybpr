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
from dataclasses import dataclass, field
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


from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


@dataclass
class TableInfo:
    name: str
    sort_by: Optional[str]
    output_path: str
    df: Optional[Union[pl.LazyFrame, pl.DataFrame]] = None

    @property
    def path(self) -> str:
        return os.path.join(self.output_path, self.name)


async def load_source_data(data_path, regenerate=False):
    output_path = os.path.join(data_path, "output")
    os.makedirs(output_path, exist_ok=True)

    tables = [
        TableInfo("products", "product_number", output_path),
        TableInfo("queries", "query_id", output_path),
        TableInfo("interactions", "time", output_path),
        TableInfo("token_map", None, output_path),
        TableInfo("feature_map", None, output_path),
        TableInfo("orderitems", "date_created", output_path),
        TableInfo("users", "user_number", output_path),
    ]

    if regenerate or any(
        (not os.path.isfile(table.path + "_0.parquet") for table in tables)
    ):
        await etl_source_data(data_path, tables)
        # products, queries, token_map, feature_map, orderitems, users = dfs
        # for (name, sort_by, path), df in zip(tables, dfs):
        #     df.write_parquet(
        #         path,
        #         compression="lz4",
        #         compression_level=9,
        #         pyarrow_options={},
        #     )
    # else:
    # products, queries, token_map, feature_map, orderitems, users = tuple(
    #     pl.collect_all(
    #         [
    #             pl.scan_parquet(path).lazy().sort(sort_by)
    #             for name, sort_by, path in tables
    #         ],
    #         streaming=True,
    #     )
    # )
    for table in tables:
        load_range_partitions(table)

    return tables


def collect_tables(tables: List[TableInfo]):
    collected_dfs = pl.collect_all(
        [table.df for table in tables],  # type: ignore
        streaming=True,
    )  # type: ignore

    for table, collected_df in zip(tables, collected_dfs):
        table.df = collected_df


async def etl_source_data(data_path: str, tables: List[TableInfo]):
    products = pl.scan_parquet(os.path.join(data_path, f"Products_*.parquet")).lazy()
    orderitems = pl.scan_parquet(
        os.path.join(data_path, f"OrderItems_*.parquet")
    ).lazy()
    clicks = pl.scan_parquet(os.path.join(data_path, f"Clicks_*.parquet")).lazy()

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
    products = products.with_row_index(name="product_number").set_sorted(
        "product_number"
    )
    print(f"Begin product collection 1...")
    products = products.collect(streaming=True).lazy()
    print(f"done.")

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
    users = users.with_row_index(name="user_number").set_sorted("user_number")
    print(f"Begin users collection 1...")
    users = users.collect(streaming=True).lazy()
    print(f"done.")
    # products, users = (df.lazy() for df in pl.collect_all([products, users]))

    # replace user id (string) and product id (int64) with user number (uint32) and product number (uint32)
    orderitems, clicks = tuple(
        (
            replace_id_with_number(
                replace_id_with_number(
                    df,
                    "product_id",
                    "product_number",
                    products,
                ),
                "user_id",
                "user_number",
                users,
            )
            for df in [orderitems, clicks]
        )
    )

    orderitems = orderitems.sort("date_created")
    print(f"Begin orderitems collection 1...")
    orderitems = orderitems.collect(streaming=True).lazy()
    print(f"done.")

    print(f"Begin queries collection...")
    queries = (
        clicks.select(pl.col("cleaned_url"))
        .unique()
        .sort("cleaned_url")
        .with_row_index("query_id")
        .collect()
        .lazy()
    )
    print(f"done.")

    clicks = clicks.join(queries, "cleaned_url").drop("cleaned_url")

    clicks = clicks.sort(
        [
            "query_id",
            "user_number",
            "date_created",
            # "product_number",
        ]
    )
    clicks = clicks.with_columns(
        (
            (pl.col("query_id") != pl.col("query_id").shift(-1))
            | (pl.col("user_number") != pl.col("user_number").shift(-1))
            | (pl.col("date_created").diff(-1) > (5 * 60 * 1e3))
        )
        .cum_sum()
        .alias("interaction_id")
    )

    interactions = clicks.group_by("interaction_id").agg(
        pl.col("query_id").first(),
        pl.col("user_number").first(),
        pl.col("date_created").first().alias("time"),
        pl.col("product_number").filter(pl.col("is_click") == 0).alias("pass_numbers"),
        pl.col("product_number").filter(pl.col("is_click") != 0).alias("click_numbers"),
    )

    interactions = interactions.with_columns(
        pl.col("pass_numbers").list.sort(),
        pl.col("click_numbers").list.sort(),
    )

    interactions = interactions.sort("time")

    # queries = queries.collect(streaming=True).lazy()

    print(f"Begin interactions collection...")
    interactions = interactions.collect(streaming=True).lazy()
    # orderitems, queries = (
    #     df.lazy()
    #     for df in pl.collect_all(
    #         [orderitems, queries],
    #         # streaming=True,
    #     )
    # )
    print(f"done.")

    print(f"Begin queries tokenization collection...")
    queries = (
        queries.with_columns(
            pl.col("cleaned_url")
            .str.extract_groups(r"https://([^/]+)/(:?(\w+)/)(.*)")
            .alias("cleaned_url"),
        )
        .with_columns(
            pl.col("cleaned_url").struct["3"].cast(pl.Categorical).alias("category"),
            simple_tokenize_column(
                pl.col("cleaned_url").struct["4"].str.replace_all(r"\+", " ")
            ).alias("query"),
        )
        .drop("cleaned_url")
        .collect()
        .lazy()
    )
    print(f"done.")

    print(f"Begin token_map collection 1...")
    token_map = (
        (
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
        )
        .collect(streaming=True)
        .with_columns(pl.col("token").map_elements(stem_token).alias("stem"))
        .lazy()
    )
    print(f"done.")

    feature_map = (
        token_map.select(pl.col("stem"), pl.col("len"))
        .group_by("stem")
        .agg(pl.col("len").sum().alias("count"))
        .sort("count", descending=True)
    )
    token_map = token_map.drop("len")
    feature_map = feature_map.with_row_index(name="feature_id").set_sorted("feature_id")

    # print(f"Begin feature_map collection...")
    # feature_map = feature_map.collect().lazy()
    # print(f"done.")

    token_map = token_map.join(feature_map.select("stem", "feature_id"), "stem")

    print(f"Begin token_map, feature_map collection...")
    token_map, feature_map = (
        df.lazy() for df in pl.collect_all([token_map, feature_map])
    )
    print(f"done.")

    print(f"Begin query stem mapping...")
    queries = (
        stem_column(token_map, queries, "query_id", "query", "feature_id")
        # .collect(streaming=True)
        # .lazy()
    )
    print(f"done.")

    print(f"Begin product stem mapping...")
    products = stem_column(
        token_map, products, "product_number", "title_stems", "feature_id"
    )
    products = (
        stem_column(
            token_map,
            products,
            "product_number",
            "long_description_stems",
            "feature_id",
        )
        # .collect(streaming=True)
        # .lazy()
    )
    print(f"done.")

    table_map = {table.name: table for table in tables}
    for name, df in [
        ("products", products),
        ("queries", queries),
        ("interactions", interactions),
        ("token_map", token_map),
        ("feature_map", feature_map),
        ("orderitems", orderitems),
        ("users", users),
    ]:
        table = table_map[name]
        table.df = df

    print(f"Begin save tables...")
    save_tables(tables)
    print(f"done.")


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


def save_tables(tables: List[TableInfo]):
    # partitions = []
    # for table in tables:
    #     partitions.extend(make_partitions(table))

    # print(f"Begin collect all partitions...")
    # collected = pl.collect_all([partition for path, partition in partitions])
    # for (path, partition), collected in zip(partitions, collected):
    #     # for path, partition in partitions:
    #     #     partition.sink_ipc(path)
    #     #     pass
    #     collected.write_parquet(
    #         path,
    #         compression="lz4",
    #         compression_level=9,
    #         pyarrow_options={},
    #     )
    for table in tables:
        table.df = table.df.collect().lazy()
        partitions = make_partitions(table)
        table.df = None

        for path, partition in pl.collect_all(partitions):
            partition.write_parquet(
                path,
                compression="lz4",
                compression_level=9,
                pyarrow_options={},
            )


def make_partitions(table: TableInfo):
    if table.sort_by is None:
        return [(table.name + f"_0.parquet", table.df)]

    num_partitions = 64

    pcol = pl.col(table.sort_by)
    pmax = pcol.max()
    pmin = pcol.min()

    partition_key_range_size = (pmax - pmin + 1) // num_partitions
    partition_number = (pcol - pmin) // partition_key_range_size
    partition_number = partition_number.alias("partition")

    df: pl.LazyFrame = table.df.with_columns(partition_number)  # type: ignore

    return [
        (
            table.name + f"_*.parquet",
            df.filter(pl.col("partition") == i).drop("partition"),
        )
        for i in range(num_partitions)
    ]


def load_range_partitions(table: TableInfo):
    return pl.scan_parquet(table.path + f"_*.parquet").lazy().sort(table.sort_by)


def replace_id_with_number(target_df, id_col, number_col, mapping_df):
    return target_df.join(
        mapping_df.select(
            [
                id_col,
                number_col,
            ]
        ),
        id_col,
    ).drop(id_col)
