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
    partition: bool
    output_path: str
    df: Optional[Union[pl.LazyFrame, pl.DataFrame]] = None
    estimated_size: Optional[int] = None

    @property
    def path(self) -> str:
        return os.path.join(self.output_path, self.name + "_0.parquet")


async def load_source_data(data_path, regenerate=False):
    output_path = os.path.join(data_path, "output")
    os.makedirs(output_path, exist_ok=True)

    tables = [
        TableInfo("products", "product_num", True, output_path),
        TableInfo("queries", "query_num", True, output_path),
        TableInfo("interactions", "interaction_num", True, output_path),
        TableInfo("token_map", "token", False, output_path),
        TableInfo("feature_map", "feature_num", True, output_path),
        TableInfo("orderitems", "date_created", True, output_path),
        TableInfo("users", "user_num", True, output_path),
        TableInfo("clicks", None, False, output_path),
    ]

    if regenerate or any((not os.path.isfile(table.path) for table in tables)):
        await etl_source_data(data_path, tables)

    tables = tables[:-1]
    for table in tables:
        # load_range_partitions(table)
        load_table(table)

    return tables


def collect_tables(tables: List[TableInfo]):
    collected_dfs = pl.collect_all(
        [table.df for table in tables],  # type: ignore
        streaming=True,
    )  # type: ignore

    for table, collected_df in zip(tables, collected_dfs):
        table.df = collected_df


async def etl_source_data(data_path: str, tables: List[TableInfo]):
    table_map = {table.name: table for table in tables}

    products = pl.scan_parquet(
        os.path.join(data_path, f"Products_*.parquet"),
        low_memory=True,
        rechunk=False,
    )
    orderitems = pl.scan_parquet(
        os.path.join(data_path, f"OrderItems_*.parquet"),
        low_memory=True,
        rechunk=False,
    )
    clicks = pl.scan_parquet(
        os.path.join(data_path, f"Clicks_*.parquet"),
        low_memory=True,
        rechunk=False,
    )

    token_counts = [f"{col}_count" for col in ["title", "long_description", "query"]]
    token_count_cols = [pl.col(col) for col in token_counts]

    # print(f"Begin queries collection 1...")
    queries = (
        clicks.select(pl.col("cleaned_url"))
        .group_by("cleaned_url")
        .len(name=token_counts[2])
        # .agg()
        # .first()
        # .unique()
        .sort("cleaned_url")
        # .with_row_index("query_num")
        # .set_sorted("query_num")
        # .collect(streaming=True)
        # .lazy()
    )

    query_url_groups = pl.col("cleaned_url").str.extract_groups(
        r"https:\/\/([^\/]+)\/(([^\/]+)\/)?(.*)"
    )
    queries = (
        queries.with_columns(
            query_url_groups.struct["3"].cast(pl.Categorical).alias("category"),
            query_url_groups.struct["4"].str.replace_all(r"\+", " ").alias("query"),
        )
        # .sort("query_count", descending=True)
        # .sort("cleaned_url")
        .with_row_index("query_num").set_sorted("query_num")
        # .drop("cleaned_url")
    )
    queries = save_table(table_map["queries"], queries, streaming=True)
    print(f"done.")

    print(f"Begin token_map collection 1...")

    token_map = join_tokens(
        join_tokens(
            get_tokens(
                products,
                "title",
                token_counts[0],
            ),
            products,
            "long_description",
            token_counts[1],
        ),
        queries,
        "query",
        token_counts[2],
        True,
    )

    token_map = token_map.sort("token")
    print(token_map.explain(streaming=True))
    token_map = token_map.collect(streaming=True)
    token_map = (
        token_map.with_columns(
            pl.col("token").map_elements(stem_token, return_dtype=pl.Utf8).alias("stem")
        )
        .drop_nulls("stem")
        .lazy()
    )
    print(f"done.")

    feature_map = (
        token_map.select(*([pl.col("stem")] + token_count_cols))
        .group_by("stem")
        .agg([col.sum() for col in token_count_cols])
        .sort(token_count_cols[2], descending=True)
    )
    feature_map = feature_map.with_row_index(name="feature_num").set_sorted(
        "feature_num"
    )

    token_map = token_map.join(feature_map.select("stem", "feature_num"), "stem").drop(
        "stem"
    )

    print(f"Begin token_map, feature_map collection...")
    token_map, feature_map = pl.collect_all([token_map, feature_map])
    # token_dict = {k: v for k, v in zip(token_map["token"], token_map["feature_num"])}
    # token_mapping = (token_map["token"].to_numpy(), token_map["feature_num"].to_numpy())
    # token_map = token_map.lazy()
    # feature_map = feature_map.lazy()
    feature_map = save_table(table_map["feature_map"], feature_map)
    print(f"done.")

    print(f"Begin query stem mapping...")
    # queries = queries.with_row_index("query_num").set_sorted("query_num")
    queries = (
        stem_column(
            # token_map["token"],
            # token_map["feature_num"],
            token_map.lazy(),
            queries,
            "query_num",
            "query",
            "query_stems",
        )
        .drop("query")
        .rename({"query_stems": "query"})
    )
    queries = save_table(table_map["queries"], queries, streaming=True)
    print(f"done.")

    print(f"Begin product stem mapping...")

    products = products.drop(
        "product_design_image_url_en_us",
        "product_image_url_en_us",
        "subject_score",
        "subject",
    )

    products = products.with_columns(
        pl.col("product_type").cast(pl.Categorical),
        # pl.col("subject_score").cast(pl.Float32),
        # pl.col("product_design_image_url_en_us").str.slice(
        #     len("https://rlv.zcache.com/")
        # ),
        # pl.col("product_image_url_en_us").str.slice(len("https://rlv.zcache.com/")),
        pl.col("product_url_en_us").str.slice(len("https://www.zazzle.com/")),
        # simple_tokenize_column(pl.col("title")).alias("title_stems"),
        # simple_tokenize_column(pl.col("long_description")).alias(
        #     "long_description_stems"
        # ),
        pl.col("vision_embedding2")
        .str.extract_all(r"[+-]?([0-9]*[.])?[0-9]+")
        .cast(pl.List(pl.Float32))
        .alias("vision_embedding2"),
    )

    products = stem_column(
        # token_map["token"],
        # token_map["feature_num"],
        token_map.lazy(),
        products,
        "product_id",
        "title",
        "title_stems",
    )

    products = (
        stem_column(
            # token_map["token"],
            # token_map["feature_num"],
            token_map.lazy(),
            products,
            "product_id",
            "long_description",
            "long_description_stems",
        )
    ).drop("long_description")

    products = products.sort("product_id")
    products = products.with_row_index(name="product_num").set_sorted("product_num")

    products = save_table(table_map["products"], products, streaming=True)

    token_map = save_table(table_map["token_map"], token_map)

    users = (
        orderitems.select("user_id")
        .unique()
        .sort("user_id")
        .merge_sorted(clicks.select("user_id").unique().sort("user_id"), "user_id")
        .unique()
    )

    users = users.with_row_index(name="user_num")
    users = save_table(table_map["users"], users, streaming=True)

    orderitems = (
        orderitems.join(users.select("user_id", "user_num"), "user_id")
        .drop("user_id")
        .join(products.select("product_id", "product_num"), "product_id")
        .drop("product_id")
    )

    orderitems = orderitems.sort("date_created")
    orderitems = save_table(table_map["orderitems"], orderitems, streaming=True)

    clicks = (
        clicks.join(queries.select("cleaned_url", "query_num"), "cleaned_url")
        .drop("cleaned_url")
        .join(users.select("user_id", "user_num"), "user_id")
        .drop("user_id")
        .join(products.select("product_id", "product_num"), "product_id")
        .drop("product_id")
    )

    print(f"Sink reduced clicks...")
    print(clicks.explain(streaming=True))
    clicks.sink_parquet(
        table_map["clicks"].path,
        compression="lz4",
        compression_level=9,
    )
    print(f"done.")
    clicks = pl.scan_parquet(
        table_map["clicks"].path,
        low_memory=True,
        rechunk=False,
    )

    clicks = clicks.sort(
        "user_num",
        "query_num",
        "date_created",
        "product_num",
    )

    clicks = clicks.with_columns(
        (
            pl.col("date_created").diff(-1, null_behavior="ignore").fill_null(0)
            > (5 * 60 * 1e6)
        )
        .cum_sum()
        .over(["user_num", "query_num"])
        .alias("interaction_num")
    )

    # clicks = clicks.sort(
    #     "user_num",
    #     "query_num",
    #     "interaction_num",
    #     "product_num",
    # )

    interactions = (
        clicks.group_by(
            "user_num",
            "query_num",
            "interaction_num",
            # maintain_order=True,
        )
        .agg(
            # pl.col("query_num").first(),
            # pl.col("user_num").first(),
            pl.col("date_created").first(),
            pl.col("product_num")
            .filter(~pl.col("is_click"))
            .drop_nulls()
            .unique()
            .alias("pass_nums"),
            pl.col("product_num")
            .filter(pl.col("is_click"))
            .unique()
            .drop_nulls()
            .alias("click_nums"),
        )
        .with_columns(
            pl.col("pass_nums").list.set_difference(pl.col("click_nums")).list.sort(),
            pl.col("click_nums").list.sort(),
        )
    )
    interactions = interactions.drop("interaction_num")

    interactions = interactions.sort("date_created")
    interactions = interactions.with_row_index(name="interaction_num")

    interactions = save_table(table_map["interactions"], interactions, streaming=False)

    print(f"done.")


def get_tokens(
    df: pl.LazyFrame,
    col,
    count_col,
    sum_count=False,
) -> pl.LazyFrame:
    token = simple_tokenize_column(pl.col(col)).alias("token")
    if sum_count:
        return (
            df.select(
                token,
                pl.col(count_col),
            )
            .explode("token")
            .drop_nulls("token")
            .group_by("token")
            .sum()
            .sort("token")
            # .agg(pl.col(count_col).sum().alias(count_col))
        )
    # return df.select(token.explode()).group_by("token").len(name=count_col)
    return (
        df.select(token.explode())
        .drop_nulls("token")
        .group_by("token")
        .len(name=count_col)
        .sort("token")
    )


def join_tokens(
    left: pl.LazyFrame,
    df: pl.LazyFrame,
    col,
    count_col,
    sum_count=False,
) -> pl.LazyFrame:
    return left.join(
        get_tokens(df, col, count_col, sum_count),
        "token",
        how="outer_coalesce",
    ).with_columns(pl.col(count_col).fill_null(0))


_stemmer = PorterStemmer()


def stem_token(token):
    return _stemmer.stem(token)


def stem_column(
    # token_dict: Dict[str, int],
    # source,
    # dest,
    token_map: pl.LazyFrame,
    df: pl.LazyFrame,
    row_id,
    source_column,
    stems_column,
):
    stems = (
        df.select(
            pl.col(row_id),
            simple_tokenize_column(pl.col(source_column)),
        )
        .explode(source_column)
        .join(
            token_map.select(
                pl.col("token").alias(source_column),
                pl.col("feature_num").alias(stems_column),
            ),
            source_column,
        )
        .drop(source_column)
        # .with_columns(
        #     pl.col(stems_column).replace(source, dest, return_dtype=pl.UInt32)
        # )
        .group_by(row_id, maintain_order=True)
        .agg(pl.col(stems_column))
    )

    return df.join(
        stems,
        row_id,
    )


def simple_tokenize_column(col):
    return col.str.extract_all(
        r"[!?.,\"'\/\\\(\)\{\}\[\]*+-_=&^%$#@~<>;:|]|([^\W_]|['])+"
    )


def save_table(
    table: TableInfo,
    df: pl.LazyFrame | pl.DataFrame,
    streaming=False,
):
    print(f"save_tables {table.path}...")
    # df: pl.LazyFrame = table.df.lazy()  # type: ignore
    df = df.lazy()
    print(df.explain(streaming=streaming))
    df.collect(streaming=streaming).write_parquet(
        table.path,
        compression="lz4",
        compression_level=9,
        pyarrow_options={},
    )
    return load_table(table, sort=False)


def load_table(table: TableInfo, sort=False):
    df = pl.scan_parquet(table.path, low_memory=True, rechunk=False).lazy()
    if sort and table.sort_by is not None:
        df = df.sort(table.sort_by)
    table.df = df
    return df
