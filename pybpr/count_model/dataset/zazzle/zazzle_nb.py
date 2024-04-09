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
    estimated_size: Optional[int] = None

    @property
    def path(self) -> str:
        return os.path.join(self.output_path, self.name)


async def load_source_data(data_path, regenerate=False):
    output_path = os.path.join(data_path, "output")
    os.makedirs(output_path, exist_ok=True)

    tables = [
        TableInfo("products", "product_number", output_path),
        TableInfo("queries", "query_id", output_path),
        TableInfo("interactions", "date_created", output_path),
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
    products = pl.scan_parquet(
        os.path.join(data_path, f"Products_*.parquet"),
        low_memory=True,
        rechunk=False,
    ).lazy()
    orderitems = pl.scan_parquet(
        os.path.join(data_path, f"OrderItems_*.parquet"),
        low_memory=True,
        rechunk=False,
    ).lazy()
    clicks = pl.scan_parquet(
        os.path.join(data_path, f"Clicks_*.parquet"),
        low_memory=True,
        rechunk=False,
    ).lazy()

    token_counts = [f"{col}_count" for col in ["title", "long_description", "query"]]
    token_count_cols = [pl.col(col) for col in token_counts]

    print(f"Begin queries collection 1...")
    queries = (
        clicks.select(pl.col("cleaned_url"))
        .group_by("cleaned_url")
        .len(name=token_counts[2])
        # .agg()
        # .first()
        # .unique()
        # .sort("cleaned_url")
        # .with_row_index("query_id")
        # .collect(streaming=True)
        # .lazy()
    )
    # print(f"done.")

    # print(f"Begin queries collection 2...")
    # queries = queries.with_row_index("query_id").collect().lazy()
    # print(f"done.")

    # print(f"Begin queries tokenization collection...")
    query_url_groups = pl.col("cleaned_url").str.extract_groups(
        r"https://([^/]+)/(:?(\w+)/)(.*)"
    )
    queries = (
        queries.with_columns(
            query_url_groups.struct["3"].cast(pl.Categorical).alias("category"),
            # query_url_groups.struct["4"].str.replace_all(r"\+", " ").alias("query"),
            query_url_groups.struct["4"].alias("query"),
            # simple_tokenize_column(
            #     query_url_groups.struct["4"].str.replace_all(r"\+", " ")
            # ).alias("query"),
        )
        .with_row_index("query_id")
        # .drop("cleaned_url")
        .collect()
        .lazy()
    )
    print(f"done.")

    print(f"Begin token_map collection 1...")

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
                .group_by("token")
                .agg(pl.col(count_col).sum().alias(count_col))
            )
        return df.select(token).explode("token").group_by("token").len(name=count_col)

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

    # token_map = (
    #     (
    #         pl.concat(
    #             [
    #                 products.select(pl.col("title_stems").explode().alias("token")),
    #                 products.select(
    #                     pl.col("long_description_stems").explode().alias("token")
    #                 ),
    #                 queries.select(pl.col("query").explode().alias("token")),
    #             ],
    #             # rechunk=False,
    #         )
    #         .group_by("token")
    #         .len()
    #         # .sort("token")
    #     )
    #     .collect()
    #     .with_columns(
    #         pl.col("token").map_elements(stem_token, return_dtype=pl.Utf8).alias("stem")
    #     )
    #     .lazy()
    # )

    token_map = token_map.collect()
    token_map = token_map.with_columns(
        pl.col("token").map_elements(stem_token, return_dtype=pl.Utf8).alias("stem")
    ).lazy()
    print(f"done.")

    feature_map = (
        token_map.select(*([pl.col("stem")] + token_count_cols))
        .group_by("stem")
        .agg([col.sum() for col in token_count_cols])
        .sort(token_count_cols[2], descending=True)
    )
    feature_map = feature_map.with_row_index(name="feature_id").set_sorted("feature_id")

    # print(f"Begin feature_map collection...")
    # feature_map = feature_map.collect().lazy()
    # print(f"done.")

    token_map = token_map.join(feature_map.select("stem", "feature_id"), "stem")

    print(f"Begin token_map, feature_map collection...")
    token_map, feature_map = pl.collect_all([token_map, feature_map])
    # token_dict = {k: v for k, v in zip(token_map["token"], token_map["feature_id"])}

    # token_map = token_map.lazy()
    feature_map = feature_map.lazy()
    print(f"done.")

    print(f"Begin query stem mapping...")
    queries = (
        stem_column(
            token_map["token"],
            token_map["feature_id"],
            queries,
            "query_id",
            "query",
            "query",
        )
        .collect(streaming=True)
        .lazy()
    )
    print(f"done.")

    print(f"Begin product stem mapping...")

    products = products.with_columns(
        pl.col("product_type").cast(pl.Categorical),
        pl.col("subject_score").cast(pl.Float32),
        pl.col("product_design_image_url_en_us").str.slice(
            len("https://rlv.zcache.com/")
        ),
        pl.col("product_image_url_en_us").str.slice(len("https://rlv.zcache.com/")),
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
        # token_dict,
        token_map["token"],
        token_map["feature_id"],
        products,
        "product_id",
        "title",
        "title_stems",
    )

    products = (
        stem_column(
            # token_dict,
            token_map["token"],
            token_map["feature_id"],
            products,
            "product_id",
            "long_description",
            "long_description_stems",
        )
        # .collect(streaming=True)
        # .lazy()
    ).drop("long_description")

    token_map = token_map.lazy()

    # products = products.sort("product_id")
    products = products.with_row_index(name="product_number").set_sorted(
        "product_number"
    )

    print(f"Begin product collection 1...")
    products = products.collect()
    print(f"products size: {products.estimated_size() / 1024**2}")
    products = products.lazy()
    print(f"done.")

    users = (
        pl.concat(
            [
                orderitems.select("user_id"),
                clicks.select("user_id"),
            ],
            rechunk=False,
        )
        .group_by("user_id")
        .agg()
        # .unique()
        .sort("user_id")
    )

    print(f"Begin users collection 1...")
    users = users.collect(streaming=True).lazy()
    print(f"done.")
    print(f"Begin users collection 2...")
    users = (
        users.with_row_index(name="user_number")
        .set_sorted("user_number")
        .collect()
        .lazy()
    )
    print(f"done.")
    # products, users = (df.lazy() for df in pl.collect_all([products, users]))

    # clicks = clicks.with_columns(pl.col("cleaned_url").replace())
    clicks = clicks.join(queries, "cleaned_url").drop("cleaned_url")

    # replace user id (string) and product id (int64) with user number (uint32) and product number (uint32)
    orderitems, clicks = tuple(
        (
            replace_id_with_number(
                replace_id_with_number(
                    df,
                    "user_id",
                    "user_number",
                    users,
                ),
                "product_id",
                "product_number",
                products,
            )
            for df in [orderitems, clicks]
        )
    )

    orderitems = orderitems.sort("date_created")
    print(f"Begin orderitems collection 1...")
    orderitems = orderitems.collect(streaming=True).lazy()
    print(f"done.")

    clicks = clicks.sort(
        [
            "user_number",
            "query_id",
            "date_created",
            # "product_number",
        ]
    )

    print(f"Begin clicks collection...")
    print(clicks.explain(streaming=True))
    clicks = clicks.collect(streaming=True)
    print(f"clicks: {clicks.estimated_size() / 1024**2}")
    clicks = clicks.lazy()
    print(f"done.")

    # interactions = clicks.rolling(
    #     index_column="date_created",
    #     period="5m",
    #     offset="0s",
    #     closed="right",
    #     group_by=[
    #         "query_id",
    #         "user_number",
    #     ],
    #     # check_sorted=True,
    # ).agg(
    #     # pl.col("query_id").first(),
    #     # pl.col("user_number").first(),
    #     # pl.col("date_created").first().alias("time"),
    #     pl.col("product_number").filter(pl.col("is_click") == 0).alias("pass_numbers"),
    #     pl.col("product_number").filter(pl.col("is_click") != 0).alias("click_numbers"),
    # )

    # print(f"Begin clicks collection...")
    # clicks = clicks.collect(streaming=True).lazy()
    # print(f"done.")

    # clicks = clicks.with_columns(
    #     (pl.col("date_created").diff(-1) > (5 * 60 * 1e3))
    #     .cum_sum()
    #     .over(["query_id", "user_number"])
    #     .alias("query_number")
    # )

    # clicks = clicks.group_by(
    #     [pl.col("user_number"), pl.col("query_id")]
    # )

    # print(f"Begin clicks collection 2...")
    # clicks = clicks.collect().lazy()
    # print(f"done.")

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
        pl.col("date_created").first(),
        pl.col("product_number").filter(pl.col("is_click") == 0).alias("pass_numbers"),
        pl.col("product_number").filter(pl.col("is_click") != 0).alias("click_numbers"),
    )

    interactions = interactions.drop("interaction_id")
    interactions = interactions.with_columns(
        pl.col("pass_numbers").list.sort(),
        pl.col("click_numbers").list.sort(),
    )

    interactions = interactions.sort("date_created")

    # queries = queries.collect(streaming=True).lazy()

    print(f"Begin interactions collection...")
    print(interactions.explain())
    interactions = interactions.collect()
    print(f"interactions: {interactions.estimated_size() / 1024**2}")
    interactions = interactions.lazy()
    # orderitems, queries = (
    #     df.lazy()
    #     for df in pl.collect_all(
    #         [orderitems, queries],
    #         # streaming=True,
    #     )
    # )
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


#  stem_column(
#             token_dict,
#             queries,
#             "query_id",
#             "query",
#             "query",
#         )


def stem_column(
    # token_dict: Dict[str, int],
    source,
    dest,
    df: pl.LazyFrame,
    row_id,
    source_column,
    stems_column,
):
    stems = (
        df.select(
            pl.col(row_id),
            simple_tokenize_column(pl.col(source_column)).alias(stems_column),
        )
        .explode(stems_column)
        .with_columns(
            pl.col(stems_column).replace(source, dest, return_dtype=pl.UInt32)
        )
        .group_by(row_id, maintain_order=True)
        .agg(pl.col(stems_column))
    )
    return df.join(
        stems,
        row_id,
    )
    # return (
    #     df.with_columns(
    #         simple_tokenize_column(pl.col(source_column)).alias(stems_column)
    #     )
    #     .explode(stems_column)
    #     .with_columns(
    #         pl.col(stems_column)
    #         .replace(token_dict, return_dtype=pl.UInt32)
    #     )
    #     .group_by(row_id, maintain_order=True)
    #     .agg(pl.col(stems_column))
    # )

    # token_map, products, "product_number", "title_stems", "feature_id"
    # stems = (
    #     df.select(
    #         pl.col(row_id),
    #         simple_tokenize_column(pl.col(source_column))
    #         .alias("token")
    #         .explode()
    #         .replace(token_dict, return_dtype=pl.UInt32)
    #     )
    #     # .join(token_map.select(pl.col("feature_id"), pl.col("token")), "token")
    #     .group_by(row_id, maintain_order=True)
    #     .agg(
    #         pl.col("feature_id").alias(stems_column),
    #         # pl.len().alias(column + "_count"),
    #     )
    # )
    # # return df.with_columns(stems[stems_column])
    # return df.join(
    #     stems,
    #     row_id,
    # )

    # stems = (
    #     df.select(
    #         pl.col(row_id),
    #         simple_tokenize_column(pl.col(source_column)).alias("token").explode(),
    #     )
    #     .join(token_map.select(pl.col("feature_id"), pl.col("token")), "token")
    #     .group_by(row_id)
    #     .agg(
    #         pl.col("feature_id").alias(stems_column),
    #         # pl.len().alias(column + "_count"),
    #     )
    # )
    # return df.drop(column).join(
    #     stems,
    #     row_id,
    # )


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
        print(f"save_tables {table.path}...")
        table.df = table.df.collect()  # type: ignore
        # table.estimated_size = table.df.estimated_size()
        # table.df = table.df.lazy()
        partitions = make_partitions(table)
        table.df = None
        collected = pl.collect_all([partition for path, partition in partitions])  # type: ignore

        for (path, partition), collected in zip(partitions, collected):
            print(f"save_tables partition {path}...")
            collected.write_parquet(
                path,
                compression="lz4",
                compression_level=9,
                pyarrow_options={},
            )
        del partitions
        del collected


def make_partitions(table: TableInfo):

    if table.sort_by is None:
        return [(table.path + f"_0.parquet", table.df.lazy())]  # type: ignore
    df = table.df  # type: ignore
    num_partitions = int(max(1, np.ceil(df.estimated_size() / (256 * 1024**2))))  # type: ignore
    table.df = table.df.lazy()  # type: ignore

    pcol = pl.col(table.sort_by)
    pmax = pcol.max()
    pmin = pcol.min()

    partition_key_range_size = (pmax - pmin + 1) // num_partitions  # type: ignore
    partition_number = (pcol - pmin) // partition_key_range_size
    partition_number = partition_number.alias("partition")

    df: pl.LazyFrame = table.df.with_columns(partition_number)  # type: ignore

    return [
        (
            table.path + f"_{i}.parquet",
            df.filter(pl.col("partition") == i).drop("partition"),
        )
        for i in range(num_partitions)
    ]


def load_range_partitions(table: TableInfo):
    return (
        pl.scan_parquet(
            table.path + f"_*.parquet",
            low_memory=True,
            rechunk=False,
        )
        .lazy()
        .sort(table.sort_by)
    )


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
