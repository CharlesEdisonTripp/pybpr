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
    ]

    if regenerate or any((not os.path.isfile(table.path) for table in tables)):
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
    # print(f"done.")

    # print(f"Begin queries collection 2...")
    # queries = queries.with_row_index("query_num").collect().lazy()
    # print(f"done.")

    # print(f"Begin queries tokenization collection...")
    query_url_groups = pl.col("cleaned_url").str.extract_groups(
        r"https:\/\/([^\/]+)\/(([^\/]+)\/)?(.*)"
    )
    queries = (
        queries.with_columns(
            query_url_groups.struct["3"].cast(pl.Categorical).alias("category"),
            # query_url_groups.struct["4"].str.replace_all(r"\+", " ").alias("query"),
            query_url_groups.struct["4"].str.replace_all(r"\+", " ").alias("query"),
            # simple_tokenize_column(
            #     query_url_groups.struct["4"].str.replace_all(r"\+", " ")
            # ).alias("query"),
        )
        # .sort("query_count", descending=True)
        # .sort("cleaned_url")
        .with_row_index("query_num").set_sorted("query_num")
        # .drop("cleaned_url")
    )
    print(queries.explain(streaming=True))
    # queries = queries.sink_csv(table_map["queries"].path + "_0.parquet")
    # queries = queries.collect(streaming=True).lazy()
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

    # print(f"Begin feature_map collection...")
    # feature_map = feature_map.collect().lazy()
    # print(f"done.")

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
            token_map["token"],
            token_map["feature_num"],
            queries,
            "query_num",
            "query",
            "query",
        )
        .drop("query")
        .rename({"query_right": "query"})
    )
    queries = save_table(table_map["queries"], queries, streaming=True)
    # print(queries.explain(streaming=True))
    # queries = queries.collect(streaming=True)
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
        token_map["token"],
        token_map["feature_num"],
        products,
        "product_id",
        "title",
        "title_stems",
    )

    products = (
        stem_column(
            token_map["token"],
            token_map["feature_num"],
            products,
            "product_id",
            "long_description",
            "long_description_stems",
        )
        # .collect(streaming=True)
        # .lazy()
    ).drop("long_description")

    products = products.sort("product_id")
    products = products.with_row_index(name="product_num").set_sorted("product_num")

    products = save_table(table_map["products"], products)

    token_map = save_table(table_map["token_map"], token_map)
    # print(f"Begin product collection 1...")
    # print(products.explain())
    # # products = products.collect(streaming=True)
    # products = products.collect().lazy()
    # print(f"products size: {products.estimated_size() / 1024**2}")

    # products = products.lazy()
    # print(f"done.")

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

    # print(f"Begin users collection 1...")
    # users = users.collect(streaming=True).lazy()
    # print(f"done.")
    # print(f"Begin users collection 2...")
    users = users.with_row_index(name="user_num").set_sorted("user_num")
    users = save_table(table_map["users"], users, streaming=True)
    # print(f"done.")
    # products, users = (df.lazy() for df in pl.collect_all([products, users]))

    # clicks = clicks.with_columns(pl.col("cleaned_url").replace())
    # clicks = clicks.join(queries, "cleaned_url").drop("cleaned_url")

    # replace user id (string) and product id (int64) with user number (uint32) and product number (uint32)
    # user_id_to_num = pl.col("user_id").replace(
    #     users["user_id"].to_list(),
    #     users["user_num"].to_list(),
    #     return_dtype=pl.UInt32,
    # )
    # product_id_to_num = pl.col("product_id").replace(
    #     products["product_id"].to_list(),
    #     products["product_num"].to_list(),
    #     return_dtype=pl.UInt32,
    # )
    # user_product_map_rename = {
    #     "product_id": "product_num",
    #     "user_id": "user_num",
    # }

    # orderitems = orderitems.with_columns(user_id_to_num, product_id_to_num)
    # orderitems = orderitems.rename(user_product_map_rename)
    orderitems = (
        orderitems.join(users.select("user_id", "user_num"), "user_id")
        .drop("user_id")
        .join(products.select("product_id", "product_num"), "product_id")
        .drop("product_id")
    )

    orderitems = orderitems.sort("date_created")
    orderitems = save_table(table_map["orderitems"], orderitems, streaming=True)

    # print(f"Begin orderitems collection 1...")
    # orderitems = orderitems.collect(streaming=True).lazy()
    # print(f"done.")

    clicks = (
        clicks.join(users.select("user_id", "user_num"), "user_id")
        .drop("user_id")
        .join(products.select("product_id", "product_num"), "product_id")
        .drop("product_id")
        .join(queries.select("cleaned_url", "query_num"), "cleaned_url")
        .drop("cleaned_url")
    )
    # clicks = clicks.with_columns(
    #     user_id_to_num,
    #     product_id_to_num,
    #     pl.col("cleaned_url").replace(
    #         # queries["cleaned_url"],
    #         # queries["query_num"],
    #         queries.select("cleaned_url"),
    #         queries.select("query_num"),
    #         return_dtype=pl.UInt32,
    #     ),
    # )
    # clicks = clicks.rename(user_product_map_rename | {"cleaned_url": "query_num"})

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

    # clicks = clicks.with_columns(
    #     (pl.col("date_created").diff(-1) > (5 * 60 * 1e3))
    #     .cum_sum()
    #     .alias("interaction_num"),
    # )
    # clicks = clicks.with_columns(
    #     (
    #         (pl.col("query_num") != pl.col("query_num").shift(-1))
    #         | (pl.col("user_num") != pl.col("user_num").shift(-1))
    #         | (pl.col("date_created").diff(-1) > (5 * 60 * 1e3))
    #     )
    #     .cum_sum()
    #     .alias("interaction_num")
    # )

    clicks = clicks.sort(
        "user_num",
        "query_num",
        "interaction_num",
        "product_num",
    )

    interactions = (
        clicks.group_by(
            "user_num",
            "query_num",
            "interaction_num",
            maintain_order=True,
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
            pl.col("pass_nums").list.set_difference(pl.col("click_nums")),
        )
    )
    interactions = interactions.drop("interaction_num")
    # interactions = interactions.with_columns(
    #     pl.col("pass_nums").list.sort(),
    #     pl.col("click_nums").list.sort(),
    # )

    interactions = interactions.sort("date_created")
    interactions = interactions.with_row_index(name="interaction_num")

    # queries = queries.collect(streaming=True).lazy()

    interactions = save_table(table_map["interactions"], interactions, streaming=False)
    # print(f"Begin interactions collection...")
    # print(interactions.explain())
    # interactions = interactions.collect()
    # print(f"interactions: {interactions.estimated_size() / 1024**2}")
    # interactions = interactions.lazy()
    # orderitems, queries = (
    #     df.lazy()
    #     for df in pl.collect_all(
    #         [orderitems, queries],
    #         # streaming=True,
    #     )
    # )
    # print(f"done.")

    # for name, df in [
    #     ("products", products.lazy()),
    #     ("queries", queries.lazy()),
    #     ("interactions", interactions),
    #     ("token_map", token_map.lazy()),
    #     ("feature_map", feature_map),
    #     ("orderitems", orderitems),
    #     ("users", users.lazy()),
    # ]:
    #     table = table_map[name]
    #     table.df = df

    # print(f"Begin save tables...")
    # save_tables(tables)
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
            .agg(pl.col(count_col).sum().alias(count_col))
        )
    # return df.select(token.explode()).group_by("token").len(name=count_col)
    return (
        df.select(token.explode())
        .drop_nulls("token")
        .group_by("token")
        .len(name=count_col)
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


# queries = stem_column(
#     *token_mapping,
#     queries,
#     "query_num",
#     "query",
#     "query",
# )


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
            simple_tokenize_column(pl.col(source_column).alias(stems_column)),
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

    # return df.with_columns(
    #     simple_tokenize_column(pl.col(source_column))
    #     .alias(stems_column)
    #     .list.explode()
    #     .replace(source, dest, return_dtype=pl.UInt32)
    #     .implode()
    #     .over(row_id),
    # )
    # return df.with_columns(
    #     simple_tokenize_column(pl.col(source_column))
    #     .list.eval(
    #         pl.element().replace(source, dest, return_dtype=pl.UInt32),
    #         # parallel=True,
    #     )
    #     .alias(stems_column),
    # )

    # stems = (
    #     df.select(
    #         pl.col(row_id),
    #         pl.col(source_column).alias(stems_column),
    #     )
    #     .with_columns(
    #         simple_tokenize_column(pl.col(stems_column)),
    #     )
    #     .explode(stems_column)
    #     .with_columns(
    #         pl.col(stems_column).replace(source, dest, return_dtype=pl.UInt32)
    #     )
    #     .group_by(row_id, maintain_order=True)
    #     .agg(pl.col(stems_column))
    # )
    # return df.join(
    #     stems,
    #     row_id,
    # )
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

    # token_map, products, "product_num", "title_stems", "feature_num"
    # stems = (
    #     df.select(
    #         pl.col(row_id),
    #         simple_tokenize_column(pl.col(source_column))
    #         .alias("token")
    #         .explode()
    #         .replace(token_dict, return_dtype=pl.UInt32)
    #     )
    #     # .join(token_map.select(pl.col("feature_num"), pl.col("token")), "token")
    #     .group_by(row_id, maintain_order=True)
    #     .agg(
    #         pl.col("feature_num").alias(stems_column),
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
    #     .join(token_map.select(pl.col("feature_num"), pl.col("token")), "token")
    #     .group_by(row_id)
    #     .agg(
    #         pl.col("feature_num").alias(stems_column),
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


# def save_tables(tables: List[TableInfo]):
#     for table in tables:
#         save_table(table, table.df)  # type: ignore


# def save_table(table):
#     print(f"save_tables {table.path}...")
#     table.df = table.df.collect()  # type: ignore
#     # table.estimated_size = table.df.estimated_size()
#     # table.df = table.df.lazy()
#     partitions = make_partitions(table)
#     table.df = None
#     collected = pl.collect_all([partition for path, partition in partitions])  # type: ignore

#     for (path, partition), collected in zip(partitions, collected):
#         print(f"save_tables partition {path}...")
#         collected.write_parquet(
#             path,
#             compression="lz4",
#             compression_level=9,
#             pyarrow_options={},
#         )
#     del partitions
#     del collected


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
    return load_table(table)


def load_table(table: TableInfo):
    df = pl.scan_parquet(table.path, low_memory=True, rechunk=False).lazy()
    if table.sort_by is not None:
        df = df.sort(table.sort_by)
    table.df = df
    return df


# def make_partitions(table: TableInfo):

#     if not table.partition:  # type: ignore
#         return [(table.path + f"_0.parquet", table.df.lazy())]  # type: ignore
#     df = table.df  # type: ignore
#     num_partitions = int(max(1, np.ceil(df.estimated_size() / (256 * 1024**2))))  # type: ignore
#     table.df = table.df.lazy()  # type: ignore

#     pcol = pl.col(table.sort_by)  # type: ignore
#     pmax = pcol.max()
#     pmin = pcol.min()

#     partition_key_range_size = (pmax - pmin + 1) // num_partitions  # type: ignore
#     partition_num = (pcol - pmin) // partition_key_range_size
#     partition_num = partition_num.alias("partition")

#     df: pl.LazyFrame = table.df.with_columns(partition_num)  # type: ignore

#     return [
#         (
#             table.path + f"_{i}.parquet",
#             df.filter(pl.col("partition") == i).drop("partition"),
#         )
#         for i in range(num_partitions)
#     ]


# def load_range_partitions(table: TableInfo):
#     df = pl.scan_parquet(
#         table.path + f"_*.parquet",
#         low_memory=True,
#         rechunk=False,
#     ).lazy()
#     if table.sort_by is not None:
#         df = df.sort(table.sort_by)
#     table.df = df
#     return table.df


# def replace_id_with_num(target_df, id_col, number_col, mapping_df):
#     return target_df.join(
#         mapping_df.select(
#             [
#                 id_col,
#                 number_col,
#             ]
#         ),
#         id_col,
#     ).drop(id_col)
