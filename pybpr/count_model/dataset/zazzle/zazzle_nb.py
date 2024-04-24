from itertools import chain
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
    Iterable,
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

from clickhouse_driver import Client
import pyarrow.parquet
import glob
import dataclasses
import joblib


@dataclass
class DatabaseInfo:

    host: str = "localhost"
    database: str = "zazzle"
    port: str = "1900"

    def make_client(self) -> Client:
        return Client(**dataclasses.asdict(self))


def create_database(db: DatabaseInfo):
    kwargs = dataclasses.asdict(db)
    del kwargs["database"]
    client = Client(**kwargs)
    client.execute(f"DROP DATABASE IF EXISTS {db.database} SYNC")
    client.execute(f"CREATE DATABASE {db.database}")


def insert_file(file_path, db, table_name, column_names):
    print(dataclasses.asdict(db))
    client = db.make_client()
    columns_sql = ", ".join(column_names)
    source_parquet = pyarrow.parquet.ParquetFile(file_path)
    source = source_parquet.iter_batches(
        batch_size=128 * 1024,
        columns=column_names,
    )
    for batch in source:
        values = tuple(zip(*(batch[col].to_pylist() for col in column_names)))
        print(values[:1])
        client.execute(
            f"""
                INSERT INTO {table_name}
                (
                   {columns_sql}
                ) VALUES
                """,
            values,
        )


def create_table(
    client: Client,
    table_name: str,
    columns: List[Tuple[str, str]],
    primary_keys: Iterable[str],
):
    client.execute(f"DROP TABLE IF EXISTS {table_name}")

    create_table_columns_clause = ", ".join(
        [f"{name} {ch_type}" for name, ch_type in columns]
    )
    primary_key_clause = ", ".join(primary_keys)
    print(
        f"""
        CREATE TABLE {table_name} 
        (
           {create_table_columns_clause}
        )
        ENGINE = MergeTree()
        PRIMARY KEY ({primary_key_clause})
        """
    )
    client.execute(
        f"""
        CREATE TABLE {table_name} 
        (
           {create_table_columns_clause}
        )
        ENGINE = MergeTree()
        PRIMARY KEY ({primary_key_clause})
        """
    )


def load_file(
    file_path: str,
    db: DatabaseInfo,
    table_name: str,
    columns: List[Tuple[str, str]],
    primary_keys: Iterable[str],
):
    client = db.make_client()

    column_names = [name for name, ch_type in columns]

    create_table(client, table_name, columns, primary_keys)

    joblib.Parallel(n_jobs=16)(
        joblib.delayed(insert_file)(file_path, db, table_name, column_names)
        for file_path in glob.glob(file_path)
    )
    # for file_path in source_files:
    #     source_parquet = pyarrow.parquet.ParquetFile(file_path)
    #     source = source_parquet.iter_batches(
    #         batch_size=1024 * 1024,
    #         columns=column_names,
    #     )
    #     for batch in source:
    #         values = zip(*(batch[col].to_pylist() for col in column_names))
    #         print(values[:1])
    #         client.execute(
    #             f"""
    #             INSERT INTO {table_name}
    #             (
    #                {columns_sql}
    #             ) VALUES
    #             """,
    #             values,
    #         )


def load_clicks(
    data_path: str,
    db: DatabaseInfo,
):
    load_file(
        os.path.join(data_path, "Clicks_*.parquet"),
        db,
        "click",
        [
            ("date_created", "UInt64"),
            ("user_id", "String"),
            ("cleaned_url", "String"),
            ("product_id", "UInt64"),
            ("is_click", "Boolean"),
        ],
        ["user_id", "date_created"],
    )


def load_orderitems(
    data_path: str,
    db: DatabaseInfo,
):
    load_file(
        os.path.join(data_path, "OrderItems_*.parquet"),
        db,
        "orderitem",
        [
            ("date_created", "UInt64"),
            ("user_id", "String"),
            ("product_id", "Int64"),
        ],
        ["user_id", "date_created"],
    )


def load_products(
    data_path: str,
    db: DatabaseInfo,
):
    load_file(
        os.path.join(data_path, "Products_*.parquet"),
        db,
        "product",
        [
            ("product_id", "Int64"),
            ("title", "String"),
            ("product_type", "LowCardinality(String)"),
            ("product_url_en_us", "String"),
            ("long_description", "String"),
            ("final_department_id", "Int64"),
            ("price_us_usd", "Int32"),
            ("date_created", "UInt32"),
            ("date_modified", "UInt32"),
            ("is_customizable", "Boolean"),
            ("seller_id", "Int64"),
            ("store_id", "Int64"),
            ("group_id", "Int64"),
            ("vision_style_id_1", "Int64"),
            ("vision_style_score_1", "Float32"),
            ("vision_embedding2", "String"),
        ],
        ["product_id"],
    )


def load_files_into_database(
    data_path: str,
    db: DatabaseInfo,
):
    load_orderitems(data_path, db)
    load_products(data_path, db)
    load_clicks(data_path, db)


def create_query_table(
    db: DatabaseInfo,
):
    client = db.make_client()

    query_columns = [
        ("query_id", "UInt64"),
        ("cleaned_url", "String"),
        ("query_count", "UInt64"),
        ("category", "LowCardinality(String)"),
        ("tokens", "Array(String)"),
        # ("query_stems", "Array(UInt32)"),
    ]

    create_table(client, "query", query_columns, ["query_id"])
    # INSERT INTO query
    client.execute(
        """
        INSERT INTO query
            SELECT 
                rowNumberInAllBlocks() AS query_id, 
                cleaned_url,
                query_count,
                components[3] AS category,
                splitByWhitespace(decodeURLFormComponent(components[4])) AS query_tokens
            FROM (
                SELECT 
                    cleaned_url, 
                    COUNT(1) AS query_count,
                    (extractAllGroups(cleaned_url, 'https:\/\/([^\/]+)\/(([^\/]+)\/)?(.*)'))[1] components
                FROM click
                GROUP BY cleaned_url
                ORDER BY query_count DESC
                ) AS src    
    """
    )


@dataclass
class TableInfo:
    name: str
    output_path: str
    row_index: Optional[str]
    set_sorted: Any
    df: Optional[Union[pl.LazyFrame, pl.DataFrame]] = None
    estimated_size: Optional[int] = None

    @property
    def path(self) -> str:
        return os.path.join(self.output_path, self.name + ".pq")


async def load_source_data(data_path, regenerate=False):
    output_path = os.path.join(data_path, "output")
    os.makedirs(output_path, exist_ok=True)

    tables = [
        TableInfo(
            "products",
            output_path,
            "product_num",
            "product_id",
        ),
        TableInfo(
            "queries",
            output_path,
            "query_num",
            "cleaned_url",
        ),
        # TableInfo(
        #     "interactions",
        #     output_path,
        #     "interaction_num",
        #     "user_num",
        # ),
        TableInfo(
            "token_map",
            output_path,
            None,
            "token",
        ),
        TableInfo(
            "feature_map",
            output_path,
            "feature_num",
            None,
        ),
        TableInfo(
            "orderitems",
            output_path,
            None,
            "date_created",
        ),
        TableInfo(
            "users",
            output_path,
            "user_num",
            "user_id",
        ),
        TableInfo(
            "clicks",
            output_path,
            None,
            "user_num",
        ),
        TableInfo(
            "passes",
            output_path,
            None,
            "user_num",
        ),
    ]

    if regenerate or any((not os.path.isfile(table.path) for table in tables)):
        await etl_source_data(data_path, tables)

    # tables = tables[:-1]
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


def make_url_utf8_mappings():

    def mapping_generator():
        yield ("+", " ")

        def get_url_utf8_mapping(value: int, num_bytes: int):
            bytes_value = value.to_bytes(num_bytes, byteorder="big")
            utf8_value = bytes_value.decode("utf8")
            string_value = "%" + "%".join(
                (bytes_value[i : i + 1].hex() for i in range(len(bytes_value)))
            )
            return string_value, utf8_value

        for v in range(0, 128):
            yield get_url_utf8_mapping(v, 1)

        def fill_following_bytes(base, depth, num_bytes):
            if depth >= num_bytes:
                try:
                    yield get_url_utf8_mapping(base, num_bytes)
                except:
                    pass
            else:
                depth += 1
                base *= 256
                for value in range(base + 128, base + 192):
                    yield from fill_following_bytes(value, depth, num_bytes)

        for value in range(192, 224):
            yield from fill_following_bytes(value, 1, 2)

        for value in range(224, 240):
            yield from fill_following_bytes(value, 1, 3)

        for value in range(224, 248):
            yield from fill_following_bytes(value, 1, 4)

    return list(mapping_generator())


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

    #  (f"Begin queries collection 1...")
    queries = (
        clicks.select(pl.col("cleaned_url"))
        .group_by("cleaned_url")
        .len(token_counts[2])
        .sort("cleaned_url")
        .with_columns(
            pl.col("cleaned_url")
            .str.extract_groups(r"https:\/\/([^\/]+)\/(([^\/]+)\/)?(.*)")
            .alias("query")
        )
    )

    hex_map = make_url_utf8_mappings()

    queries = (
        queries.with_columns(
            pl.col("query").struct["3"].cast(pl.Categorical).alias("category"),
            # query_url_groups.struct["4"].str.replace_all(r"\+", " ").alias("query"),
            pl.col("query")
            .struct["4"]
            .str.replace_many(
                [k for k, v in hex_map],
                [v for k, v in hex_map],
                ascii_case_insensitive=True,
            )
            .alias("query"),
        )
        # .sort("query_count", descending=True)
        # .sort("cleaned_url")
        # .with_row_index("query_num").set_sorted("query_num")
        # .drop("cleaned_url")
    )
    del hex_map

    queries = save_table(table_map["queries"], queries, streaming=True)
    # print(queries.collect(streaming=True).sample(10))
    # print(get_tokens(queries, "query", token_counts[2], True).collect().sample(10))
    # print(
    #     get_tokens(queries, "query", token_counts[2], True)
    #     .collect(streaming=True)
    #     .sample(10)
    # )
    print(f"done.")

    print(f"Begin token_map collection 1...")

    token_map = get_tokens(
        queries,
        "query",
        token_counts[2],
        True,
    )

    token_map = join_tokens(
        token_map,
        products,
        "title",
        token_counts[0],
    )

    token_map = join_tokens(
        token_map,
        products,
        "long_description",
        token_counts[1],
    )

    # token_map = join_tokens(
    #     join_tokens(
    #         get_tokens(
    #             products,
    #             "title",
    #             token_counts[0],
    #         ),
    #         products,
    #         "long_description",
    #         token_counts[1],
    #     ),
    #     queries,
    #     "query",
    #     token_counts[2],
    #     True,
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
        .sum()
        # .agg([col.sum() for col in token_count_cols])
        # .sort(token_count_cols[2], descending=True)
    )
    # feature_map = feature_map.with_row_index(name="feature_num").set_sorted(
    #     "feature_num"
    # )

    print(f"Begin feature_map collection...")
    feature_map = save_table(table_map["feature_map"], feature_map).collect().lazy()
    print(f"done.")

    print(f"Begin token_map collection 2...")
    token_map = token_map.join(feature_map.select("stem", "feature_num"), "stem").drop(
        "stem"
    )

    # token_map, feature_map = pl.collect_all([token_map, feature_map])
    # token_dict = {k: v for k, v in zip(token_map["token"], token_map["feature_num"])}
    # token_mapping = (token_map["token"].to_numpy(), token_map["feature_num"].to_numpy())
    # token_map = token_map.lazy()
    # feature_map = feature_map.lazy()

    # token_map = save_table(table_map["token_map"], token_map).collect().lazy()
    token_map = save_table(table_map["token_map"], token_map).collect().lazy()
    print(f"done.")

    print(f"Begin query stem mapping...")
    # queries = queries.with_row_index("query_num").set_sorted("query_num")
    queries = (
        queries.join(
            stem_column(
                # token_map["token"],
                # token_map["feature_num"],
                token_map,
                queries,
                "query_num",
                "query",
                "query_stems",
            ),
            "query_num",
        )
        .drop("query")
        .rename({"query_stems": "query"})
    )

    queries = save_table(table_map["queries"], queries, streaming=True)
    print(f"done.")

    print(f"Begin product sort and preprocess...")
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

    products = products.sort("product_id")

    products = save_table(table_map["products"], products, streaming=True, sink=False)
    print(f"done.")

    # products_table = table_map["products"]
    print(f"Begin product stem mapping...")

    title_stems = stem_column(
        # token_map["token"],
        # token_map["feature_num"],
        token_map,
        products,
        "product_num",
        "title",
        "title_stems",
    )
    # # title_stems = save_df(
    # #     title_stems,
    # #     "title_stems",
    # #     products_table.output_path,
    # #     products_table.row_index,
    # #     products_table.set_sorted,
    # #     True,
    # #     False,
    # # )

    long_description_stems = stem_column(
        # token_map["token"],
        # token_map["feature_num"],
        token_map,
        products,
        "product_num",
        "long_description",
        "long_description_stems",
    )

    # # long_description_stems = save_df(
    # #     long_description_stems,
    # #     "long_description_stems",
    # #     products_table.output_path,
    # #     products_table.row_index,
    # #     products_table.set_sorted,
    # #     True,
    # #     False,
    # # )

    products = (
        products.drop("long_description")
        .join(title_stems, "product_num")
        .join(long_description_stems, "product_num")
    )

    # products = products.with_row_index(name="product_num").set_sorted("product_num")

    products = save_table(table_map["products"], products, streaming=True, sink=False)
    print(f"done.")

    print("User collection...")
    # users = (
    #     orderitems.select("user_id")
    #     .unique()
    #     .sort("user_id")
    #     .merge_sorted(clicks.select("user_id").unique().sort("user_id"), "user_id")
    #     .unique()
    # )

    users = (
        pl.concat(
            [orderitems.select("user_id").unique(), clicks.select("user_id").unique()]
        )
        .unique()
        .sort("user_id")
    )

    # users = users.with_row_index(name="user_num")
    users = save_table(table_map["users"], users, streaming=True)
    print("done.")

    print("Orderitems preparation...")
    orderitems = (
        orderitems.join(users.select("user_id", "user_num"), "user_id")
        .drop("user_id")
        .join(products.select("product_id", "product_num"), "product_id")
        .drop("product_id")
    )

    orderitems = orderitems.sort("date_created")
    orderitems = save_table(table_map["orderitems"], orderitems, streaming=True)
    print("done.")

    click_groups = clicks.select(
        "user_id", "cleaned_url", pl.col("date_created") // (30 * 1e6)
    )
    click_groups = map_column_with_table(click_groups, table_map["queries"])
    click_groups = map_column_with_table(click_groups, table_map["users"])
    click_groups = click_groups.unique(
        [
            "user_num",
            "query_num",
            "date_created",
        ]
    ).sort(
        "user_num",
        "query_num",
        "date_created",
    )

    click_groups = save_df(
        click_groups,
        "click_groups",
        table_map["clicks"].output_path,
        "click_group",
        None,
        sink=True,
    )

    interactions = map_column_with_table(clicks, table_map["queries"])
    interactions = map_column_with_table(interactions, table_map["users"])
    interactions = map_column_with_table(interactions, table_map["products"])
    interactions = click_groups.join(
        interactions,
        ["user_num", "query_num"],
    )  # .group_by("click_group").

    # clicks = map_column_with_table(clicks, table_map["products"])

    # print(f"Sink reduced clicks...")

    # events = clicks

    # clicks = events.filter(pl.col("is_click"))
    # clicks = map_column_with_table(clicks, table_map["queries"])
    # clicks = map_column_with_table(clicks, table_map["users"])
    # clicks = map_column_with_table(clicks, table_map["products"])
    # # clicks = save_table(table_map["clicks"], clicks, sink=True)
    # # print(f"done.")

    # # print(f"Sort clicks...")
    # # clicks = clicks.sort(
    # #     "user_num",
    # #     "query_num",
    # #     "date_created",
    # #     # "product_num",
    # # )
    # # clicks_table_info = table_map["clicks"]
    # # clicks = save_df(
    # #     clicks,
    # #     "reduced_clicks",
    # #     clicks_table_info.output_path,
    # #     None,
    # #     None,
    # #     sink=True,
    # # )
    # # print(f"done.")

    # # print(f"Sink sorted clicks...")
    # clicks = clicks.sort(
    #     "user_num",
    #     "query_num",
    #     "date_created",
    #     # "product_num",
    # )
    # clicks = save_table(table_map["clicks"], clicks, sink=True)
    # print(f"done.")

    # print(f"Passes...")

    # passes = events.filter(~pl.col("is_click"))
    # passes = map_column_with_table(passes, table_map["queries"])
    # passes = map_column_with_table(passes, table_map["users"])
    # passes = map_column_with_table(passes, table_map["products"])
    # # passes = passes.join(
    # #     clicks.collect().lazy(), ["user_num", "query_num", "date_created"], "anti"
    # # )
    # passes = passes.sort(
    #     "user_num",
    #     "query_num",
    #     "date_created",
    #     # "product_num",
    # )

    # passes = save_table(table_map["passes"], passes, sink=True)
    # print(f"done.")

    # print(f"Assign interaction numbers to clicks...")
    # clicks = clicks.with_columns(
    #     (
    #         (
    #             pl.col("date_created").diff(-1, null_behavior="ignore").fill_null(0)
    #             > (5 * 60 * 1e6)
    #         )
    #         | (pl.col("user_num").diff(-1, null_behavior="ignore").fill_null(0) != 0)
    #         | (pl.col("query_num").diff(-1, null_behavior="ignore").fill_null(0) != 0)
    #     ).cum_sum()
    #     # .over(["user_num", "query_num"])
    #     .alias("interaction_num")
    # )

    # clicks = save_table(table_map["clicks"], clicks).set_sorted(
    #     "user_num", "interaction_num"
    # )
    # print(f"done.")

    # clicks = clicks.sort(
    #     "user_num",
    #     "query_num",
    #     "interaction_num",
    #     "product_num",
    # )

    # interactions = (
    #     clicks
    #     .sort("date_created")
    #     # .with_columns(
    #     #     pl.col('date_created').explode().sort().over('user_num', 'query_num')
    #     # )
    #     .group_by(
    #         "user_num",
    #         "query_num",
    #         # maintain_order=True,
    #     )
    #     .all()
    #     # .agg(
    #     #     pl.col("date_created"),
    #     #     pl.col("product_num"),
    #     #     pl.col("is_click"),
    #     # )
    #     .with_columns(
    #         (
    #             pl.col("date_created")
    #             .list.diff(-1)
    #             .list.eval((pl.element() > (5 * 60 * 1e6)).cum_sum())
    #             .alias("interaction_num")
    #         ),
    #     )
    #     .explode(
    #         "date_created",
    #         "product_num",
    #         "is_click",
    #         "interaction_num",
    #     )
    #     # .set_sorted("user_num")
    #     .group_by(
    #         "user_num",
    #         "query_num",
    #         "interaction_num",
    #         # maintain_order=True,
    #     )
    #     .agg(
    #         pl.col("date_created").first(),
    #         pl.col("product_num")
    #         .filter(~pl.col("is_click"))
    #         .drop_nulls()
    #         .unique()
    #         .alias("passes"),
    #         pl.col("product_num")
    #         .filter(pl.col("is_click"))
    #         .drop_nulls()
    #         .unique()
    #         .alias("clicks"),
    #     )
    #     .drop("product_num")
    #     .with_columns(
    #         pl.col("passes").list.set_difference(pl.col("clicks")),
    #     )
    #     .with_columns(
    #         pl.col("passes").list.concat(pl.col("clicks")).alias("product_num"),
    #         pl.col("clicks")
    #         .list.eval(
    #             pl.element().is_not_null()
    #         )  # hack to get around list evaluation literal bug in polars
    #         .list.concat(
    #             pl.col("passes").list.eval(pl.element().is_null())
    #         )  # hack to get around list evaluation literal bug in polars
    #         .alias("click"),
    #     )
    #     .drop("clicks", "passes")
    #     .sort("user_num", "date_created")
    # )

    # interactions = (
    #     clicks.sort("date_created")
    #     .group_by(
    #         "user_num",
    #         "query_num",
    #         # maintain_order=True,
    #     )
    #     .all()
    #     # .agg(
    #     #     pl.col("date_created"),
    #     #     pl.col("product_num"),
    #     #     pl.col("is_click"),
    #     # )
    #     .with_columns(
    #         (
    #             pl.col("date_created")
    #             .list.diff(-1)
    #             .list.eval((pl.element() > (5 * 60 * 1e6)).cum_sum())
    #             .alias("interaction_num")
    #         ),
    #     )
    #     .explode(
    #         "date_created",
    #         "product_num",
    #         "is_click",
    #         "interaction_num",
    #     )
    #     # .set_sorted("user_num")
    #     .group_by(
    #         "user_num",
    #         "query_num",
    #         "interaction_num",
    #         # maintain_order=True,
    #     )
    #     .agg(
    #         pl.col("date_created").first(),
    #         pl.col("product_num")
    #         .filter(~pl.col("is_click"))
    #         .drop_nulls()
    #         .unique()
    #         .alias("passes"),
    #         pl.col("product_num")
    #         .filter(pl.col("is_click"))
    #         .drop_nulls()
    #         .unique()
    #         .alias("clicks"),
    #     )
    #     .drop("product_num")
    #     .with_columns(
    #         pl.col("passes").list.set_difference(pl.col("clicks")),
    #     )
    #     .with_columns(
    #         pl.col("passes").list.concat(pl.col("clicks")).alias("product_num"),
    #         pl.col("clicks")
    #         .list.eval(
    #             pl.element().is_not_null()
    #         )  # hack to get around list evaluation literal bug in polars
    #         .list.concat(
    #             pl.col("passes").list.eval(pl.element().is_null())
    #         )  # hack to get around list evaluation literal bug in polars
    #         .alias("click"),
    #     )
    #     .drop("clicks", "passes")
    #     .sort("user_num", "date_created")
    # )

    # interactions = save_table(
    #     table_map["interactions"], interactions, streaming=True, sink=False
    # )

    print(f"done.")


def map_column_with_table(
    target: pl.LazyFrame,
    table_info: TableInfo,
    drop=True,
):
    # table_info = table_map[table_name]
    df = target.join(
        table_info.df.select(table_info.row_index, table_info.set_sorted)  # type: ignore
        .collect()  # type: ignore
        .set_sorted(table_info.row_index, table_info.set_sorted)  # type: ignore
        .lazy(),
        table_info.set_sorted,
    )
    if drop:
        df = df.drop(table_info.set_sorted)
    return df


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
        df.select(token)
        .explode("token")
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
    # token_map_df: pl.DataFrame,
    df: pl.LazyFrame,
    row_id,
    source_column,
    stems_column,
):
    result = (
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
        .set_sorted(row_id)
        # .with_columns(
        #     pl.col(stems_column).replace(source, dest, return_dtype=pl.UInt32)
        # )
        .group_by(
            row_id,
            # maintain_order=True,
        )
        .all()
        # .agg(pl.col(stems_column))
        # .set_sorted(row_id)  # NB: could possibly not be true??
    )

    # order_check = (
    #     result.select(row_id).select((pl.col(row_id).diff(1) < 0).any()).collect()
    # )
    # print(f"---************--- order check: {order_check}")

    # return df.set_sorted(row_id).join(
    #     stems,
    #     row_id,
    # )
    return result


def simple_tokenize_column(col):
    return col.str.extract_all(
        r"[!?.,\"'\/\\\(\)\{\}\[\]*+-_=&^%$#@~<>;:|]|([^\W_]|['])+"
    )


def make_map_file_frame(source_table: TableInfo):
    map_path = os.path.join(source_table.output_path, f"{source_table.name}_map.ipc")
    print(f"make_map_file_frame {source_table.name} {map_path}")
    source_table.df.select(source_table.set_sorted).collect().write_ipc(  # type: ignore
        map_path,
        compression="uncompressed",
        # maintain_order=True,
    )
    return pl.scan_ipc(
        map_path,
        memory_map=True,
        row_index_name=source_table.row_index,
    )


def save_df(
    df: pl.LazyFrame | pl.DataFrame,
    name: str,
    dir: str,
    row_index: Optional[str],
    set_sorted: Optional[str],
    streaming=False,
    sink=False,
    mode="pq",
) -> pl.LazyFrame:
    streaming = streaming or sink

    print(f"************* Writing {name} with streaming={streaming} sink={sink}...")

    df = df.lazy()

    path = os.path.join(dir, f"{name}.{mode}")

    if mode == "ipc":
        if row_index in df.columns:
            df = df.drop(row_index)

        print(df.explain(streaming=streaming))
        if sink:
            df.sink_ipc(
                path,
                compression="lz4",
                # maintain_order=table.set_sorted is not None or table.row_index is not None,
                maintain_order=True,
            )
        else:
            # df.collect(streaming=streaming).write_parquet(
            #     table.path,
            #     compression="lz4",
            #     compression_level=9,
            #     pyarrow_options={},
            # )
            df.collect(streaming=streaming).write_ipc(
                path,
                # compression="uncompressed",
                compression="lz4",
                # compression_level=9,
                # pyarrow_options={},
            )
    elif mode == "pq":
        if row_index is not None and row_index not in df.columns:
            df = df.with_row_index(row_index)

        print(df.explain(streaming=streaming))
        if sink:
            df.sink_parquet(
                path,
                compression="lz4",
                # maintain_order=table.set_sorted is not None or table.row_index is not None,
                maintain_order=True,
            )
        else:
            # df.collect(streaming=streaming).write_parquet(
            #     table.path,
            #     compression="lz4",
            #     compression_level=9,
            #     pyarrow_options={},
            # )
            df.collect(streaming=streaming).write_parquet(
                path,
                # compression="uncompressed",
                compression="lz4",
                # compression_level=9,
                # pyarrow_options={},
            )
    else:
        raise ValueError("Unknown mode.")

    return load_df(
        name,
        dir,
        row_index,
        set_sorted,
        mode,
    )


def load_df(
    name: str,
    dir: str,
    row_index: Optional[str],
    set_sorted: Optional[str],
    mode: str = "pq",
) -> pl.LazyFrame:
    path = os.path.join(dir, f"{name}.{mode}")

    # df = pl.scan_parquet(table.path, low_memory=True, rechunk=False).lazy()
    if mode == "ipc":
        df = pl.scan_ipc(
            path,
            rechunk=False,
            cache=False,
            memory_map=False,
            row_index_name=row_index,
        )
    elif mode == "pq":
        df = pl.scan_parquet(
            path,
            rechunk=False,
            cache=False,
            # row_index_name=row_index,
        )

    sorted_cols = [
        e for e in (row_index, set_sorted) if e is not None and e in df.columns
    ]
    if len(sorted_cols) > 0:
        df = df.set_sorted(*sorted_cols)

    return df


def save_table(
    table: TableInfo,
    df: pl.LazyFrame | pl.DataFrame,
    streaming=False,
    sink=False,
) -> pl.LazyFrame:
    table.df = save_df(
        df,
        table.name,
        table.output_path,
        table.row_index,
        table.set_sorted,
        streaming,
        sink,
        # table.mode,
    )
    return table.df


def load_table(table: TableInfo) -> pl.LazyFrame:
    table.df = load_df(
        table.name,
        table.output_path,
        table.row_index,
        table.set_sorted,
        # table.mode,
    )
    return table.df
