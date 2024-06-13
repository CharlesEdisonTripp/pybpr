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
    port: str = "19000"

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
    sample_by: Optional[Iterable[str]] = None,
):
    client.execute(f"DROP TABLE IF EXISTS {table_name}")

    create_table_columns_clause = ", ".join(
        [f"{name} {ch_type}" for name, ch_type in columns]
    )
    primary_key_clause = ", ".join(primary_keys)
    query = f"""
        CREATE TABLE {table_name} 
        (
           {create_table_columns_clause}
        )
        ENGINE = MergeTree()
        ORDER BY ({primary_key_clause})
        """
    if sample_by is not None:
        key_clause = ", ".join(sample_by)
        query = f"""{query}
        SAMPLE BY ({key_clause})"""

    client.execute(query)


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
        "click_input",
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
            ("product_id", "UInt64"),
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
            ("product_id", "UInt64"),
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


def get_distinct_queries():
    return """
    SELECT
        cleaned_url,
        query_count,
        components[3] AS category,
        decodeURLFormComponent(components[4]) AS query_string
    FROM (
        SELECT
            cleaned_url,
            COUNT(1) AS query_count,
            (extractAllGroups(cleaned_url, 'https:\/\/([^\/]+)\/(([^\/]+)\/)?(.*)'))[1] components
        FROM click_input
        GROUP BY cleaned_url
        )
    """


def make_stem_command(input: str) -> str:
    tokenize_regexp = r"([^\p{L}\d]+)"
    return f"arrayFilter(x -> isNotNull(x) and notEmpty(x), arrayMap(x -> stem('en', x), splitByRegexp('{tokenize_regexp}', lowerUTF8({input}))))"


def get_stems(source_table: str, source_column: str):
    return f"""
    (
        SELECT
            arrayJoin({make_stem_command(source_column)}) as stem,
            COUNT(1) {source_column}_count
        FROM
            {source_table}
        GROUP BY stem
    )
    """


def make_stem_table(
    db: DatabaseInfo,
):
    client = db.make_client()
    client.execute("SET allow_experimental_nlp_functions = 1;")

    stem_columns = [
        ("stem_id", "UInt32"),
        ("stem", "String"),
        ("query_count", "UInt64"),
        ("title_count", "UInt64"),
        ("long_description_count", "UInt64"),
        ("total_count", "UInt64"),
    ]

    create_table(client, "stem", stem_columns, ["stem_id"], ["stem_id"])

    q = f"""
        INSERT INTO stem
        SELECT 
            rowNumberInAllBlocks() stem_id,
            stem,
            coalesce(query_count, 0) query_count,
            coalesce(title_count, 0) title_count,
            coalesce(long_description_count, 0) long_description_count,
            (query_count + title_count + long_description_count) total_count
        FROM
            {get_stems('product', 'title')} AS titles
            FULL OUTER JOIN 
            {get_stems('product', 'long_description')} AS descriptions
            USING (stem)
            FULL OUTER JOIN
            (
            SELECT
                    arrayJoin({make_stem_command('query_string')}) as stem,
                    SUM(query_count) query_count
                FROM
                    ({get_distinct_queries()}) as queries
                GROUP BY stem
            ) AS queries
            USING (stem)
        ORDER BY total_count DESC
    """
    print(q)
    client.execute(q)


def get_stem_ids_from_column(table: str, id_column: str, target_column: str) -> str:
    return f"""
    SELECT
        {id_column},
        stem_id
    FROM
    (
        SELECT 
            {id_column},
            arrayJoin({make_stem_command(target_column)}) stem
        FROM
            {table}
    ) x
    INNER JOIN stem USING (stem)
    """


def make_product_title_stem_table(
    db: DatabaseInfo,
):
    client = db.make_client()
    client.execute("SET allow_experimental_nlp_functions = 1;")

    columns = [
        ("product_id", "UInt64"),
        ("feature_id", "UInt32"),
    ]

    create_table(client, "product_title_stem", columns, ["product_id", "feature_id"])

    client.execute(
        f"""
            INSERT INTO product_title_stem
            SELECT
                product_id,
                stem_id AS feature_id
            FROM
                (
                SELECT 
                    product_id,
                    arrayJoin({make_stem_command('title')}) as stem
                FROM
                    product
                ) product_data
                INNER JOIN 
                stem
                USING (stem)
                """
    )


def make_query_table(
    db: DatabaseInfo,
):
    client = db.make_client()

    query_columns = [
        ("query_id", "UInt32"),
        ("cleaned_url", "String"),
        ("query_count", "UInt64"),
        ("category", "LowCardinality(String)"),
        ("query_string", "String"),
        # ("tokens", "Array(String)"),
        # ("query_stems", "Array(UInt32)"),
    ]

    create_table(
        client,
        "query",
        query_columns,
        ["query_id"],
        sample_by=["query_id"],
    )
    # INSERT INTO query
    client.execute(
        f"""
        INSERT INTO query
            SELECT 
                rowNumberInAllBlocks() AS query_id, 
                *
            FROM (
                {get_distinct_queries()}
                ORDER BY query_count DESC
                ) AS src
    """
    )


def make_query_stem_table(
    db: DatabaseInfo,
):
    client = db.make_client()
    client.execute("SET allow_experimental_nlp_functions = 1;")

    columns = [
        ("query_id", "UInt32"),
        ("feature_id", "UInt32"),
    ]

    create_table(
        client,
        "query_stem",
        columns,
        ["query_id", "feature_id"],
    )

    q = f"""
    INSERT INTO query_stem
        {get_stem_ids_from_column('query', 'query_id', 'query_string')}
    """
    print(q)
    client.execute(q)


def make_user_table(
    db: DatabaseInfo,
):
    client = db.make_client()

    columns = [
        ("user_num", "UInt64"),
        ("user_id", "String"),
        ("num_interactions", "UInt64"),
    ]

    create_table(
        client,
        "user",
        columns,
        [
            "user_num",
        ],
        sample_by=["user_num"],
    )
    # INSERT INTO query
    client.execute(
        f"""
        INSERT INTO user
        SELECT
            rowNumberInAllBlocks() AS user_num, 
            user_id,
            COUNT(1) num_interactions
        FROM
            click_input
        GROUP BY user_id
    """
    )


def make_click_table(
    db: DatabaseInfo,
):
    """
    click replaces pass ->
        + only get passes
        + set is_click when a click happened after this view and before another view of the same product by the same user

    """

    client = db.make_client()

    columns = [
        ("click_num", "UInt64"),
        ("user_num", "UInt32"),
        ("date_created", "UInt32"),
        ("clicked", "Boolean"),
        ("query_id", "UInt32"),
        ("product_id", "UInt64"),
    ]

    create_table(
        client,
        "click",
        columns,
        ["click_num"],
    )

    # (CAST(date_created, 'UInt64') * 1000000 + (row_number() OVER (PARTITION BY user_num, date_created))) date_created,
    client.execute(
        f"""
        INSERT INTO click
        SELECT
            rowNumberInAllBlocks() AS click_num,
            user_num,
            date_created,
            clicked,
            query_id,
            product_id     
        FROM
        (
            SELECT 
                user_num,
                date_created,
                query_id,
                product_id,
                is_click,
                (NOT is_click AND (leadInFrame(is_click, 1) OVER w)) clicked
            FROM
                click_input
                INNER JOIN query USING (cleaned_url)
                INNER JOIN user USING (user_id)
            WINDOW
                w AS (PARTITION BY (user_num, query_id, product_id) ORDER BY date_created Rows BETWEEN CURRENT ROW AND 1 FOLLOWING)
        ) src
        WHERE
            NOT is_click
        ORDER BY date_created, user_num, query_id, product_id
    """
    )


def make_user_feature_count_table(
    db: DatabaseInfo,
):
    client = db.make_client()
    columns = [
        ("user_num", "UInt64"),
        ("feature_id", "UInt32"),
        ("date_created", "UInt32"),
        ("cumulative_clicks", "UInt32"),
        ("cumulative_passes", "UInt32"),
        # ("cumulative_query_clicks", "UInt32"),
        # ("cumulative_query_passes", "UInt32"),
    ]

    create_table(
        client,
        "user_feature_count",
        columns,
        ["user_num", "feature_id", "date_created"],
    )

    client.execute(
        f"""
            INSERT INTO user_feature_count
            SELECT
                src.user_num,
                product_title_stem.feature_id,
                src.date_created,
                (SUM(clicked) OVER w) AS cumulative_clicks,
                (SUM(NOT clicked) OVER w) AS cumulative_passes
            FROM
                (
                    SELECT 
                        user_num,
                        date_created,
                        clicked,
                        product_id
                    FROM 
                        click
                    ORDER BY user_num, date_created
                ) src
                INNER JOIN product_title_stem USING (product_id)
            WINDOW
                w AS (
                    PARTITION BY (user_num, product_title_stem.feature_id) 
                    ORDER BY date_created
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                    )
            """
    )
    # LEFT JOIN query_stem USING(query_id, feature_id)
    # (SUM(is_click AND (query_stem.query_id IS NOT NULL)) OVER w) AS cumulative_query_clicks,
    # (SUM((NOT is_click) AND (query_stem.query_id IS NOT NULL)) OVER w) - cumulative_query_clicks AS cumulative_query_passes


def make_global_feature_count_table(
    db: DatabaseInfo,
):
    client = db.make_client()
    columns = [
        ("feature_id", "UInt32"),
        ("date_created", "UInt32"),
        ("cumulative_clicks", "UInt32"),
        ("cumulative_passes", "UInt32"),
    ]

    create_table(
        client,
        "global_feature_count",
        columns,
        ["feature_id", "date_created"],
    )

    client.execute(
        f"""
            INSERT INTO global_feature_count
            SELECT
                product_title_stem.feature_id,
                src.date_created,
                (SUM(clicked) OVER w) AS cumulative_clicks,
                (SUM(NOT clicked) OVER w) AS cumulative_passes
            FROM
                (
                    SELECT 
                        date_created,
                        clicked,
                        product_id
                    FROM 
                        click
                    ORDER BY date_created
                ) src
                INNER JOIN product_title_stem USING (product_id)
            WINDOW
                w AS (
                    PARTITION BY (product_title_stem.feature_id) 
                    ORDER BY date_created
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                    )
            """
    )
    # LEFT JOIN query_stem USING(query_id, feature_id)
    # (SUM(is_click AND (query_stem.query_id IS NOT NULL)) OVER w) AS cumulative_query_clicks,
    # (SUM((NOT is_click) AND (query_stem.query_id IS NOT NULL)) OVER w) - cumulative_query_clicks AS cumulative_query_passes


def make_global_product_count_table(
    db: DatabaseInfo,
):
    client = db.make_client()
    columns = [
        ("product_id", "UInt64"),
        ("date_created", "UInt32"),
        ("cumulative_clicks", "UInt32"),
        ("cumulative_passes", "UInt32"),
    ]

    create_table(
        client,
        "global_product_count",
        columns,
        ["product_id", "date_created"],
    )

    client.execute(
        f"""
            INSERT INTO global_product_count
            SELECT
                product_id,
                date_created,
                (SUM(clicked) OVER w) AS cumulative_clicks,
                (SUM(NOT clicked) OVER w) AS cumulative_passes
            FROM
                click
            WINDOW
                w AS (
                    PARTITION BY (product_id) 
                    ORDER BY date_created
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                    )
            """
    )


def make_user_product_count_table(
    db: DatabaseInfo,
):
    client = db.make_client()
    columns = [
        ("user_num", "UInt64"),
        ("product_id", "UInt64"),
        ("date_created", "UInt32"),
        ("cumulative_clicks", "UInt32"),
        ("cumulative_passes", "UInt32"),
    ]

    create_table(
        client,
        "user_product_count",
        columns,
        ["user_num", "product_id", "date_created"],
    )

    client.execute(
        f"""
            INSERT INTO user_product_count
            SELECT
                user_num,
                product_id,
                date_created,
                (SUM(clicked) OVER w) AS cumulative_clicks,
                (SUM(NOT clicked) OVER w) AS cumulative_passes
            FROM
                click
            WINDOW
                w AS (
                    PARTITION BY (user_num, product_id) 
                    ORDER BY date_created
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                    )
            """
    )


def make_click_counts_table(
    db: DatabaseInfo,
):
    client = db.make_client()
    columns = [
        ("click_num", "UInt64"),
        ("user_num", "UInt32"),
        ("date_created", "UInt32"),
        ("clicked", "Boolean"),
        ("query_id", "UInt32"),
        ("product_id", "UInt64"),
        ("product_clicks", "UInt32"),
        ("product_passes", "UInt32"),
        ("product_user_clicks", "UInt32"),
        ("product_user_passes", "UInt32"),
        ("feature_id", "Array(UInt32)"),
        ("feature_clicks", "Array(UInt32)"),
        ("feature_passes", "Array(UInt32)"),
    ]

    create_table(
        client,
        "click_counts",
        columns,
        ["date_created"],
    )

    client.execute(
        f"""
            INSERT INTO click_counts
            SELECT
                product_counts.click_num,
                product_counts.user_num,
                product_counts.date_created,
                product_counts.clicked,
                product_counts.query_id,
                product_counts.product_id,
                product_counts.product_clicks,
                product_counts.product_passes,
                product_counts.product_user_clicks,
                product_counts.product_user_passes,
                feature_counts.feature_id,
                feature_counts.feature_clicks,
                feature_counts.feature_passes
            FROM
                (
                    SELECT
                        click_num,
                        user_num,
                        date_created,
                        clicked,
                        query_id,
                        product_id,
                        (SUM(clicked) OVER product_window) AS product_clicks,
                        (SUM(NOT clicked) OVER product_window) AS product_passes,
                        (SUM(clicked) OVER product_user_window) AS product_user_clicks,
                        (SUM(NOT clicked) OVER product_user_window) AS product_user_passes
                    FROM
                        click
                    WINDOW
                        product_window AS (
                            PARTITION BY (click.product_id) 
                            ORDER BY click_num
                            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                            ),
                        product_user_window AS (
                            PARTITION BY (click.product_id, click.user_num) 
                            ORDER BY click_num
                            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                            )
                    ORDER BY click_num
                ) product_counts
                INNER JOIN
                (
                    SELECT
                        click_num,
                        groupArray(feature_id) feature_id,
                        groupArray(feature_clicks) feature_clicks,
                        groupArray(feature_passes) feature_passes
                    FROM
                        (
                            SELECT
                                click_num,
                                product_title_stem.feature_id,
                                (SUM(clicked) OVER feature_window) feature_clicks,
                                (SUM(NOT clicked) OVER feature_window) feature_passes
                            FROM
                                click
                                INNER JOIN product_title_stem USING (product_id)
                            WINDOW
                                feature_window AS (
                                    PARTITION BY (product_title_stem.feature_id) 
                                    ORDER BY click_num
                                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                                    )
                            ORDER BY click_num
                        ) src
                    GROUP BY click_num
                ) feature_counts USING (click_num)
                SETTINGS join_algorithm = 'full_sorting_merge'
            """
    )
