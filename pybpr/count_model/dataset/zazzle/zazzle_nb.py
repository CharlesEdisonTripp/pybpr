from enum import IntEnum
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


def make_stem_table(
    db: DatabaseInfo,
):
    client = db.make_client()

    stem_columns = [
        ("stem_id", "UInt32"),
        ("stem", "String"),
        ("query_count", "UInt32"),
        ("title_count", "UInt32"),
        ("long_description_count", "UInt32"),
        ("total_count", "UInt32"),
    ]

    create_table(client, "stem", stem_columns, ["stem_id"], ["stem_id"])

    """
    FULL OUTER JOIN
            {get_stems('product', 'long_description')} AS descriptions
    """
    client.execute("SET allow_experimental_nlp_functions = 1;")
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
        ORDER BY total_count DESC;
    """
    print(q)
    client.execute(q)


def make_stem_command(input: str) -> str:
    tokenize_regexp = r"([^\p{L}\d]+)"
    return f"arrayFilter(x -> isNotNull(x) and notEmpty(x), arrayMap(x -> stem('en', x), splitByRegexp('{tokenize_regexp}', lowerUTF8({input}))))"


def get_stems(source_table: str, source_column: str):
    return f"""
    (
        SELECT
            'stem' feature_type,
            NULL int_value,
            arrayJoin({make_stem_command(source_column)}) string_value,
            0 query_count,
            1 product_count
        FROM
            {source_table}
    )
    """


def make_feature_table(
    db: DatabaseInfo,
):
    client = db.make_client()

    columns = [
        ("feature_id", "UInt32"),
        ("feature_type", "LowCardinality(String)"),
        ("int_value", "Int64"),
        ("string_value", "String"),
        ("query_count", "UInt32"),
        ("product_count", "UInt32"),
    ]

    create_table(client, "feature", columns, ["feature_id"], ["feature_id"])

    client.execute("SET allow_experimental_nlp_functions = 1;")

    def make_product_column_feature(
        feature_type: str, int_column: str, str_column: str
    ) -> str:
        return f"""
        (
                SELECT
                    '{feature_type}' feature_type,
                    {int_column} int_value,
                    {str_column} string_value,
                    0 query_count,
                    1 product_count
                FROM
                    product
        )
        """

    q = f"""
        INSERT INTO feature
        SELECT 
            rowNumberInAllBlocks() feature_id,
            feature_type,
            int_value,
            string_value,
            SUM(query_count) query_count,
            SUM(product_count) product_count
        FROM
        (
            {get_stems('product', 'title')}
            UNION ALL
            (
                SELECT
                    'stem' feature_type,
                    NULL int_value,
                    arrayJoin({make_stem_command('query_string')}) string_value,
                    1 query_count,
                    0 product_count
                FROM
                    ({get_distinct_queries()}) as queries
            )
            UNION ALL
            {make_product_column_feature('final_department_id', 'final_department_id', 'NULL')}
            UNION ALL
            {make_product_column_feature('product_type', 'NULL', 'product_type')}
            UNION ALL
            {make_product_column_feature('seller_id', 'seller_id', 'NULL')}
            UNION ALL
            {make_product_column_feature('store_id', 'store_id', 'NULL')}
        )
        GROUP BY feature_type, int_value, string_value
        ORDER BY product_count DESC
    """
    print(q)
    client.execute(q)


def get_stem_ids_from_column(table: str, id_column: str, target_column: str) -> str:
    return f"""
    SELECT
        {id_column},
        feature_id
    FROM
    (
        SELECT 
            {id_column},
            arrayJoin({make_stem_command(target_column)}) string_value
        FROM
            {table}
    ) x
    INNER JOIN feature ON (feature.feature_type = 'stem' AND feature.string_value = string_value)
    """


def make_product_feature_map_table(
    db: DatabaseInfo,
):
    client = db.make_client()
    client.execute("SET allow_experimental_nlp_functions = 1;")

    columns = [
        ("product_id", "UInt64"),
        ("feature_id", "UInt32"),
        ("context", "LowCardinality(String)"),
        ("num", "UInt16"),
    ]

    create_table(client, "product_feature_map", columns, ["product_id", "feature_id"])

    def get_stems_from_product(column: str):
        return f"""
            SELECT
                product_id,
                feature_id,
                '{column}' context
            FROM
                (
                SELECT 
                    product_id,
                    arrayJoin({make_stem_command(column)}) as string_value
                FROM
                    product
                ) product_data
                INNER JOIN 
                feature
                ON (feature.feature_type = 'stem' AND feature.string_value = string_value)
        """

    def get_feature_from_product_column(
        column: str,
        is_int: bool,
    ) -> str:

        if is_int:
            join_clause = f"feature.int_value = value"
        else:
            join_clause = f"feature.string_value = value"

        return f"""
           SELECT
                product_id,
                feature_id,
                '{column}' context
            FROM
                (
                SELECT 
                    product_id,
                    {column} value
                FROM
                    product
                ) product_data
                INNER JOIN 
                feature
                ON (feature.feature_type = '{column}' AND {join_clause})
        """

    client.execute(
        f"""
            INSERT INTO product_feature_map
            SELECT
                product_id,
                feature_id,
                context,
                COUNT(1) num
            FROM
            (
                {get_stems_from_product('title')}
                UNION ALL
                {get_feature_from_product_column('final_department_id', True)}
                UNION ALL
                {get_feature_from_product_column('product_type', False)}
                UNION ALL
                {get_feature_from_product_column('seller_id', True)}
                UNION ALL
                {get_feature_from_product_column('store_id', True)}
            )
            GROUP BY product_id, feature_id, context
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
        ("interaction_num", "UInt64"),
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
            dense_rank() OVER (ORDER BY date_created_rounded, user_num, query_id) interaction_num,
            user_num,
            date_created_rounded date_created,
            clicked,
            query_id,
            product_id  
        FROM
        (
            SELECT 
                user_num,
                round(date_created, -3) date_created_rounded,
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
            ORDER BY user_num, query_id, product_id, date_created
        ) src
        WHERE
            NOT is_click
        ORDER BY date_created_rounded, user_num, query_id
    """
    )


def make_user_feature_count_table(
    db: DatabaseInfo,
):
    client = db.make_client()
    columns = [
        # ("interaction_num", "UInt64"),
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
                user_num,
                feature_id,
                date_created,
                (SUM(clicks) OVER w) AS cumulative_clicks,
                (SUM(passes) OVER w) AS cumulative_passes
            FROM
            (
                SELECT
                    user_num,
                    product_feature_map.feature_id,
                    date_created,
                    SUM(clicked) clicks,
                    SUM(NOT clicked) passes
                FROM
                    click
                    INNER JOIN product_feature_map USING (product_id)
                WHERE
                    clicked IS NOT NULL
                GROUP BY user_num, feature_id, date_created
                ORDER BY user_num, feature_id, date_created ASC
                SETTINGS join_algorithm = 'partial_merge'
            ) src
            WINDOW
                w AS (
                    PARTITION BY (user_num, feature_id)
                    ORDER BY date_created ASC
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
                feature_id,
                date_created,
                (SUM(clicks) OVER w) AS cumulative_clicks,
                (SUM(passes) OVER w) AS cumulative_passes
            FROM
            (
                SELECT
                    product_feature_map.feature_id,
                    click.date_created,
                    SUM(clicked) clicks,
                    SUM(NOT clicked) passes
                FROM
                    click
                    INNER JOIN product_feature_map USING (product_id)
                WHERE
                    clicked IS NOT NULL
                GROUP BY feature_id, date_created
                ORDER BY feature_id, date_created ASC
            ) src
            WINDOW
                w AS (
                    PARTITION BY (feature_id) 
                    ORDER BY date_created ASC
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                    )
            """
    )


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
                (SUM(clicks) OVER w) AS cumulative_clicks,
                (SUM(passes) OVER w) AS cumulative_passes
            FROM
                (
                    SELECT
                        product_id,
                        date_created,
                        SUM(clicked) clicks,
                        SUM(NOT clicked) passes
                    FROM
                        click
                    GROUP BY product_id, date_created
                    ORDER BY product_id, date_created ASC
                ) src
            WINDOW
                w AS (
                    PARTITION BY (product_id) 
                    ORDER BY date_created ASC
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
                (SUM(clicks) OVER w) AS cumulative_clicks,
                (SUM(passes) OVER w) AS cumulative_passes
            FROM
                (
                    SELECT
                        user_num,
                        product_id,
                        date_created,
                        SUM(clicked) clicks,
                        SUM(NOT clicked) passes
                    FROM
                        click
                    GROUP BY user_num, product_id, date_created
                    ORDER BY user_num, product_id, date_created ASC
                ) src
            WINDOW
                w AS (
                    PARTITION BY (user_num, product_id) 
                    ORDER BY date_created ASC
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                    )
            """
    )


def make_global_click_count_table(
    db: DatabaseInfo,
):
    client = db.make_client()
    columns = [
        ("date_created", "UInt32"),
        ("cumulative_clicks", "UInt64"),
        ("cumulative_passes", "UInt64"),
    ]

    create_table(
        client,
        "global_click_count",
        columns,
        ["date_created"],
    )

    client.execute(
        f"""
            INSERT INTO global_click_count
            SELECT
                date_created,
                (SUM(clicks) OVER w) AS cumulative_clicks,
                (SUM(passes) OVER w) AS cumulative_passes
            FROM
                (
                    SELECT
                        date_created,
                        SUM(clicked) clicks,
                        SUM(NOT clicked) passes
                    FROM
                        click
                    GROUP BY date_created
                    ORDER BY date_created ASC
                ) src
            WINDOW
                w AS (
                    ORDER BY date_created ASC
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                    )
            """
    )


def make_user_counts_table(
    db: DatabaseInfo,
):
    client = db.make_client()
    columns = [
        ("user_num", "UInt64"),
        ("date_created", "UInt32"),
        ("cumulative_clicks", "UInt64"),
        ("cumulative_passes", "UInt64"),
    ]

    create_table(
        client,
        "user_counts",
        columns,
        ["user_num", "date_created"],
    )

    client.execute(
        f"""
            INSERT INTO user_counts
            SELECT
                user_num,
                date_created,
                (SUM(clicks) OVER w) AS cumulative_clicks,
                (SUM(passes) OVER w) AS cumulative_passes
            FROM
                (
                    SELECT
                        user_num,
                        date_created,
                        SUM(clicked) clicks,
                        SUM(NOT clicked) passes
                    FROM
                        click
                    GROUP BY user_num, date_created
                    ORDER BY user_num, date_created ASC
                ) src
            WINDOW
                w AS (
                    PARTITION BY (user_num) ORDER BY date_created ASC
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                    )
            """
    )


def make_interaction_query(clicked, limit, offset):
    return f"""
        SELECT 
            interaction_num, click_num, user_num, date_created, clicked, query_id, product_id 
        FROM click
        WHERE clicked = {clicked}
        ORDER BY intHash64(interaction_num) 
        LIMIT {limit} OFFSET {offset}
    """


def get_counts_for_interaction1(db: DatabaseInfo, interaction_query) -> pd.DataFrame:
    # aggregating feature counts

    # P(feature | click) = P(feature and click) / P(click) ~= # feature clicks / # clicks
    # P(feature | pass) = P(feature and pass) / P(pass) ~= # feature passes / # passes
    # O(feature | click) = P(feature | click) / P(feature| pass)
    #       = (# feature clicks / # clicks) / (# feature passes / # passes)
    #       = (# feature clicks / # feature passes) * (# passes / # clicks)

    client = db.make_client()
    user_feature_counts_query = f"""
        SELECT
            interaction_query.interaction_num interaction_num,
            interaction_query.click_num click_num,
            interaction_query.user_num user_num,
            interaction_query.date_created date_created,
            interaction_query.clicked clicked,
            interaction_query.query_id query_id,
            interaction_query.product_id product_id,

            global_click_count.cumulative_clicks clicks,
            global_click_count.cumulative_passes passes,
            global_product_count.cumulative_clicks product_clicks,
            global_product_count.cumulative_passes product_passes,
            user_counts.cumulative_clicks user_clicks,
            user_counts.cumulative_passes user_passes,
            user_product_count.cumulative_clicks user_product_clicks,
            user_product_count.cumulative_passes user_product_passes,

            product_feature_map.feature_id feature_id,
            (query_stem.feature_id = product_feature_map.feature_id) feature_in_query,
            global_feature_count.cumulative_clicks feature_clicks,
            global_feature_count.cumulative_passes feature_passes,
            user_feature_count.cumulative_clicks user_feature_clicks,
            user_feature_count.cumulative_passes user_feature_passes
        FROM
            ({interaction_query}) AS interaction_query
            INNER JOIN global_click_count USING (date_created)
            INNER JOIN global_product_count USING (product_id, date_created)
            INNER JOIN user_counts USING (user_num, date_created)
            INNER JOIN user_product_count USING (user_num, product_id, date_created)
            INNER JOIN product_feature_map USING (product_id)
            INNER JOIN global_feature_count USING (feature_id, date_created)
            INNER JOIN user_feature_count USING (user_num, feature_id, date_created)
            LEFT OUTER JOIN query_stem USING (query_id, feature_id)
        ORDER BY click_num
        """
    join_settings_clause = "\nSETTINGS join_algorithm = 'full_sorting_merge'"
    values = client.query_dataframe(user_feature_counts_query + join_settings_clause)
    return values


def get_counts_for_interaction(db: DatabaseInfo, interaction_query) -> pd.DataFrame:
    # aggregating feature counts

    # P(feature | click) = P(feature and click) / P(click) ~= # feature clicks / # clicks
    # P(feature | pass) = P(feature and pass) / P(pass) ~= # feature passes / # passes
    # O(feature | click) = P(feature | click) / P(feature| pass)
    #       = (# feature clicks / # clicks) / (# feature passes / # passes)
    #       = (# feature clicks / # feature passes) * (# passes / # clicks)

    client = db.make_client()
    user_feature_counts_query = f"""
        SELECT
            interaction.interaction_num interaction_num,
            interaction.click_num click_num,
            interaction.user_num user_num,
            interaction.date_created date_created,
            interaction.clicked clicked,
            interaction.query_id query_id,
            interaction.product_id product_id,

            global_click_count.cumulative_clicks clicks,
            global_click_count.cumulative_passes passes,
            global_product_count.cumulative_clicks product_clicks,
            global_product_count.cumulative_passes product_passes,
            user_counts.cumulative_clicks user_clicks,
            user_counts.cumulative_passes user_passes,
            user_product_count.cumulative_clicks user_product_clicks,
            user_product_count.cumulative_passes user_product_passes,

            interaction.feature_id feature_id,
            interaction.feature_in_query feature_in_query,
            interaction.feature_clicks feature_clicks,
            interaction.feature_passes feature_passes,
            interaction.user_feature_clicks user_feature_clicks,
            interaction.user_feature_passes user_feature_passes
        FROM
            (
                SELECT
                    interaction.interaction_num,
                    interaction.click_num click_num,
                    any(interaction.user_num) user_num,
                    any(interaction.date_created) date_created,
                    any(interaction.clicked) clicked,
                    any(interaction.query_id) query_id,
                    any(interaction.product_id) product_id,
                    groupArray(product_feature_map.feature_id) feature_id,
                    groupArray(query_stem.feature_id = product_feature_map.feature_id) feature_in_query,
                    groupArray(global_feature_count.cumulative_clicks) feature_clicks,
                    groupArray(global_feature_count.cumulative_passes) feature_passes,
                    groupArray(user_feature_count.cumulative_clicks) user_feature_clicks,
                    groupArray(user_feature_count.cumulative_passes) user_feature_passes
                FROM
                    ({interaction_query}) AS interaction
                    INNER JOIN product_feature_map USING (product_id)
                    INNER JOIN global_feature_count USING (feature_id, date_created)
                    INNER JOIN user_feature_count USING (user_num, feature_id, date_created)
                    LEFT OUTER JOIN query_stem USING (query_id, feature_id)
                GROUP BY interaction.interaction_num, interaction.click_num
            ) AS interaction            
            INNER JOIN global_click_count USING (date_created)
            INNER JOIN global_product_count USING (product_id, date_created)
            INNER JOIN user_counts USING (user_num, date_created)
            INNER JOIN user_product_count USING (user_num, product_id, date_created)
            
        ORDER BY click_num
        """
    join_settings_clause = "\nSETTINGS join_algorithm = 'full_sorting_merge'"
    interactions = client.query_dataframe(
        user_feature_counts_query + join_settings_clause
    )

    feature_columns = (
        "feature_id",
        "feature_in_query",
        "feature_clicks",
        "feature_passes",
        "user_feature_clicks",
        "user_feature_passes",
    )

    interactions["features"] = interactions.apply(
        lambda row: pd.DataFrame({k: row[k] for k in feature_columns}),  # type: ignore
        axis=1,
    )  # type: ignore

    interactions["event"] = interactions["clicked"].astype(np.float32)
    return interactions


def briar_score(probability, event):
    return np.square(probability - event)


def log_score(probability, event):
    return event * -np.log(probability) + (1 - event) * -np.log(1.0 - probability)


def accuracy_score(probability, event):
    return (probability > 0.5) * event + (probability < 0.5) * (1 - event)
