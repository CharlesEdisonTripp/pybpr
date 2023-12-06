import itertools
import os
from typing import Any, Callable, Iterable, Iterator, List, Optional, Tuple
from pybpr.count_model.assessor.nb_element import NBElement
from pybpr.count_model.assessor.nb_factor import NBFactor
from pybpr.count_model.event_counter import EventCounter
import numpy

import pandas

from pybpr.count_model.evaluate.evaluate import (
    brier_score,
    compute_ndcg,
    log_score,
    prob_score,
    accuracy_score,
)
from pybpr.count_model.evaluate.evaluation import Evaluation
from pybpr.count_model.evaluate.score_summary import ScoreSummary
from pybpr.count_model.interaction import Interaction


# TODO: compute and graph distribution based on # of previous ratings and scores (incl iqr/var at each point)


def load_movielens_file(dataset_name, filename, names, sep="\t"):
    data_path = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
                    )
                )
            )
        ),
        "data",
    )
    file_path = os.path.join(data_path, dataset_name, filename)
    return pandas.read_csv(file_path, sep=sep, names=names, encoding="iso_8859_1")


def load_movielens_data(dataset_name="ml-100k"):
    """Function to read movielens data"""
    filename = {
        "ml-100k": "u.data",
        "ml-1m": "ratings.dat",
        "ml-25m": "ratings.csv",
        "ml-10M100K": "ratings.dat",
    }[dataset_name]
    df = load_movielens_file(
        dataset_name,
        filename,
        [
            "user_id",
            "item_id",
            "rating",
            "timestamp",
        ],
    )

    df["user_id"] = df["user_id"].astype("int32")
    df["item_id"] = df["item_id"].astype("int32")
    df["rating"] = df["rating"].astype("int8")
    df["timestamp"] = df["timestamp"].astype("int64")

    # df.set_index(["timestamp"], inplace=True, drop=False)
    # df.sort_index(inplace=True)
    df.sort_values(["timestamp", "user_id"], inplace=True)

    # df.reset_index(inplace=True)
    # df.drop(columns="index", inplace=True)
    return df


category_columns = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


def load_movielens_items(dataset_name="ml-100k"):
    """Function to read movielens items"""

    filename = {
        "ml-100k": "u.item",
        "ml-25m": "movies.csv",
    }[dataset_name]

    movies = load_movielens_file(
        dataset_name,
        filename,
        [
            "movie id",
            "movie title",
            "release date",
            "video release date",
            "IMDb URL",
        ]
        + category_columns,
        sep="|",
    )
    movies.rename(columns={"movie id": "item_id"}, inplace=True)

    movies["item_id"] = movies["item_id"].astype("int32")

    for column in category_columns:
        movies[column] = movies[column].astype("bool")

    

    movies.set_index("item_id", inplace=True, drop=False)
    movies.sort_index(inplace=True)
    return movies


def make_user_item_interaction(row) -> Interaction:
    return Interaction(
        subject=int(row["user_id"]),
        verb=row["positive"] == 1,
        object=int(row["item_id"]),
        timestamp=row["timestamp"],
    )


def get_stem_interactions(row) -> Iterable[Interaction]:
    # for row_index, row in df.iterrows():
    return (
        Interaction(
            subject=int(row["user_id"]),
            verb=row["positive"] == 1,
            object=stem,
            timestamp=row["timestamp"],
        )  # user, rating/action, stem
        for stem in row["stems"]
    )


def stem_movies(movies):
    import nltk
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize

    movies_ = movies.copy()
    stemmer = PorterStemmer()
    movies["stems"] = movies["movie title"].apply(
        lambda text: [
            stemmer.stem(token) for token in nltk.tokenize.word_tokenize(text[:-6])
        ]
    )
    return movies


def join_ratings_and_movies(ratings, movies):
    return ratings.join(movies, on="item_id", how="left", lsuffix="", rsuffix="_")


# def train_test_split(ratings):
#     ratings = ratings.sample(frac=1)  # shuffle df

#     test_proportion = 0.1
#     total = len(ratings)
#     slice_index = int((1.0 - test_proportion) * total)

#     print(f"{total} total entries. {(1-test_proportion)*100}% = {slice_index}")

#     # train_df = df.iloc[:slice_index].sort_values(['user_id', 'timestamp']).reset_index()
#     # test_df = df.iloc[slice_index:].sort_values(['user_id', 'timestamp']).reset_index()
#     train_df = ratings.iloc[:slice_index].sort_values(["user_id", "timestamp"])
#     test_df = ratings.iloc[slice_index:].sort_values(["user_id", "timestamp"])
#     # test_timestamp = test_df['timestamp'].iloc[0]
#     # print(f'test_timestamp: {test_timestamp}')
#     return test_df, train_df


action_key = "a"
object_key = "o"
verb_object_key = "vo"
interaction_key = "i"
unconditional_interaction_key = "u"
link_key = "l"

stem_key = "s"
stem_action_key = "sa"
stem_user_action_key = "sua"


def count_events(train_df):
    counter = EventCounter()

    user_groups = train_df.groupby("user_id")
    for user_id, group in user_groups:
        user_df = user_groups.get_group(user_id)

        interactions = []
        for row_index, row in user_df.iterrows():
            interaction = make_user_item_interaction(row)

            # counter.observe(interaction.subject)  # user action
            # counter.observe(interaction.subject, interaction.verb)  # user, rating

            # counter.observe(interaction.subject, interaction.verb, interaction.object)  # user, rating, movie
            # counter.observe(interaction.subject, interaction.object)  # user-movie interaction
            interactions.append(interaction)
            counter.observe(action_key, interaction.verb)

            for stem_interaction in get_stem_interactions(row):
                counter.observe(
                    stem_key, stem_interaction.object
                )  # interaction on stem
                counter.observe(
                    stem_action_key, stem_interaction.verb, stem_interaction.object
                )  # rating on stem
                counter.observe(
                    stem_user_action_key,
                    stem_interaction.subject,
                    stem_interaction.verb,
                    stem_interaction.object,
                )  # user rated stem

            #     # counter.observe(stem_interaction.subject)  # user action
            #     # counter.observe(stem_interaction.subject, stem_interaction.verb)  # user, rating

            #     counter.observe(stem_interaction.subject, stem_interaction.verb, stem_interaction.object)  # user, rating, stem
            #     counter.observe(stem_interaction.subject, stem_interaction.object)  # user-stem interaction

        # print(f'{user_df.size} {interactions}')
        for src in interactions:
            counter.observe(object_key, src.object)  # action taken on movie
            counter.observe(verb_object_key, src.verb, src.object)  # rating, movie

            for dst in interactions:
                if src == dst:
                    continue
                counter.observe(
                    interaction_key, src.verb, src.object, dst.verb, dst.object
                )  # (rating, movie) -> (rating, movie)
                counter.observe(
                    unconditional_interaction_key, src.object, dst.verb, dst.object
                )  # (movie) -> (rating, movie)
                counter.observe(link_key, src.object, dst.object)  # (movie) -> (movie)
    return counter


def compute_evaluations(
    interactions: Iterable[Interaction],
    assess: Callable[[Interaction], float],
    evaluation_functions: Optional[
        Iterable[Tuple[str, Callable[[float, Interaction], float]]]
    ] = None,
) -> pandas.DataFrame:
    if evaluation_functions is None:
        evaluation_functions = (
            ("log", log_score),
            ("brier", brier_score),
            ("prob", prob_score),
            ("accuracy", accuracy_score),
        )
    evaluation_data = [(*t, []) for t in evaluation_functions]
    for interaction in interactions:
        assesment = assess(interaction)
        for name, evaluation_func, evaluations in evaluation_data:
            evaluations.append(evaluation_func(assesment, interaction))

    print(
        [
            (name, len(evaluations))
            for name, evaluation_func, evaluations in evaluation_data
        ]
    )
    return pandas.DataFrame(
        {name: evaluations for name, evaluation_func, evaluations in evaluation_data}
    )


def compute_evaluations_from_dataframe(
    interaction_dataframe: pandas.DataFrame,
    assess: Callable[[Interaction], float],
    evaluation_functions: Optional[
        Iterable[Tuple[str, Callable[[float, Interaction], float]]]
    ] = None,
) -> pandas.DataFrame:
    return compute_evaluations(
        (
            make_user_item_interaction(row)
            for row_index, row in interaction_dataframe.iterrows()
        ),
        assess,
        evaluation_functions,
    )


def compute_scores(
    test_df: pandas.DataFrame,
    assess: Callable[[Interaction], float],
):
    evaluations = compute_evaluations_from_dataframe(
        test_df,
        assess,
        None,
    )
    print(len(evaluations), len(test_df))

    merged_evaluations = pandas.concat(
        (test_df.reset_index(drop=True), evaluations),
        axis=1,
        copy=False,
    )

    def summarize_evaluations(evals):
        result = []
        for column in evaluations.columns:
            e = merged_evaluations[column]
            result.append(
                Evaluation(
                    column,
                    e.mean(),
                    e[merged_evaluations["positive"] == 1].mean(),
                    e[merged_evaluations["positive"] == 0].mean(),
                )
            )
        return result

    return ScoreSummary(
        summarize_evaluations(evaluations),
        0.0,
        # compute_mean_ndcg(df, test_df, assessment_function),
        # summarize_evaluations(static_evaluations),
        # compute_mean_ndcg(train_df, test_df, assessment_function),
    )


def compute_ndcg_for_counter(
    train_df,
    test_df,
    assess_function,
):
    ndcgs = []
    for user_id, test_actions in test_df.groupby("user_id"):
        conditioning_actions = [
            make_user_item_interaction(row)
            for row_index, row in train_df[train_df["user_id"] == user_id].iterrows()
        ]
        user_action_seq = [
            (
                assess_function(
                    conditioning_actions,
                    make_user_item_interaction(action_row),
                ),
                action_row["positive"],
            )
            for idx, action_row in test_actions.iterrows()
        ]
        ndcg = compute_ndcg(user_action_seq)
        ndcgs.append(ndcg)
    return numpy.mean(ndcgs)


def compute_mean_ndcg(
    train_df,
    test_df,
    assess_function,
) -> float:
    ndcgs = []
    for user_id, test_actions in test_df.groupby("user_id"):
        train_user_actions = train_df[train_df["user_id"] == user_id]

        # get_conditioning_actions = None
        # if train_user_actions['timestamp'].max() < test_actions['timestamp'].min():
        #     # static mode
        #     conditioning_actions = list(
        #         make_user_item_interaction_sequence(
        #             train_user_actions
        #         )
        #     )
        #     get_conditioning_actions = lambda action_row: conditioning_actions
        # else:
        #     # dynamic mode
        #     get_conditioning_actions = lambda action_row: train_user_actions[
        #         train_user_actions['timestamp'] < action_row['timestamp']
        #     ]

        # def make_conditioning_interactions(action_row):
        #     return

        user_action_seq = [
            (
                assess_function(
                    train_user_actions[
                        train_user_actions["timestamp"] < action_row["timestamp"]
                    ],
                    action_row,
                ),
                action_row["positive"],
            )
            for idx, action_row in test_actions.iterrows()
        ]
        ndcg = compute_ndcg(user_action_seq)
        ndcgs.append(ndcg)
    return float(numpy.mean(ndcgs))
