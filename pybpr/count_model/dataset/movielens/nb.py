from typing import Dict
import pandas
from pybpr.count_model.assessor.naive_bayes import NaiveBayes
from pybpr.count_model.dataset.movielens.data import *
import pybpr.count_model.assessor.factored_naive_bayes as factored_naive_bayes


def get_user_set(ratings):
    return ratings["user_id"].drop_duplicates().sort_values()


def get_user_set2(ratings):
    return ratings["user_id"]


def dataset_loc(
    dataframe,
    index,
):
    if len(index) > 0:
        try:
            return dataframe.loc[index, :]
        except KeyError as e:
            # print(e)
            pass
    return pandas.DataFrame(columns=dataframe.columns)


def make_nb_prior(
    ratings_by_item: pandas.DataFrame,
    global_prior: NBElement,
) -> Callable[[Interaction], NBElement]:
    def get_nb_prior(query: Interaction) -> NBElement:
        # P(rating | all ratings on this movie)
        timestamp_range = slice(None, query.timestamp - 1)
        num_matching = len(
            dataset_loc(
                ratings_by_item, (query.object, query.verb, timestamp_range)
            ).index
        )

        num_opposite = len(
            dataset_loc(
                ratings_by_item, (query.object, not query.verb, timestamp_range)
            ).index
        )

        return global_prior + NBElement(num_matching, num_opposite + num_matching)

    return get_nb_prior


def get_count(dataframe, index) -> int:
    if len(index) > 0:
        try:
            s = dataframe.loc[index, "count"]
            if len(s) > 0:
                return s.iloc[-1]
        except KeyError as e:
            pass
    return 0


def make_nb_prior_with_cumulative_sums(
    per_item_interaction_counts: pandas.DataFrame,
    global_prior: NBElement,
) -> Callable[[Interaction], NBElement]:
    def get_nb_prior(query: Interaction) -> NBElement:
        # P(rating | all ratings on this movie)
        timestamp_range = slice(None, query.timestamp - 1)

        num_matching = get_count(
            per_item_interaction_counts,
            pandas.IndexSlice[query.object, query.verb, timestamp_range],
        )
        num_opposite = get_count(
            per_item_interaction_counts,
            pandas.IndexSlice[query.object, not query.verb, timestamp_range],
        )

        return global_prior + NBElement(num_matching, num_opposite + num_matching)

    return get_nb_prior


def make_interaction_factors_accessor(
    interaction_prior: NBElement,
    ratings_by_user: pandas.DataFrame,
    ratings_by_item_user: pandas.DataFrame,
) -> Callable[[Interaction], Iterable[NBFactor]]:
    def get_interaction_factors(query: Interaction) -> Iterable[NBFactor]:
        timestamp_range = slice(None, query.timestamp - 1)
        users_previous_ratings = dataset_loc(
            ratings_by_user, (query.subject, timestamp_range)
        )
        # previous_ratings = ratings[ratings['timestamp'] < interaction.timestamp]
        factors = []

        def time_filter(df):
            return df.loc[df["timestamp"] < query.timestamp]

        # previous ratings of this item
        matching_ratings = time_filter(
            dataset_loc(
                ratings_by_item_user,
                (
                    query.object,
                    query.verb,
                    slice(None),
                ),
            )
        )["user_id"]

        nonmatching_ratings = time_filter(
            dataset_loc(
                ratings_by_item_user,
                (
                    query.object,
                    not query.verb,
                    slice(None),
                ),
            )
        )["user_id"]

        for (
            row_index,
            row,
        ) in users_previous_ratings.iterrows():  # for each previous rating on a movie
            previous_interaction = make_user_item_interaction(row)

            def make_element(
                matching_ratings,
            ) -> NBElement:
                other_item_matching_ratings = len(
                    time_filter(
                        dataset_loc(
                            ratings_by_item_user,
                            (
                                previous_interaction.object,
                                previous_interaction.verb,
                                matching_ratings,
                                # timestamp_range,
                            ),
                        )
                    ).index
                )

                other_item_nonmatching_ratings = len(
                    time_filter(
                        dataset_loc(
                            ratings_by_item_user,
                            (
                                previous_interaction.object,
                                not previous_interaction.verb,
                                matching_ratings,
                                # timestamp_range,
                            ),
                        )
                    ).index
                )

                return interaction_prior + NBElement(
                    other_item_matching_ratings,
                    other_item_matching_ratings + other_item_nonmatching_ratings,
                )

            factors.append(
                NBFactor(
                    # P(previous rating | same rating)
                    # of all occourances of this rating, what proportion co-occoured with previous rating?
                    # num prev rating & this rating  / (num this rating & either prev ratings)
                    # overall set occourances of a rating on both items
                    # positive ratio filters by occourances of this rating
                    # get coocourances of this rating and a rating on this item
                    make_element(matching_ratings),
                    # P(previous rating | oppostie rating)
                    # of all occourances of the opposite rating, what proportion co-occoured with previous rating?
                    # num prev rating & opposite rating  / (num opposite rating & either prev ratings)
                    make_element(
                        nonmatching_ratings,
                    ),
                )
            )
        return factors

    return get_interaction_factors


def make_interaction_factors_accessor_prev(
    interaction_prior: NBElement,
    ratings_by_user: pandas.DataFrame,
    ratings_by_item: pandas.DataFrame,
    ratings_by_item_user: pandas.DataFrame,
) -> Callable[[Interaction], Iterable[NBFactor]]:
    def get_interaction_factors(query: Interaction) -> Iterable[NBFactor]:
        timestamp_range = slice(None, query.timestamp - 1)
        users_previous_ratings = dataset_loc(
            ratings_by_user, (query.subject, timestamp_range)
        )
        # previous_ratings = ratings[ratings['timestamp'] < interaction.timestamp]
        factors = []

        # user id's of all users who rated this movie the same
        matching_ratings = get_user_set2(
            dataset_loc(ratings_by_item, (query.object, query.verb, timestamp_range))
        )

        # user id's of all users who rated this movie differently
        nonmatching_ratings = get_user_set2(
            dataset_loc(
                ratings_by_item, (query.object, not query.verb, timestamp_range)
            )
        )

        for (
            row_index,
            row,
        ) in users_previous_ratings.iterrows():  # for each previous rating on a movie
            previous_interaction = make_user_item_interaction(row)

            def make_element(
                # evidence: Interaction,  # this rating
                query: Interaction,  # previous rating
                evidence_observations,  # all occourances of this rating
            ) -> NBElement:
                co_occourances = len(
                    dataset_loc(
                        ratings_by_item_user,
                        (
                            query.object,
                            query.verb,
                            evidence_observations,
                            timestamp_range,
                        ),
                    ).index
                )

                return interaction_prior + NBElement(
                    co_occourances,
                    len(evidence_observations.index),
                )

            factors.append(
                NBFactor(
                    # P(previous rating | same rating)
                    # of all occourances of this rating, what proportion co-occoured with previous rating?
                    make_element(
                        previous_interaction,
                        matching_ratings,
                    ),
                    # P(previous rating | oppostie rating)
                    # of all occourances of the opposite rating, what proportion co-occoured with previous rating?
                    make_element(
                        previous_interaction,
                        nonmatching_ratings,  # type: ignore
                    ),
                )
            )
        return factors

    return get_interaction_factors


def make_genre_factors_accessor(
    movies: pandas.DataFrame,
    ratings_by_time: pandas.DataFrame,
    ratings_by_genre: Dict[str, pandas.DataFrame],
    interaction_prior: NBElement,
) -> Callable[[Interaction], Iterable[NBFactor]]:
    def get_factors(query: Interaction) -> Iterable[NBFactor]:

        def filter_out_this_movie(df):
            return df
            # return df.loc[df["item_id"] != query.object]

        timestamp_range = slice(None, query.timestamp - 1)
        matching_ratings = filter_out_this_movie(
            dataset_loc(ratings_by_time, (query.verb, timestamp_range))
        )

        non_matching_ratings = filter_out_this_movie(
            dataset_loc(ratings_by_time, (not query.verb, timestamp_range))
        )

        movie = movies.loc[query.object]
        factors = []
        for genre in category_columns:
            movie_in_genre = movie[genre]
            if not movie_in_genre:
                continue

            genre_ratings = ratings_by_genre[genre]

            # P(genre value | same rating)
            # of all occourances of this rating, what proportion had this genre?
            matching_factor = interaction_prior + NBElement(
                len(
                    filter_out_this_movie(
                        dataset_loc(
                            genre_ratings,
                            (movie_in_genre, query.verb, timestamp_range),
                        )
                    ).index
                ),
                len(matching_ratings.index),
            )

            # P(genre value | opposite rating)
            # of all occourances of the opposite rating, what proportion had this genre?
            not_matching_factor = interaction_prior + NBElement(
                len(
                    filter_out_this_movie(
                        dataset_loc(
                            genre_ratings,
                            (movie_in_genre, not query.verb, timestamp_range),
                        )
                    ).index
                ),
                len(non_matching_ratings.index),
            )

            factor = NBFactor(matching_factor, not_matching_factor)
            # print(f"genre: {genre}, movie_in_genre {movie_in_genre}, factor: {factor}")

            # print(
            #     f"rating: {query.verb} genre: {genre}, movie_in_genre {movie_in_genre}, factor: {factor}"
            # )
            factors.append(factor)
            break
        return factors
        # return [NBFactor(NBElement(0.9, 1), NBElement(0.1, 1))]

    return get_factors


def make_personalized_genre_factors_accessor(
    movies: pandas.DataFrame,
    ratings_by_user: pandas.DataFrame,
    # interaction_prior: NBElement,
    factor_prior_accessor: Callable[[Interaction], Iterable[NBFactor]],
    prior_weight: float,
) -> Callable[[Interaction], Iterable[NBFactor]]:
    def get_factors(query: Interaction) -> Iterable[NBFactor]:
        timestamp_range = slice(None, query.timestamp - 1)
        users_previous_ratings = dataset_loc(
            ratings_by_user, (query.subject, timestamp_range)
        )

        users_previous_matching_ratings = users_previous_ratings[
            users_previous_ratings["positive"] == query.verb
        ]
        users_previous_nonmatching_ratings = users_previous_ratings[
            users_previous_ratings["positive"] != query.verb
        ]

        # print(
        #     f"query: {query}, matching: {len(users_previous_matching_ratings)}, nonmatching: {len(users_previous_nonmatching_ratings)}"
        # )

        prior_factors = factor_prior_accessor(query)
        movie = movies.loc[query.object]
        # print(movie)
        factors = []
        for genre, prior_factor in zip(category_columns, prior_factors):
            movie_in_genre = movie[genre]
            # P(genre matches | same rating)
            # of all occourances of this rating, what proportion had this genre?
            matching_factor = prior_factor.positive_element.rescaled(
                prior_weight
            ) + NBElement(
                len(
                    users_previous_matching_ratings[
                        users_previous_matching_ratings[genre] == movie_in_genre
                    ]
                ),
                len(users_previous_matching_ratings),
            )

            # P(genre matches | opposite rating)
            # of all occourances of the opposite rating, what proportion had this genre?
            not_matching_factor = prior_factor.negative_element.rescaled(
                prior_weight
            ) + NBElement(
                len(
                    users_previous_nonmatching_ratings[
                        users_previous_nonmatching_ratings[genre] == movie_in_genre
                    ]
                ),
                len(users_previous_nonmatching_ratings),
            )

            factor = NBFactor(matching_factor, not_matching_factor)
            # print(f"genre: {genre}, movie_in_genre {movie_in_genre}, factor: {factor}")

            factors.append(factor)
        return factors

    return get_factors


def make_assess_nb(
    prior_assesor: Callable[[Interaction], NBElement],
    factor_generators: List[Callable[[Interaction], Iterable[NBFactor]]],
):
    def assess(interaction: Interaction) -> float:
        return factored_naive_bayes.compute_naive_bayes(
            prior_assesor(interaction).probability,
            itertools.chain.from_iterable(
                (
                    factor_generator(interaction)
                    for factor_generator in factor_generators
                )
            ),
        )

    return assess


def make_item_factor_accessor(
    interaction_prior: NBElement,
    ratings_by_item: pandas.DataFrame,
    ratings_by_time: pandas.DataFrame,
) -> Callable[[Interaction], Iterable[NBFactor]]:
    def get_factors(query: Interaction) -> Iterable[NBFactor]:
        timestamp_range = slice(None, query.timestamp - 1)

        # P(id | rating) / P(id | not rating)
        # (#ratings with id / # ratings) / (# not ratings with id / # not ratings)
        matching_factor = interaction_prior + NBElement(
            len(
                dataset_loc(
                    ratings_by_item, (query.object, query.verb, timestamp_range)
                ).index
            ),
            len(dataset_loc(ratings_by_time, (query.verb, timestamp_range)).index),
        )

        non_matching_factor = interaction_prior + NBElement(
            len(
                dataset_loc(
                    ratings_by_item, (query.object, not query.verb, timestamp_range)
                ).index
            ),
            len(dataset_loc(ratings_by_time, (not query.verb, timestamp_range)).index),
        )

        return [NBFactor(matching_factor, non_matching_factor)]

    return get_factors
