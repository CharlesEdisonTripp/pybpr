import pandas
from pybpr.count_model.assessor.naive_bayes import NaiveBayes
from pybpr.count_model.dataset.movielens.data import *
import pybpr.count_model.assessor.factored_naive_bayes as factored_naive_bayes


def get_user_set(ratings):
    return ratings["user_id"].drop_duplicates().sort_values()


def dataset_loc(
    dataframe,
    index,
):
    if len(index) > 0:
        try:
            return dataframe.loc[index, :]
        except KeyError:
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
            dataset_loc(ratings_by_item, (query.object, query.verb, timestamp_range))
        )
        num_opposite = len(
            dataset_loc(
                ratings_by_item, (query.object, not query.verb, timestamp_range)
            )
        )
        # num_positive = len(
        #     ratings_by_item.loc[query.object, query.verb, timestamp_range]
        # )  # type: ignore
        # num_negative = len(
        #     ratings_by_item.loc[query.object, not query.verb, timestamp_range]
        # )  # type: ignore

        return global_prior + NBElement(num_matching, num_opposite + num_matching)

    return get_nb_prior


def make_interaction_factors_accessor(
    ratings_by_user: pandas.DataFrame,
    ratings_by_item: pandas.DataFrame,
    ratings_by_item_user: pandas.DataFrame,
    interaction_prior: NBElement,
) -> Callable[[Interaction], Iterable[NBFactor]]:
    def get_interaction_factors(query: Interaction) -> Iterable[NBFactor]:
        timestamp_range = slice(None, query.timestamp - 1)
        users_previous_ratings = dataset_loc(
            ratings_by_user, (query.subject, timestamp_range)
        )
        # previous_ratings = ratings[ratings['timestamp'] < interaction.timestamp]
        factors = []

        # user id's of all users who rated this movie the same
        matching_ratings = get_user_set(
            dataset_loc(ratings_by_item, (query.object, query.verb, timestamp_range))
        )

        # user id's of all users who rated this movie differently
        nonmatching_ratings = get_user_set(
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
                    )
                )

                return NBElement(
                    co_occourances,
                    evidence_observations.size,
                )

            factors.append(
                NBFactor(
                    # P(previous rating | same rating)
                    # of all occourances of this rating, what proportion co-occoured with previous rating?
                    interaction_prior
                    + make_element(
                        previous_interaction,
                        matching_ratings,
                    ),
                    # P(previous rating | oppostie rating)
                    # of all occourances of the opposite rating, what proportion co-occoured with previous rating?
                    interaction_prior
                    + make_element(
                        previous_interaction.negative(),
                        nonmatching_ratings,  # type: ignore
                    ),
                )
            )
        return factors

    return get_interaction_factors


def make_genre_factors_accessor(
    movies: pandas.DataFrame,
    ratings_by_user: pandas.DataFrame,
    interaction_prior: NBElement,
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

        print(
            f"query: {query}, matching: {len(users_previous_matching_ratings)}, nonmatching: {len(users_previous_nonmatching_ratings)}"
        )

        movie = movies.loc[query.object]
        # print(movie)
        factors = []
        for genre in category_columns:
            movie_in_genre = movie[genre]
            # P(genre matches | same rating)
            # of all occourances of this rating, what proportion had this genre?
            matching_factor = interaction_prior + NBElement(
                len(
                    users_previous_matching_ratings[
                        users_previous_matching_ratings[genre] == movie_in_genre
                    ]
                ),
                len(users_previous_matching_ratings),
            )
            # matching_factor = interaction_prior + NBElement(
            #     100,
            #     100,
            # )

            # P(genre matches | opposite rating)
            # of all occourances of the opposite rating, what proportion had this genre?
            not_matching_factor = interaction_prior + NBElement(
                len(
                    users_previous_nonmatching_ratings[
                        users_previous_nonmatching_ratings[genre] == movie_in_genre
                    ]
                ),
                len(users_previous_nonmatching_ratings),
            )
            # not_matching_factor = interaction_prior + NBElement(0, 100)

            factor = NBFactor(matching_factor, not_matching_factor)
            print(f"genre: {genre}, movie_in_genre {movie_in_genre}, factor: {factor}")

            factors.append(factor)
        return factors

    return get_factors


def make_assess_nb(
    prior_assesor: Callable[[Interaction], NBElement],
    factor_generators: List[Callable[[Interaction], Iterable[NBFactor]]],
):
    def assess(interaction: Interaction) -> float:
        p = factored_naive_bayes.compute_naive_bayes(
            prior_assesor(interaction).probability,
            itertools.chain.from_iterable(
                (
                    factor_generator(interaction)
                    for factor_generator in factor_generators
                )
            ),
        )

        # print(f"target: {target_event.verb} {p}")
        return p

    return assess
