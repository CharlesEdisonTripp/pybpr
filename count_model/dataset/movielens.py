import os
import numpy

import pandas

from count_model.evaluate.evaluate import (
    brier_score,
    compute_ndcg,
    log_score,
    prob_score,
    accuracy_score,
)
from count_model.evaluate.evaluation import Evaluation
from count_model.evaluate.score_summary import ScoreSummary
from count_model.interaction import Interaction


# TODO: compute and graph distribution based on # of previous ratings and scores (incl iqr/var at each point)


def load_movielens_file(dataset_name, filename, names, sep='\t'):
    data_path = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            )
        ),
        'data',
    )
    file_path = os.path.join(data_path, dataset_name, filename)
    return pandas.read_csv(file_path, sep=sep, names=names, encoding='iso_8859_1')


def load_movielens_data(dataset_name='ml-100k'):
    """Function to read movielens data"""
    filename = {
        'ml-100k': 'u.data',
        'ml-1m': 'ratings.dat',
        'ml-25m': 'ratings.csv',
        'ml-10M100K': 'ratings.dat',
    }[dataset_name]
    df = load_movielens_file(
        dataset_name,
        filename,
        [
            'user_id',
            'item_id',
            'rating',
            'timestamp',
        ],
    )

    df.sort_values(['timestamp', 'user_id'], inplace=True)
    df.reset_index(inplace=True)
    df.drop(columns='index', inplace=True)
    return df



def load_movielens_items(dataset_name='ml-100k'):
    """Function to read movielens items"""

    filename = {
        'ml-100k': 'u.item',
        'ml-25m': 'movies.csv',
    }[dataset_name]
    df = load_movielens_file(
        dataset_name,
        filename,
        [
            'movie id',
            'movie title',
            'release date',
            'video release date',
            'IMDb URL',
            'unknown',
            'Action',
            'Adventure',
            'Animation',
            "Children's",
            'Comedy',
            'Crime',
            'Documentary',
            'Drama',
            'Fantasy',
            'Film-Noir',
            'Horror',
            'Musical',
            'Mystery',
            'Romance',
            'Sci-Fi',
            'Thriller',
            'War',
            'Western',
        ],
        sep='|',
    )
    df.rename(columns={'movie id':'item_id'}, inplace=True)
    df.set_index('item_id', inplace=True)
    return df




def get_user_id(row)->int:
    return int(row['user_id'])

def get_action_id(row)->bool:
    return row['positive'] == 1

def get_item_id(row)->int:
    return int(row['item_id'])


def make_user_item_interaction(row):
    return Interaction(
        get_user_id(row),
        get_action_id(row),
        get_item_id(row),
    )

def make_user_item_interaction_sequence(df):
    return (make_user_item_interaction(row) for row_index, row in df.iterrows())



def compute_evaluations(
    train_df,
    test_df,
    assess_function,
    evaluation_functions,
):
    action_indicies = []
    num_conditioning_actions_list = []
    evaluation_data = [(*t, []) for t in evaluation_functions]

    # action_attrs = ('user_id', 'timestamp', 'positive')

    for user_id, test_actions in test_df.groupby('user_id'):
        train_user_actions = train_df[train_df['user_id'] == user_id]

        get_conditioning_actions = None
        if train_user_actions['timestamp'].max() < test_actions['timestamp'].min():
            # static mode
            conditioning_actions_ = list(make_action_sequence(train_user_actions))
            get_conditioning_actions = lambda action_row: conditioning_actions_
        else:
            # dynamic mode
            get_conditioning_actions = lambda action_row: list(
                make_action_sequence(
                    train_user_actions[
                        train_user_actions['timestamp'] < action_row['timestamp']
                    ]
                )
            )

        action_indicies.extend(test_actions.index)
        # print(test_actions.loc[test_actions.index].head())

        for idx, action_row in test_actions.iterrows():
            conditioning_actions = get_conditioning_actions(action_row)
            action = make_interaction_tuple(action_row)
            assesment = assess_function(
                conditioning_actions,
                action,
            )

            num_conditioning_actions_list.append(len(conditioning_actions))
            for name, evaluation_func, evaluations in evaluation_data:
                evaluations.append(
                    evaluation_func(
                        assesment,
                        conditioning_actions,
                        action,
                    )
                )

    res = test_df.loc[action_indicies]
    res['num_conditioning_actions'] = num_conditioning_actions_list
    for name, evaluation_func, evaluations in evaluation_data:
        res[name] = evaluations
    return res


def compute_ndcg_for_counter(
    train_df,
    test_df,
    assess_function,
):
    ndcgs = []
    for user_id, test_actions in test_df.groupby('user_id'):
        conditioning_actions = list(
            make_action_sequence(train_df[train_df['user_id'] == user_id])
        )
        user_action_seq = [
            (
                assess_function(
                    conditioning_actions,
                    make_interaction_tuple(action_row),
                ),
                action_row['positive'],
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
    for user_id, test_actions in test_df.groupby('user_id'):
        train_user_actions = train_df[train_df['user_id'] == user_id]

        get_conditioning_actions = None
        if train_user_actions['timestamp'].max() < test_actions['timestamp'].min():
            # static mode
            conditioning_actions = list(
                make_action_sequence(
                    train_user_actions[train_user_actions['user_id'] == user_id]
                )
            )
            get_conditioning_actions = lambda action_row: conditioning_actions
        else:
            # dynamic mode
            get_conditioning_actions = lambda action_row: train_user_actions[
                train_user_actions['timestamp'] < action_row['timestamp']
            ]

        user_action_seq = [
            (
                assess_function(
                    get_conditioning_actions(action_row),
                    make_interaction_tuple(action_row),
                ),
                action_row['positive'],
            )
            for idx, action_row in test_actions.iterrows()
        ]
        ndcg = compute_ndcg(user_action_seq)
        ndcgs.append(ndcg)
    return float(numpy.mean(ndcgs))


def compute_scores(
    df,
    train_df,
    test_df,
    assessment_function,
):
    scores = (
        ('log', log_score),
        ('brier', brier_score),
        ('prob', prob_score),
        ('accuracy', accuracy_score),
    )

    dynamic_evaluations = compute_evaluations(df, test_df, assessment_function, scores)
    # static_evaluations = compute_evaluations(train_df, test_df, assessment_function, scores)

    def summarize_evaluations(evals):
        result = []
        for name, _ in scores:
            e = evals[name]
            result.append(
                Evaluation(
                    name,
                    e.mean(),
                    e[evals['positive'] == 1].mean(),
                    e[evals['positive'] == 0].mean(),
                )
            )
        return result

    return ScoreSummary(
        summarize_evaluations(dynamic_evaluations),
        compute_mean_ndcg(df, test_df, assessment_function),
        # summarize_evaluations(static_evaluations),
        # compute_mean_ndcg(train_df, test_df, assessment_function),
    )
