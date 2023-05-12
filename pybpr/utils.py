"""Utilities"""
# pylint: disable=invalid-name

import os
from typing import Callable, List
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import scipy.sparse as ss


def compute_mse(y_true, y_pred):
    """ignore zero terms prior to comparing the mse"""
    mask = np.nonzero(y_true)
    assert mask[0].shape[0] > 0, 'Truth matrix empty'
    mse = mean_squared_error(np.array(y_true[mask]).ravel(), y_pred[mask])
    return mse


def load_movielens_file(flag, filename, names, sep='\t'):
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
        'data',
    )
    file_path = os.path.join(data_path, flag, filename)
    return pd.read_csv(file_path, sep=sep, names=names, encoding='iso_8859_1')


def load_movielens_data(flag='ml-100k'):
    """Function to read movielens data"""
    filename = {
        'ml-100k': 'u.data',
        'ml-1m': 'ratings.dat',
        'ml-25m': 'ratings.csv',
        'ml-10M100K': 'ratings.dat',
    }[flag]
    return load_movielens_file(
        flag,
        filename,
        [
            'user_id',
            'item_id',
            'rating',
            'timestamp',
        ],
    )


def load_movielens_items(flag='ml-100k'):
    """Function to read movielens items"""

    filename = {
        'ml-100k': 'u.item',
        'ml-25m': 'movies.csv',
    }[flag]
    return load_movielens_file(
        flag,
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
        sep ='|',
    )


def compute_ndcg(ranked_item_idx: List[int], K: int, wgt_fun: Callable = np.log2):
    """Comutes NDCG metric"""
    assert K > 0, 'Should have atleast one recomendation, choose K >1'
    ndcg_score = 0.0
    if np.array(ranked_item_idx).size > 0:
        assert np.max(ranked_item_idx) < K, 'entry in ranked_item_idx > K-1!'
        Rup = np.zeros(K, dtype=int)
        Rup[ranked_item_idx] = 1.0
        wgt = np.array([1 / wgt_fun(ix + 1) for ix in np.arange(1, K + 1)])
        ndcg_score = np.sum(np.multiply(wgt, Rup)) / np.sum(wgt)
    return ndcg_score


def get_interaction_weights(train_mat, strategy: str = 'same'):
    """Function for getting the weights"""
    row_inds, col_inds, _ = ss.find(train_mat)
    num_users, num_items = train_mat.shape
    match strategy.lower():
        case 'uniform':
            weight_mat = np.random.uniform(size=train_mat.shape)
            weight_mat[row_inds, col_inds] = 1.0
        case 'user-oriented':
            weight_mat = train_mat.sum(axis=1) / num_items
            weight_mat = np.array(np.repeat(weight_mat, num_items, axis=1))
            weight_mat[row_inds, col_inds] = 1.0
        case 'item-oriented':
            weight_mat = 1.0 - train_mat.sum(axis=0) / num_users
            weight_mat = np.array(np.repeat(weight_mat, num_users, axis=0))
            weight_mat[row_inds, col_inds] = 1.0
        case _:
            weight_mat = np.ones(train_mat.shape)
    return weight_mat
