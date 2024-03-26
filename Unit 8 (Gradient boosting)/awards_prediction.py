from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd

import numpy as np
from catboost import Pool

from catboost import CatBoostRegressor
from numpy import ndarray
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
"""
 Внимание!
 В проверяющей системе имеется проблема с catboost.
 При использовании этой библиотеки, в скрипте с решением необходимо инициализировать метод с использованием `train_dir` как показано тут:
 CatBoostRegressor(train_dir='/tmp/catboost_info')
"""


def transform(df):
    '''
    Transforms the dataframe (withotu modifying it)
    and then returns:
    Tuple: (embedding_features, cat_features, categories, df)
    '''
    df = df.copy()
    categories = ['genres', 'directors', 'filming_locations', 'keywords', 'actor_0_gender', 'actor_1_gender', 'actor_2_gender']
    unique_genres = list(df['genres'].explode().unique())
    unique_directors = list(df['directors'].explode().unique())
    unique_locations = list(df['filming_locations'].explode().unique())
    unique_keywords = list(df['keywords'].explode().unique())
    unique_genders_0 = list(df['actor_0_gender'].unique())
    unique_genders_1 = list(df['actor_1_gender'].unique())
    unique_genders_2 = list(df['actor_2_gender'].unique())

    for cat in ['genres', 'directors', 'filming_locations', 'keywords']:
        df.loc[df[cat] != 'unknown', cat] = \
            df.loc[df[cat] != 'unknown', cat].apply(lambda x: ','.join(str(e) for e in x if e is not None))

    categories = ['genres', 'directors', 'filming_locations', 'keywords', 'actor_0_gender', 'actor_1_gender', 'actor_2_gender']

    embedding_features = ['genres', 'directors', 'filming_locations', 'keywords']
    cat_features = ['actor_0_gender', 'actor_1_gender', 'actor_2_gender']
    df[cat_features] = df[cat_features].astype('category')

    embedding_features = ([df.columns.get_loc(col) for col in embedding_features])
    cat_features = ([df.columns.get_loc(col) for col in cat_features])
    categories = ([df.columns.get_loc(col) for col in categories])
    return (embedding_features, cat_features, categories, df)


def train_model_and_predict(train_file: str, test_file: str) -> ndarray:
    """
    This function reads dataset stored in the folder, trains predictor and returns predictions.
    :param train_file: the path to the training dataset
    :param test_file: the path to the testing dataset
    :return: predictions for the test file in the order of the file lines (ndarray of shape (n_samples,))
    """

    df_train = pd.read_json(train_file, lines=True)
    df_test = pd.read_json(test_file, lines=True)

    y_train = df_train["awards"]
    del df_train["awards"]

    (embedding_features, cat_features, categories, df_train_transformed) = transform(df_train)
    (_, _, _, df_test_transformed) = transform(df_test)

    model1 = CatBoostRegressor(
        n_estimators=200,
        cat_features=categories,
        eval_metric='MAE',
        random_seed=42,
    )
    model1.fit(df_train_transformed.to_numpy(), y_train.to_numpy())

    return model1.predict(df_test_transformed.to_numpy())
