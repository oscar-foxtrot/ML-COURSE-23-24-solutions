import numpy as np
import typing


class MinMaxScaler:
    def __init__(self):
        self.mins = 0
        self.shrink_vals = 1

    def fit(self, data: np.ndarray) -> None:
        """Store calculated statistics

        Parameters:
        data: train set, size (num_obj, num_features)
        """
        self.mins = data.min(axis=0)
        self.shrink_vals = data.max(axis=0) - data.min(axis=0)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Parameters:
        data: train set, size (num_obj, num_features)

        Return:
        scaled data, size (num_obj, num_features)
        """
        data_copy = data.copy()
        data_copy -= self.mins
        data_copy /= self.shrink_vals
        return data_copy


class StandardScaler:
    def __init__(self) -> None:
        self.means = 0
        self.stds = 1

    def fit(self, data: np.ndarray) -> None:
        """Store calculated statistics

        Parameters:
        data: train set, size (num_obj, num_features)
        """
        self.means = data.mean(axis=0)
        self.stds = data.std(axis=0)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Parameters:
        data: train set, size (num_obj, num_features)

        Return:
        scaled data, size (num_obj, num_features)
        """
        data_copy = data.copy()
        data_copy -= self.means
        data_copy /= self.stds
        return data_copy
