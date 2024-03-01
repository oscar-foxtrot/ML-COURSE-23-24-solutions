import os

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesRegressor

import numpy as np


def move(matrix, right, up):
    # matrix = potentials_matrix.copy()
    # matrix = potentials_matrix
    for i in range(matrix.shape[0]):
        if right > 0:
            matrix[i][right:] = matrix[i][:-right]
            matrix[i][:right] = 0
        elif right < 0:
            matrix[i][:right] = matrix[i][-right:]
            matrix[i][right:] = 0

    matrix = matrix.T
    new_right = -up
    for i in range(matrix.shape[0]):
        if new_right > 0:
            matrix[i][new_right:] = matrix[i][:-new_right]
            matrix[i][:new_right] = 0
        elif new_right < 0:
            matrix[i][:new_right] = matrix[i][-new_right:]
            matrix[i][new_right:] = 0

    matrix = matrix.T
    return matrix


def get_center(matrix):
    center = np.array([0, 0], dtype=np.float32)
    mass = np.sum(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            vect = np.array([i, j], dtype=np.float32)
            center += 1 / mass * vect * matrix[i][j]
    return center


def get_centerpositioned(matrix, center):
    # matrix = potentials_matrix.copy()
    # matrix = potentials_matrix
    matr_center = [(matrix.shape[0] - 1) / 2, (matrix.shape[1] - 1) / 2]
    matrix = move(matrix, int(np.floor(matr_center[1] - center[1])), -int(np.floor(matr_center[0] - center[0])))
    return matrix


def is_sharp(matrix):
    for i in range(matrix.shape[0] - 1):
        for j in range(matrix.shape[1] - 1):
            if np.abs(matrix[i][j] - matrix[i + 1][j]) >= 5:
                return 1
            if np.abs(matrix[i][j] - matrix[i][j + 1]) >= 5:
                return 1
    return 0


def max_mass_ratio(matrix):
    val1 = np.max(np.sum(matrix, axis=1))
    val2 = np.max(np.sum(matrix, axis=0))
    if val1 != 0 and val2 != 0:
        if val1 / val2 <= 1:
            return val1 / val2
        else:
            return val2 / val1
    else:
        return 0


def mass_ratio(matrix):
    val1 = np.max(np.sum(matrix, axis=1))
    val2 = np.max(np.sum(matrix, axis=0))
    if val1 != 0 and val2 != 0:
        return val1 / val2
    else:
        return 0


def get_entropy(matrix):
    epsilon = 0.01
    entropy = 0
    mass = np.sum(matrix) + matrix.shape[0] * matrix.shape[1] * epsilon
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            entropy -= (matrix[i][j] + epsilon) / mass * np.log((matrix[i][j] + epsilon) / mass)
    return entropy


class PotentialTransformer:
    """
    A potential transformer.

    This class is used to convert the potential's 2d matrix to 1d vector of features.
    """

    def fit(self, x, y):
        """
        Build the transformer on the training set.
        :param x: list of potential's 2d matrices
        :param y: target values (can be ignored)
        :return: trained transformer
        """
        return self

    def fit_transform(self, x, y):
        """
        Build the transformer on the training set and return the transformed dataset (1d vectors).
        :param x: list of potential's 2d matrices
        :param y: target values (can be ignored)
        :return: transformed potentials (list of 1d vectors)
        """
        return self.transform(x)

    def transform(self, potentials):
        """
        Transform the list of potential's 2d matrices with the trained transformer.
        :param x: list of potential's 2d matrices
        :return: transformed potentials (list of 1d vectors)
        """
        x = potentials
        np.subtract(potentials, 20, out=potentials)
        np.abs(potentials, out=potentials)

        oldshape = potentials.shape

        centers = []
        for i in range(potentials.shape[0]):
            centerpos = get_center(potentials[i])
            potentials[i] = get_centerpositioned(potentials[i], centerpos)
            centers.append(centerpos)

        centers = np.array(centers)

        potentials = potentials.reshape((potentials.shape[0], -1))
        potentials = potentials.astype(np.float32, copy=False)

        masses = np.sum(potentials, axis=1, dtype=np.float32)
        negmasses = 20 * oldshape[1] * oldshape[2] - masses
        # GOOD GOOD
        potentials = np.concatenate((
                potentials,
                masses[:, None],  # Added MASS analogues
                negmasses[:, None],  # Added MASS analogues
        ), axis=1, dtype=np.float32)

        return potentials


def load_dataset(data_dir):
    """
    Read potential dataset.

    This function reads dataset stored in the folder and returns three lists
    :param data_dir: the path to the potential dataset
    :return:
    files -- the list of file names
    np.array(X) -- the list of potential matrices (in the same order as in files)
    np.array(Y) -- the list of target value (in the same order as in files)
    """
    files, X, Y = [], [], []
    for file in sorted(os.listdir(data_dir)):
        potential = np.load(os.path.join(data_dir, file))
        files.append(file)
        X.append(potential["data"])
        Y.append(potential["target"])
    return files, np.array(X), np.array(Y)


def train_model_and_predict(train_dir, test_dir):
    _, X_train, Y_train = load_dataset(train_dir)
    test_files, X_test, _ = load_dataset(test_dir)
    # it's suggested to modify only the following line of this function
    regressor = make_pipeline(
        PotentialTransformer(),
        MinMaxScaler(copy=False),
        ExtraTreesRegressor(n_estimators=50, max_depth=10, n_jobs=-1, criterion='poisson')
    )
    regressor.fit(X_train, Y_train)
    predictions = regressor.predict(X_test)
    return {file: value for file, value in zip(test_files, predictions)}
