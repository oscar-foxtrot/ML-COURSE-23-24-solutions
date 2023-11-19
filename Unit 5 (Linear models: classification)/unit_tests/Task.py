import numpy as np


class Preprocessor:

    def __init__(self):
        self.data = None

    def fit(self, X, Y=None):
        categories = [np.sort(X[i].unique()) for i in X]
        self.categories = [{categories[j][i]: i
                           for i in range(len(categories[j]))}
                           for j in range(len(categories))]

    def transform(self, X):
        pass

    def fit_transform(self, X, Y=None):
        pass


class MyOneHotEncoder(Preprocessor):

    def __init__(self, dtype=np.float64):
        super(Preprocessor).__init__()
        self.dtype = dtype

    def fit(self, X, Y=None):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: unused
        """
        super().fit(X, Y)

    def transform(self, X):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        returns: transformed objects, numpy-array, shape [n_objects, |f1| + |f2| + ...]
        """
        for i in range(X.shape[1]):
            old_cats = X.iloc[:, i].copy(deep=True)
            X.iloc[:, i] = X.iloc[:, i].astype(object)
            nzeroes = len(self.categories[i])
            for j in range(X.shape[0]):
                X.iloc[:, i][j] = np.zeros(nzeroes)
                X.iloc[:, i][j][self.categories[i][old_cats[j]]] = 1
        return np.array([np.concatenate([X.iloc[i][j]
                         for j in range(X.shape[1])])
                         for i in range(X.shape[0])])

    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


class SimpleCounterEncoder:

    def __init__(self, dtype=np.float64):
        self.dtype = dtype
        self.counters = []  # list of dicts {feature value: counters}
        # Counters: [successes, counters]

    def fit(self, X, Y):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        """
        for col in X.columns:
            uniques = X[col].unique()
            to_add = {i: np.array([0, 0, 0], dtype=np.float64) for i in uniques}
            for j in range(len(X[col])):
                to_add[X[col].iloc[j]][1] += 1
                to_add[X[col].iloc[j]][0] += Y.iloc[j]
            for elem in uniques:
                to_add[elem][0] /= to_add[elem][1]
                to_add[elem][1] /= X.shape[0]
            self.counters.append(to_add)

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3 * n_features]
        """
        for col in range(len(X.columns)):
            old_cats = X.iloc[:, col].copy(deep=True)
            X.iloc[:, col] = X.iloc[:, col].astype(object)
            for j in range(len(old_cats)):
                X.iat[j, col] = self.counters[col][old_cats.iat[j]].copy()
                X.iat[j, col][2] = \
                    (X.iat[j, col][0] + a) / (X.iat[j, col][1] + b)
        return np.array([np.concatenate([X.iat[i, j]
                         for j in range(X.shape[1])])
                         for i in range(X.shape[0])])

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


def group_k_fold(size, n_splits=3, seed=1):
    idx = np.arange(size)
    np.random.seed(seed)
    idx = np.random.permutation(idx)
    n_ = size // n_splits
    for i in range(n_splits - 1):
        yield idx[i * n_: (i + 1) * n_], np.hstack((idx[:i * n_], idx[(i + 1) * n_:]))
    yield idx[(n_splits - 1) * n_:], idx[:(n_splits - 1) * n_]


class FoldCounters:

    def __init__(self, n_folds=3, dtype=np.float64):
        self.dtype = dtype
        self.n_folds = n_folds
        self.groups = []  # groups
        self.fold_encoders = []  # counters corresponding to the "smallest" group in self.groups

    def fit(self, X, Y, seed=1):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        param seed: random seed, int
        """
        for fold in group_k_fold(X.shape[0], self.n_folds, seed):
            self.groups.append(fold)
        for group in self.groups:
            enc = SimpleCounterEncoder()
            enc.fit(X.iloc[group[1]], Y.iloc[group[1]])
            self.fold_encoders.append(enc)

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3 * n_features]
        """
        print(X)
        temptrans = []
        for group in range(len(self.groups)):
            enc = self.fold_encoders[group]
            group = self.groups[group]
            trans = X.iloc[group[0]].copy(deep=True)
            enc.transform(trans, a, b)
            temptrans.append(trans)
        for group in range(len(self.groups)):
            X.iloc[self.groups[group][0]] = temptrans[group]
        return np.array([np.concatenate([X.iat[i, j]
                         for j in range(X.shape[1])])
                         for i in range(X.shape[0])])

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)


def weights(x, y):
    """
    param x: training set of one feature, numpy-array, shape [n_objects,]
    param y: target for training objects, numpy-array, shape [n_objects,]
    returns: optimal weights, numpy-array, shape [|x unique values|,]
    """

    def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
        w = np.zeros(X.shape[1]) + 0.5
        for epoch in range(epochs):
            gradient = np.zeros(X.shape[1])
            for i in range(X.shape[0]):
                xi = X[i]
                yi = y[i]
                p = np.dot(w, xi)
                gradient += (p - yi) * xi
            w -= learning_rate * gradient
        return w

    x_one_hot = np.eye(len(np.unique(x)))[x]
    return gradient_descent(x_one_hot, y)
