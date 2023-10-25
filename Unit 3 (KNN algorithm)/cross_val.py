import numpy as np
import typing
from collections import defaultdict


def kfold_split(num_objects: int,
                num_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split [0, 1, ..., num_objects - 1] into equal num_folds folds
       (last fold can be longer) and returns num_folds train-val
       pairs of indexes.

    Parameters:
    num_objects: number of objects in train set
    num_folds: number of folds for cross-validation split

    Returns:
    list of length num_folds, where i-th element of list
    contains tuple of 2 numpy arrays, he 1st numpy array
    contains all indexes without i-th fold while the 2nd
    one contains i-th fold
    """
    folds = []
    all_indices = np.arange(num_objects)
    fold_size = num_objects // num_folds  # Expecting 1 or more folds
    for i in range(0, num_folds - 1):
        train_indices = np.concatenate((all_indices[:i * fold_size],
                                       all_indices[(i + 1) * fold_size:]))
        validation_indices = all_indices[i * fold_size: (i + 1) * fold_size]
        folds.append((train_indices, validation_indices))
    folds.append((all_indices[:(num_folds - 1) * fold_size],
                  all_indices[(num_folds - 1) * fold_size:]))
    return folds


def knn_cv_score(X: np.ndarray, y: np.ndarray, parameters: dict[str, list],
                 score_function: callable,
                 folds: list[tuple[np.ndarray, np.ndarray]],
                 knn_class: object) -> dict[str, float]:
    """Takes train data, counts cross-validation score over
    grid of parameters (all possible parameters combinations)

    Parameters:
    X: train set
    y: train labels
    parameters: dict with keys from
        {n_neighbors, metrics, weights, normalizers}, values of type list,
        parameters['normalizers'] contains tuples (normalizer, normalizer_name)
        see parameters example in your jupyter notebook

    score_function: function with input (y_true, y_predict)
        which outputs score metric
    folds: output of kfold_split
    knn_class: class of knn model to fit

    Returns:
    dict: key - tuple of (normalizer_name, n_neighbors, metric, weight),
    value - mean score over all folds
    """
    results = {}
    for n_neighbors in parameters['n_neighbors']:
        for metric in parameters['metrics']:
            for weight in parameters['weights']:
                for normalizer, normalizer_name in parameters['normalizers']:
                    scores = []
                    for train_indices, val_indices in folds:
                        if normalizer:
                            normalizer.fit(X[train_indices])
                            X_scaled = normalizer.transform(X[train_indices])
                        else:
                            X_scaled = X[train_indices]
                        X_train, X_val = X_scaled, X[val_indices]
                        y_train, y_val = y[train_indices], y[val_indices]

                        # normalizer.fit(X_val)
                        knn = knn_class(n_neighbors=n_neighbors, metric=metric, weights=weight)
                        knn.fit(X_train, y_train)
                        if normalizer:
                            X_val = normalizer.transform(X_val)
                        y_pred = knn.predict(X_val)

                        score = score_function(y_val, y_pred)
                        scores.append(score)

                    mean_score = np.mean(scores)
                    results[(normalizer_name, n_neighbors, metric, weight)] = mean_score

    return results
