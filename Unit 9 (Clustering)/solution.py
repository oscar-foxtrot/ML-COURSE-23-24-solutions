import numpy as np

import sklearn
import sklearn.metrics


def silhouette_score(x, labels):
    '''
    :param np.ndarray x: Непустой двумерный массив векторов-признаков
    :param np.ndarray labels: Непустой одномерный массив меток объектов
    :return float: Коэффициент силуэта для выборки x с метками labels
    '''
    if len(set(labels)) == 1:
        return 0
    labl_sort = np.argsort(labels)
    x = x[labl_sort]
    labels = labels[labl_sort]

    label_encoder = sklearn.preprocessing.LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    dists = sklearn.metrics.pairwise_distances(x)
    sils = []
    for i in range(x.shape[0]):
        num_cluster = labels[i]
        n_in_all_clusters = np.bincount(labels)
        if n_in_all_clusters[num_cluster] == 1:
            sils.append(0)
        else:
            sums_dists = np.bincount(labels, weights=dists[i])
            ai = 1 / (n_in_all_clusters[num_cluster] - 1) * sums_dists[num_cluster]
            positions = np.arange(len(set(labels)))
            positions = np.delete(positions, num_cluster)
            means = sums_dists[positions] / n_in_all_clusters[positions]
            bi = np.min(means)
            sils.append((bi - ai) / np.max([bi, ai]) if np.max([bi, ai]) != 0 else 0)

    return np.mean(sils)


def bcubed_score(real, pred):
    eq_C = pred[:, np.newaxis] == pred
    eq_L = real[:, np.newaxis] == real
    corr = eq_C * eq_L
    prec = eq_C * corr
    prec = np.mean(np.mean(prec, axis=1, where=eq_C))
    rec = eq_L * corr
    rec = np.mean(np.mean(rec, axis=1, where=eq_L))
    return 2 * rec * prec / (prec + rec)
