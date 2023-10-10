import numpy as np
from typing import Tuple


def sum_non_neg_diag(X: np.ndarray) -> int:
    """
    Вернуть  сумму неотрицательных элементов на диагонали прямоугольной матрицы X. 
    Если неотрицательных элементов на диагонали нет, то вернуть -1
    """
    if np.all((np.diag(X) < 0) == True):
        return -1
    else:
        return np.sum(np.diag(X)[np.diag(X) >= 0])


def are_multisets_equal(x: np.ndarray, y: np.ndarray) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    return np.array_equal(np.sort(x), np.sort(y))


def max_prod_mod_3(x: np.ndarray) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    res = (x[1:] * x[:-1])[(x[1:] % 3 == 0) | (x[:-1] % 3 == 0)]
    if res.size == 0:
        return -1
    else:
        return res.max()


def convert_image(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Сложить каналы изображения с указанными весами.
    """
    channels_weighted = image * weights
    return np.sum(image * weights, axis=2)


def rle_scalar(x: np.ndarray, y: np.ndarray) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    if x[:, 1].sum() != y[:, 1].sum():
        return -1
    newx = np.repeat(x[:, 0], x[:, 1])
    newy = np.repeat(y[:, 0], y[:, 1])
    result = np.dot(newx, newy)
    return result


def cosine_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y.
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    norm_X = np.linalg.norm(X, axis=1, keepdims=True)
    norm_Y = np.linalg.norm(Y, axis=1, keepdims=True)
    norms = norm_X.dot(norm_Y.T)
    res = np.dot(X, Y.T)
    
    cosine_distance = np.where(norms == 0, 1, res / norms)
    
    return cosine_distance