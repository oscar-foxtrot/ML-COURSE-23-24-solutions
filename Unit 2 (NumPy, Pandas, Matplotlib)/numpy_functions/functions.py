from typing import List


def sum_non_neg_diag(X: List[List[int]]) -> int:
    """
    Вернуть  сумму неотрицательных элементов на диагонали прямоугольной матрицы X. 
    Если неотрицательных элементов на диагонали нет, то вернуть -1
    """ 
    res = 0
    flagpos = False
    for i in range(min(len(X), len(X[0]))):
        if X[i][i] >= 0:
            res += X[i][i]
            flagpos = True
    if not flagpos:
        return -1
    else:
        return res

def are_multisets_equal(x: List[int], y: List[int]) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    a = dict()
    b = dict()
    for i in range(len(x)):
        if x[i] in a.keys():
            a[x[i]] += 1
        else:
            a[x[i]] = 0
            
    for i in range(len(y)):
        if y[i] in b.keys():
            b[y[i]] += 1
        else:
            b[y[i]] = 0
            
    return a == b


def max_prod_mod_3(x: List[int]) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    res = list()
    for i in range(len(x) - 1):
        if (x[i] % 3 == 0) or (x[i+1] % 3 == 0):
            res.append(x[i] * x[i+1])
        
    if not res:
        return -1
    else:
        return max(res)
    


def convert_image(image: List[List[List[float]]], weights: List[float]) -> List[List[float]]:
    """
    Сложить каналы изображения с указанными весами.
    """
    res = image.copy()
    for i in range(len(res)):
        for j in range(len(res[i])):
            for k in range(len(res[i][j])):
                res[i][j][k] *= weights[k]
    
    res2 = res
    for i in range(len(res)):
        for j in range(len(res[i])):
            res2[i][j] = sum(res[i][j])
    return res2


def rle_scalar(x: List[List[int]], y:  List[List[int]]) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    newx = list()
    newy = list()
    
    for i in range(len(x)):
        newx += [x[i][0]]*x[i][1]
        
    for i in range(len(y)):
        newy += [y[i][0]]*y[i][1]
        
    if len(newx) != len(newy):
        return -1
    else:
        res = list()
        for i in range(len(newx)):
            res.append(newx[i] * newy[i])
    return sum(res)
    

def cosine_distance(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y. 
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    res = [[0] * len(Y) for _ in range(len(X))]
    for i in range(len(X)):
        for j in range(len(Y)):
            normx = 0
            normy = 0
            for d in range(len(X[0])):
                normx += X[i][d]**2
                normy += Y[j][d]**2
            normx = normx**0.5
            normy = normy**0.5
            if normx == 0 or normy == 0:
                res[i][j] = 1
            else:
                for d in range(len(X[0])):
                    res[i][j] += X[i][d] * Y[j][d]
                res[i][j] /= normy
                res[i][j] /= normx
    return res