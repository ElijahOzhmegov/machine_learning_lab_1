import numpy as np
import math


comp_val = lambda a, b: 1 if a == b else 0


def sum_of_squares(v):
    """ v1 * v1 + v2 * v2 ... + vn * vn"""
    # or return dot_product(v, v)
    return sum(vi ** 2 for vi in v)


def subtract_vectors(v, w):
    return [vi - wi for vi, wi in zip(v, w)]


def magnitude(v):
    return math.sqrt(sum_of_squares(v))


def distance(a, b):
    return magnitude(subtract_vectors(a, b))


def calc_with_kernel(radius):
    return math.sin(radius * math.pi / 2)


def get_duplicates(X):
    i_duplicates = []

    for i in range(len(X), 1, -1):
        for j in range(i + 1, len(X)):
            if distance(X[i], X[j]) == 0:
                i_duplicates.append(i)

    return i_duplicates


def get_distances(X):
    size = len(X)
    D = [[0 for col in range(size)] for row in range(size)]

    for i in range(size):
        for j in range(i):
            D[i][j] = distance(X[i], X[j])
            D[j][i] = D[i][j]

    return D


def ranging(D_i):
    indexed_distances = [[D_i[i], i] for i in range(len(D_i))]
    return indexed_distances.sort()


def main():
    u = [[1, 2], [3, 4]]
    x = [3, 3]

    print(calc_with_kernel(0.1))


if __name__ == '__main__':
    main()