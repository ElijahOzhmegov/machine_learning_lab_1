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
    if radius > 1:
        return 0
    return 1 - math.sin(radius * math.pi / 2)



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
    indexed_distances.sort()
    return indexed_distances


# it allows you to get k omega for desired Xi
def get_k_omega(Di, r_Di, k):
    omega = []

    limit_rho = r_Di[k+1][0]
    
    for i in range(k):
        rho_i = r_Di[i][0]
        tmp_omega = calc_with_kernel(rho_i / limit_rho)
        omega.append(tmp_omega)  # suffer, bitch!

    return omega


def get_Gamma(Y, y_u, Di, r_Di, k):
    gamma = 0
    omega = get_k_omega(Di, r_Di, k)

    for i in range(k):
        index = r_Di[i][1]
        gamma += comp_val(Y(index), y_u)*omega[i]


def main():
    u = [[1, 2], [3, 4]]
    x = [3, 3]

    print(calc_with_kernel(1.5))


if __name__ == '__main__':
    main()