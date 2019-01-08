# -----------------------------------------------------------------
#                        Machine Learning: LAB 1
#                       ElijahOzhmegov@gmail.com
#                             January 2019
#                         assessment function
# -----------------------------------------------------------------
import numpy as np
import math


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


def get_rid_of_duplicates(x, y):
    new_x = []
    new_y = []

    bad_indexes = []

    for i in range(len(x)):

        for j in range(i + 1, len(x)):
            if distance(x[i], x[j]) == 0 and y[i] != y[j]:
                bad_indexes.append(i)
                bad_indexes.append(j)
                break

        if i not in bad_indexes:
            new_x.append(x[i])
            new_y.append(y[i])

    return np.asarray(new_x), np.asarray(new_y)


def get_accuracy(method, x_test, y_test):
    accuracy = 0
    n = len(y_test)

    y_new = method.predict(x_test)

    for i in range(n):
        if y_new[i] != y_test[i]:
            accuracy += 1.0

    return 1 - accuracy/n


if __name__ == '__main__':
    x = [[0, 0], [1, 1], [1, 2], [1, 1], [3, 1]]
    y = [j for j in range(len(x))]

    new_x, new_y = get_rid_of_duplicates(x, y)

    print(new_x, '\n', new_y)
