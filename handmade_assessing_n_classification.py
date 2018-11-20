import numpy as np

comp_val = lambda a, b: 1 if a == b else 0


def euclidean_distance(u, x):
    dist = 0

    for i in range(len(x)):
        dist += (u[i] - x[i]) ** 2

    dist = dist ** 0.5

    return dist

# def get_omega(j, X):
#     omega = np.zeros(len(X))
#     distance = {}
#
#     for i in range(len(X)):
#         distance[euclidean_distance(X[j], X[i])] = i
#
#     distance = sorted(distance)
#
#     k = 1
#     for key in distance:
#         omega[distance[key]] = comp_val(k, 3)
#
#     return omega


# def get_gamma(X, Y, l):
#     gamma = 0
#     m_gamma = 0
#
#     y_u = Y[j]
#
#     for i in range(len(Y)):
#         is_equal = comp_val(y_u, Y[i])
#
#         omega = get_omega(i, X)
#         gamma = gamma + is_equal*omega
#
#         if i != l:
#             m_gamma = m_gamma + is_equal*omega
#
#     return gamma - m_gamma


def get_distances(X):
    D = []
    for i in range(1, len(X)):
        row = []
        for j in range(i - 1):
            dist = euclidean_distance(X[i], X[j])
            row.append(dist)
        D.append(row)

    return D

def main():
    u = [[1, 2], [3, 4]]
    x = [3, 3]

    print(u[0][0])


if __name__ == '__main__':
    main()