import numpy as np
from scipy.optimize import leastsq


def func(params, x):
    a, b, c = params
    return a * x * x + b * x + c


def error(params, x, y):
    return func(params, x) - y


def slovePara(X, Y):
    p0 = np.array([1, 1, 1])
    Para = leastsq(error, p0, args=(X, Y))
    return Para


def get_coef(nx_list):
    X = np.array(nx_list)
    nx_max = float(np.max(X))
    Y = np.array([1.0 - get_reg(nx, nx_max) for nx in nx_list])
    Para = slovePara(X, Y)
    a, b, c = Para[0]
    return a, b, c


def get_reg(nx, nx_max):
    # return (- nx + nx_max) / nx_max
    return (3 * nx_max - 0.9 * nx) / (3 * nx_max)
    # return (1 + nx) / (1 + 4 * nx)
    # return 1 / (nx ** 2 + 1e-5)


if __name__ == '__main__':
    print(get_coef([6, 7, 8]))
