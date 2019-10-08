# __filename__: background_function.py
#
# __description__: Functions used to set background
#
# __remark__:
#
# __todos__:
#
# Created by Tobias Wenzel in August 2017
# Copyright (c) 2017 Tobias Wenzel

import numpy as np


def get_xsqr(x: np.ndarray,
             y: np.ndarray):
    """

    :param x:
    :param y:
    :return:
    """
    m_size = len(x)
    x = x ** 2
    for i in range(m_size):
        y[i, :] = x[i] + x
    return y


def get_rastrigin(x: np.ndarray,
                  y: np.ndarray):
    """
        Recommended range: -5.2, 5.2

    -5.2, 5.2
    :param x:
    :param y:
    :return:
    """
    m_size = len(x)
    for i in range(m_size):
        for j in range(m_size):
            y[i, j] = (x[i] ** 2 - 10 * np.cos(np.pi * x[i]) + 10) \
                      + (x[j] ** 2 - 10 * np.cos(np.pi * x[j]) + 10)
    return y




def get_rosenbrock(x: np.ndarray,
                   y: np.ndarray):
    """
            Recommended range: -10, 10
    :param x:
    :param y:
    :return:
    """
    m_size = len(x)
    for i in range(m_size):
        for j in range(m_size):
            a = 1. - x[i]
            b = x[j] - x[i] ** 2
            y[i, j] = a * a + b * b * 100.
    return y


def get_griewank(x: np.ndarray,
                 y: np.ndarray):
    """
                Recommended range: -10, 10
    :param x:
    :param y:
    :return:
    """
    m_size = len(x)
    for i in range(m_size):
        for j in range(m_size):
            prod = (np.cos(x[i] / 0.0000001) + 1) * (np.cos(x[j] / 1) + 1)
            y[i, j] = (1 / 4000) * (x[i] ** 2 - prod + x[j] ** 2 - prod)
    return y


def get_schaffer_f6(x: np.ndarray,
                    y: np.ndarray):
    """
            Recommended range: -10, 10
    :param x:
    :param y:
    :return:
    """
    m_size = len(x)
    for i in range(m_size):
        for j in range(m_size):
            y[i, j] = (0.5 - ((np.sin(np.sqrt(x[i] ** 2 + x[j] ** 2)) ** 2 - 0.5) \
                             / (1 + 0.001 * (x[i] ** 2 + x[j] ** 2) ** 2)))
    return y


def get_eggholder(x: np.ndarray,
                  y: np.ndarray):
    """
        Recommended range: -512, 512

    :return x
    :return y:
    """
    m_size = len(x)
    for i_1 in range(m_size):
        for i_2 in range(m_size):
            y[i_1, i_2] -= (x[i_2] + 47) \
                           * np.sin(np.sqrt(np.abs((x[i_1] / 2) + x[i_2] + 47))) \
                           - x[i_1] \
                           * np.sin(np.sqrt(np.abs(x[i_1] - (x[i_2] + 47))))
    return y


def get_hoelder_table(x: np.ndarray,
                      y: np.ndarray):
    """
    :return x
    :return y:
    """
    m_size = len(x)
    for i_1 in range(m_size):
        for i_2 in range(m_size):
            y[i_1, i_2] = -np.abs(np.sin(x[i_1]) * np.cos(x[i_2]) \
                                  * np.exp(np.abs(1 - (np.sqrt(x[i_1] ** 2 + x[i_2] ** 2) / np.pi))))
    return y


def get_styblinsky_tang(x: np.ndarray,
                        y: np.ndarray):
    """
    @not yet working (?)
    :param x:
    :param y:
    :return:
    """
    m_size = len(x)
    f = lambda d: (d ** 4) - (16 * (d**2)) + (5 * d)
    f_arr = np.frompyfunc(f, 1, 1)
    for i_1 in range(m_size):
        y[i_1, :] = f_arr(x[i_1])
        for i_2 in range(m_size):
            y[i_1, i_2] += f(x[i_2])
    return y


background_function = {}

background_function['square'] = get_xsqr
background_function['rastrigin'] = get_rastrigin
background_function['schaffer_f6'] = get_schaffer_f6
background_function['griewank'] = get_griewank
background_function['rosenbrock'] = get_rosenbrock
background_function['eggholder'] = get_eggholder
background_function['hoelder_table'] = get_hoelder_table
background_function['styblinsky_tang'] = get_styblinsky_tang


def generate_2d_background(func_name: str, n: int):
    """

    :param func_name:
    :param n:
    :return:
    """
    m_size = 2*n
    bg_fn = np.zeros((m_size, m_size))
    vals = np.linspace(-n, n, m_size)
    print(f"background-function: {func_name}")
    return background_function[func_name](vals, bg_fn)