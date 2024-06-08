# __filename__: background_function.py
#
# __description__: Functions used to set background
#
#
# Created by Tobias Wenzel in August 2017
# Copyright (c) 2017 Tobias Wenzel

import numpy as np


def xsqr_matrix(x: np.ndarray, y: np.ndarray):
    """

    :param x:
    :param y:
    :return:
    """
    m_size = len(x)
    x = x**2
    for i in range(m_size):
        y[i, :] = x[i] + x
    return y


def rastrigin_matrix(x: np.ndarray, y: np.ndarray):
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
            y[i, j] = (x[i] ** 2 - 10 * np.cos(np.pi * x[i]) + 10) + (
                x[j] ** 2 - 10 * np.cos(np.pi * x[j]) + 10
            )
    return y


def rosenbrock_matrix(x: np.ndarray, y: np.ndarray):
    """
            Recommended range: -10, 10
    :param x:
    :param y:
    :return:
    """
    m_size = len(x)
    for i in range(m_size):
        for j in range(m_size):
            a = 1.0 - x[i]
            b = x[j] - x[i] ** 2
            y[i, j] = a * a + b * b * 100.0
    return y


def griewank_matrix(x: np.ndarray, y: np.ndarray):
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


def schaffer_f6_matrix(x: np.ndarray, y: np.ndarray):
    """
            Recommended range: -10, 10
    :param x:
    :param y:
    :return:
    """
    m_size = len(x)
    for i in range(m_size):
        for j in range(m_size):
            y[i, j] = 0.5 - (
                (np.sin(np.sqrt(x[i] ** 2 + x[j] ** 2)) ** 2 - 0.5)
                / (1 + 0.001 * (x[i] ** 2 + x[j] ** 2) ** 2)
            )
    return y


def eggholder_matrix(x: np.ndarray, y: np.ndarray):
    """
        Recommended range: -512, 512

    :return x
    :return y:
    """
    m_size = len(x)
    for i_1 in range(m_size):
        for i_2 in range(m_size):
            y[i_1, i_2] -= (x[i_2] + 47) * np.sin(
                np.sqrt(np.abs((x[i_1] / 2) + x[i_2] + 47))
            ) - x[i_1] * np.sin(np.sqrt(np.abs(x[i_1] - (x[i_2] + 47))))
    return y


def hoelder_table_matrix(x: np.ndarray, y: np.ndarray):
    """
    :return x
    :return y:
    """
    m_size = len(x)
    for i_1 in range(m_size):
        for i_2 in range(m_size):
            y[i_1, i_2] = -np.abs(
                np.sin(x[i_1])
                * np.cos(x[i_2])
                * np.exp(np.abs(1 - (np.sqrt(x[i_1] ** 2 + x[i_2] ** 2) / np.pi)))
            )
    return y


def styblinsky_tang_matrix(x: np.ndarray, y: np.ndarray):
    """
    @not yet working (?)
    :param x:
    :param y:
    :return:
    """

    def _func(d):
        return (d**4) - (16 * (d**2)) + (5 * d)

    m_size = len(x)
    f_arr = np.frompyfunc(_func, 1, 1)
    for i_1 in range(m_size):
        y[i_1, :] = f_arr(x[i_1])
        for i_2 in range(m_size):
            y[i_1, i_2] += _func(x[i_2])
    return y


def generate_2d_background(func_name: str, n: int):
    """

    :param func_name:
    :param n:
    :return:
    """
    m_size = 2 * n
    matrix = np.zeros((m_size, m_size))
    values = np.linspace(-n, n, m_size)
    print(f"background-function: {func_name}")
    background_function = {}
    background_function["square"] = xsqr_matrix
    background_function["rastrigin"] = rastrigin_matrix
    background_function["schaffer_f6"] = schaffer_f6_matrix
    background_function["griewank"] = griewank_matrix
    background_function["rosenbrock"] = rosenbrock_matrix
    background_function["eggholder"] = eggholder_matrix
    background_function["hoelder_table"] = hoelder_table_matrix
    background_function["styblinsky_tang"] = styblinsky_tang_matrix

    return background_function[func_name](values, matrix)
