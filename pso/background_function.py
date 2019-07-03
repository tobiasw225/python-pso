# __filename__: main.py
#
# __description__: Functions used to set background (PSO, but could be
#  also other optimizer.
#
# __remark__:
#
# __todos__:
#
# Created by Tobias Wenzel in August 2017
# Copyright (c) 2017 Tobias Wenzel

import numpy as np


def get_xsqr(n=100):
    """
    ...used for background image
    :param n:
    :return:
    """
    print("background-function: x^2")
    m_size = 2*n
    bg_fn = np.zeros((m_size,m_size))
    vals = np.linspace(-n,n,m_size)
    for i in range(m_size):
        for j in range(m_size):
            bg_fn[i,j]= vals[i]**2+ vals[j]**2
    return vals, bg_fn


def get_rastrigin(n=6):
    """
    -5.2, 5.2
    :param n:
    :return:
    """

    print("background-function: rastrigin")
    m_size = 2*n
    bg_fn = np.zeros((m_size,m_size))
    vals = np.linspace(-n,n,m_size)
    for i in range(0,2*n):
        for j in range(0, 2*n):
            bg_fn[i,j]= (vals[i]**2-10*np.cos(np.pi*vals[i])+10)\
        +(vals[j]**2-10*np.cos(np.pi*vals[j])+10)
    return vals, bg_fn


def get_rosenbrock(n: int = 10):
    """

    :param n:
    :return:
    """
    print("background-function: rosenbrock")
    m_size = 2*n
    bg_fn = np.zeros((m_size,m_size))
    vals = np.linspace(-n,n,m_size)
    a = 1
    b = 100
    for i in range(m_size):
        for j in range(m_size):
            a = 1. - vals[i]
            b = vals[j] - vals[i]*vals[j]
            bg_fn[i,j]= a*a + b*b*100
    return vals, bg_fn


def get_griewank(n: int = 10):
    """
    @working (?)
    :param n:
    :return:
    """
    m_size = 2*n
    bg_fn = np.zeros((m_size,m_size))
    vals = np.linspace(-n,n,m_size)
    for i in range(m_size):
        for j in range(m_size):
            # das ist irgendwie schäbig.
            prod = (np.cos(vals[i] / 0.0000001) + 1) * (np.cos(vals[j] / 1) + 1)
            bg_fn[i,j] = (1 / 4000) * (vals[i] ** 2 - prod + vals[j] ** 2 - prod)
    return vals, bg_fn


def get_schaffner_f6(n: int = 10):
    """
    @working
    :param n:
    :return:
    """
    m_size = 2*n
    bg_fn = np.zeros((m_size,m_size))
    vals = np.linspace(-n,n,m_size)
    for i in range(m_size):
        for j in range(m_size):
            bg_fn[i,j] =(0.5 - ((np.sin(np.sqrt(vals[i]**2 + vals[j]**2))**2 - 0.5) \
                    / (1 + 0.001*(vals[i]**2 + vals[j]**2)**2)))
    return vals, bg_fn


def get_eggholder(n: int=512):
    """
    @working
    -512, 512
    :param n:
    :return vals
    :return bg_fn:
    """
    m_size = 2*n
    bg_fn = np.zeros((m_size,m_size))
    vals = np.linspace(-n, n, m_size)
    for i in range(m_size):
        for j in range(m_size):
            bg_fn[i,j] -= (vals[1] + 47)*np.sin(np.sqrt(np.abs(vals[i] / 2 + (vals[j] + 47))))\
                         - vals[i] * np.sin(np.sqrt(np.abs(vals[i] - (vals[i] + 47))))
    return vals, bg_fn


def get_hoelder_table(n: int = 10):
    """
    @working
    :return vals
    :return bg_fn:
    """
    m_size = 2*n
    bg_fn = np.zeros((m_size,m_size))
    vals = np.linspace(-n, n, m_size)
    for i in range(m_size):
        for j in range(m_size):
            bg_fn[i,j] = np.abs(np.sin(vals[i])*np.cos(vals[j])\
                                * np.exp(np.abs(1-(np.sqrt(vals[i]**2+vals[j]**2)/np.pi))))
    return vals, bg_fn


def get_styblinsky_tang(n: int = 5):
    """
    @not yet working
    :param n:
    :return:
    """
    m_size = 2*n
    bg_fn = np.zeros((m_size,m_size))
    vals = np.linspace(-n, n, m_size)
    for i in range(m_size):
        for j in range(m_size):
            sum = 0.0
            for z in range(2):
                sum += (vals[z] ** 4 - 16 * vals[z] + 5 * vals[z])
            bg_fn[i, j] = sum/2
    return vals, bg_fn

background_function = {}
background_function['square'] = get_xsqr
background_function['rastrigin'] = get_rastrigin
background_function['schaffer_f6'] = get_schaffner_f6
background_function['griewank'] = get_griewank
background_function['rosenbrock'] = get_rosenbrock
background_function['eggholder'] = get_eggholder
background_function['hoelder_table'] = get_hoelder_table
background_function['styblinsky_tang'] = get_styblinsky_tang
