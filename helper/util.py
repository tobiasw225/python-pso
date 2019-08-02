import numpy as np


def singleton(c):
    """
        very simple singleton

    :param c:
    :return:
    """
    instance = None

    def init(*args, **kwargs):
        nonlocal instance
        if not instance:
            instance = c(*args, **kwargs)
        return instance
    return init


def is_orthonomal(A):
    return np.allclose(A @ A.T, np.eye(A.shape[0]))