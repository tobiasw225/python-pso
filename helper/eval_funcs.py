import numpy as np

def eval_rastrigin(row: np.ndarray):
    """
        apply rastrigin-function to row

    :param row:
    :return:
    """
    f = lambda d: d ** 2 - 10 * np.cos(np.pi * d) + 10
    f_arr = np.frompyfunc(f, 1, 1)
    return np.sum(f_arr(row))


def eval_square(row: np.ndarray):
    """

    :param row:
    :return:
    """
    f_arr = np.frompyfunc(lambda d: d ** 2, 1, 1)
    return np.sum(f_arr(row))


def eval_rosenbrock(row: np.ndarray):
    """

    :param row:
    :return:
    """
    assert len(row) == 2
    a = 1. - row[0]
    b = row[1] - row[0]*row[0]
    return a*a + b*b*100 #rosen(row)#


def eval_eggholder(row: np.ndarray):
    """

    :param row:
    :return:
    """
    assert len(row) == 2
    return -(row[1]+47)\
           * np.sin(np.sqrt(np.abs((row[0]/2) + row[1]+47)))\
           - row[0]\
           * np.sin(np.sqrt(np.abs(row[0]-(row[1]+47))))


def eval_styblinsky_tang(row: np.ndarray):
    """
        not working
    :param row:
    :return:
    """
    f = lambda d: (d**4)-(16*(d**2))+(5*d)
    f_arr = np.frompyfunc(f, 1, 1)
    return np.sum(f_arr(row))/2