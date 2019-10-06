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


def eval_hoelder_table(x: np.ndarray) -> np.float:
    assert len(x) == 2
    return np.abs(np.sin(x[0]) * np.cos(x[1])\
                  * np.exp(np.abs(1 - (np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi))))

def eval_griewank(x: np.ndarray) -> np.float:
    assert len(x) == 2
    prod = (np.cos(x[0] / 0.0000001) + 1) * (np.cos(x[1] / 1) + 1)
    return (1 / 4000) * (x[0] ** 2 - prod + x[1] ** 2 - prod)

def eval_schaffer_f6(x: np.ndarray) -> np.float:
    assert len(x) == 2
    return (0.5 - ((np.sin(np.sqrt(x[0] ** 2 + x[1] ** 2)) ** 2 - 0.5)
                   / (1 + 0.001 * (x[0] ** 2 + x[1] ** 2) ** 2)))


eval_function = dict()
eval_function['square'] = eval_square
eval_function['rastrigin'] = eval_rastrigin
eval_function['schaffer_f6'] = eval_schaffer_f6
eval_function['griewank'] = eval_griewank
eval_function['rosenbrock'] = eval_rosenbrock
eval_function['eggholder'] = eval_eggholder
eval_function['hoelder_table'] = eval_hoelder_table
eval_function['eval_styblinsky_tang'] = eval_styblinsky_tang