import sys
import numpy as np


class Particle:

    __slots__ = ['x', 'v', 'dims', 'best_point', 'best_solution']

    def __init__(self, n: int, dims: int):
        """

        ensure particle starts with random velocity
        at random position.

        :param n:
        :param dims:
        """
        self.x = 2 * n * np.random.ranf(dims) - n
        self.v = np.random.random(dims)*n + 1
        self.best_solution = sys.maxsize
        self.best_point = np.zeros(dims)
        self.dims = dims
