# __filename__: pso_np.py
#
# __description__: other variations of pso.
#
# __remark__:
#
# __todos__:
#
# Created by Tobias Wenzel in Summer 2019
# Copyright (c) 2019 Tobias Wenzel


import numpy as np
import sys
from helper.eval_funcs import eval_function


def rastrigin_bg(n, offset) -> np.ndarray:
    """

    :param n:
    :return:
    """

    m_size = 2*n
    x = np.linspace(-n, n, m_size)
    y = np.zeros((m_size, m_size))

    for i in range(m_size):
        x_1 = x[i] + offset[0]
        part_1 = x_1 ** 2 - 10 * np.cos(np.pi * x_1) + 10
        for j in range(m_size):
            x_2 = x[j] +  offset[1]
            y[i, j] = part_1 + (x_2 ** 2 - 10 * np.cos(np.pi * x_2) + 10)
    return y


class MovingTarget:
    """
        evaluate function and moves target
    """
    def __init__(self, func: callable, dims: int, n: float):
        """

        :param func:
        :param dims:
        :param n:
        """
        self.func = func
        self.offsets = np.zeros(dims)
        self.dims = dims
        self.n = n  # [-n, n]

    def move_offset(self):
        """
            for testing purposes, this is kept purely random.
            it should be possible to pass values for a more
            targeted change.

        :return:
        """
        r_dim = np.random.randint(self.dims)
        #self.offsets[r_dim] += np.random.ranf() * ((2 * self.n) - self.n)/10
        self.offsets[0] += np.random.ranf() * ((2 * self.n) - self.n)/10

    def eval(self, x: np.ndarray) -> np.ndarray:
        """
            eval function and apply an offset.
            I'm not actually doing this.

            -> By shifting the x-values I get the
            same result. The x-values in the calling
            method are not changed.

        :param x:
        :return:
        """
        x += self.offsets
        fx = np.apply_along_axis(self.func, axis=1, arr=x)
        ri = np.random.randint(0, 11)
        #if ri <= 2:
        self.move_offset()
        return fx



def run_pso(num_particles: int,
            dims: int,
            iterations: int):
    """
        pure functional. runs algorithm until the end,
        then output of the solution.

    :param num_particles:
    :param dims:
    :param iterations:
    :return:
    """
    def chaos_for_position(j):
        nonlocal weights, x, n
        # select random dimension
        r_dim = np.random.randint(dims)
        r_particles = np.random.choice(range(num_particles), num_particles//10)
        # constant factor to keep the chaos realistic
        x[r_particles, r_dim] = np.random.ranf() * ((2 * n) - n) * weights[j] * 0.45

    def chaos_for_velocity():
        nonlocal v, max_velocity
        # don't fall asleep.
        p_vel_abs = np.sum(np.abs(v), axis=1)
        v[p_vel_abs < lowest_speed, :] += np.random.ranf(dims)
        # largest absolute velocity sum larger than max_vel
        max_vel_sum = np.sum(np.abs(max_velocity))
        if np.any(max_vel_sum < p_vel_abs):
            _idx = np.argmax(np.sum((v[ max_vel_sum < p_vel_abs , :]), axis=1))
            max_velocity = v[_idx, :]
        r_particles = np.random.choice(range(num_particles), num_particles // 10)
        vals = np.random.ranf((len(r_particles), dims)) * max_velocity
        v[r_particles, :] = vals


    lowest_speed = 0.1
    max_velocity = np.zeros(dims)
    n = 5.2
    func = rastrigin
    c1 = c2 = 1.494
    v = np.random.rand(num_particles, dims) * n + 1
    x = 2 * n * np.random.rand(num_particles, dims) - n
    assert v.shape == x.shape

    best_solution = np.full((1, num_particles), sys.maxsize)
    best_point = np.zeros_like(x)
    best_global_point = np.zeros(dims)
    best_global_solution = sys.maxsize
    weights = np.linspace(0.9, 0.4, iterations)

    for i in range(iterations):
        # apply function to all particles
        fx = np.apply_along_axis(func, axis=1, arr=x)

        # update best solutions of points
        idx = fx < best_solution
        best_solution[idx] = fx[idx[0]]
        best_point[idx[0]] = x[idx[0], :]

        # update global best solutions
        idx = np.argmin(best_solution)
        if best_solution[:, idx][0] < best_global_solution:
            best_global_solution = best_solution[:, idx][0]
            best_global_point = x[idx, :]

        # update velocity after formula given constants c1 & c2, some randomness,
        # and personal/ global best solutions.
        r = np.random.ranf(2)
        v = weights[i] * v + r[0] * c1 * (best_point - x)
        # you can start the global updating later, if you want.
        v += c2 * r[1] * (best_global_point - x)

        chaos_for_position(i)
        chaos_for_velocity()

        # update position
        x += v

        # keep all particles in range [-n, n]
        x = np.clip(x, a_min=-n, a_max=n)

        print(f"{best_global_solution}")


class DynamicPSO:
    """
        Idea: Make interruption possible &
        dynamically change the function, which is evaluated
        to make things (a lot) harder.

    """
    def __init__(self, num_particles: int,
                 dims: int,
                 iterations: int):
        """

        :param num_particles:
        :param dims:
        :param iterations:
        """
        self.num_particles = num_particles
        self.dims = dims
        self.iterations = iterations
        self.n = 5.2
        self.v = np.random.rand(self.num_particles, self.dims) * self.n + 1
        self.x = 2 * self.n * np.random.rand(self.num_particles, self.dims) - self.n
        assert self.v.shape == self.x.shape
        self.best_solution = np.full((1, self.num_particles), sys.maxsize)
        self.best_point = np.zeros_like(self.x)
        self.best_global_point = np.zeros(self.dims)
        self.best_global_solution = sys.maxsize
        self.lowest_speed = 0.1
        self.max_velocity = np.zeros(dims)
        self.weights = np.linspace(0.9, 0.4, self.iterations)

        # instead of simply evaluating the function, let it move.
        self.func = MovingTarget(dims=self.dims,
                                 func=rastrigin,
                                 n=self.n)

    def chaos_for_position(self, j: int):
        """

        :param j:
        :return:
        """
        # select random dimension
        r_dim = np.random.randint(self.dims)
        r_particles = np.random.choice(range(self.num_particles), self.num_particles//10)
        # constant factor to keep the chaos realistic
        self.x[r_particles, r_dim] = np.random.ranf() * ((2 * self.n) - self.n) * self.weights[j] * 0.45

    def chaos_for_velocity(self):
        """
            don't fall asleep.
        :return:
        """
        p_vel_abs = np.sum(np.abs(self.v), axis=1)
        self.v[p_vel_abs < self.lowest_speed, :] += np.random.ranf(self.dims)
        # largest absolute velocity sum larger than max_vel
        max_vel_sum = np.sum(np.abs(self.max_velocity))
        if np.any(max_vel_sum < p_vel_abs):
            _idx = np.argmax(np.sum((self.v[ max_vel_sum < p_vel_abs , :]), axis=1))
            self.max_velocity = self.v[_idx, :]
        r_particles = np.random.choice(range(self.num_particles), self.num_particles // 10)
        vals = np.random.ranf((len(r_particles), self.dims)) * self.max_velocity
        self.v[r_particles, :] = vals

    def run(self):

        c1 = c2 = 1.494

        for i in range(iterations):
            # apply function to all particles
            fx = self.func.eval(self.x)


            # update best solutions of points
            idx = fx < self.best_solution
            self.best_solution[idx] = fx[idx[0]]
            self.best_point[idx[0]] = self.x[idx[0], :]

            # update global best solutions
            idx = np.argmin(self.best_solution)
            if self.best_solution[:, idx][0] < self.best_global_solution:
                self.best_global_solution = self.best_solution[:, idx][0]
                self.best_global_point = self.x[idx, :]

            # update velocity after formula given constants c1 & c2, some randomness,
            # and personal/ global best solutions.
            r = np.random.ranf(2)
            self.v = self.weights[i] * self.v + r[0] * c1 * (self.best_point - self.x)
            # you can start the global updating later, if you want.
            self.v += c2 * r[1] * (self.best_global_point - self.x)

            self.chaos_for_position(i)
            self.chaos_for_velocity()

            # update position
            self.x += self.v

            # keep all particles in range [-n, n]
            self.x = np.clip(self.x, a_min=-self.n, a_max=self.n)
            yield self




if __name__ == '__main__':

    num_particles = 1
    dims = 2
    iterations = 100

    #run_pso(num_particles, dims, iterations)
    n = 10

    pso = DynamicPSO(num_particles=num_particles, dims=dims,
                     iterations=iterations)

    from vis.PSOVisualization import Particle2DVis

    vis = Particle2DVis(n=n, num_runs=iterations)

    background_function = rastrigin_bg(n=10, offset=np.zeros(dims))

    import time

    vis.set_background_function(background_function)
    background_function = rastrigin_bg(n=10, offset=np.zeros(dims))

    offset = np.array([0., 0.])
    for solution in pso.run():
        #vis.animate(solution=solution.x)
        vis.animate(solution=np.zeros(2))
        time.sleep(.01)
        background_function = rastrigin_bg(n=10, offset=solution.func.offsets)
        print(background_function)
        vis.set_background_function(background_function)
