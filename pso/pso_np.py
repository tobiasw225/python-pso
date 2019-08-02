
import numpy as np
#import cupy as cp
import sys


def rastrigin(row: np.ndarray):
    """
        apply rastrigin-function to matrix-row

    :param row:
    :return:
    """
    f_arr = np.frompyfunc(lambda d: d ** 2 - 10 * np.cos(np.pi * d) + 10, 1, 1)
    return np.sum(f_arr(row))


def run_pso(num_particles, dims, iterations):
    """

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


num_particles = 100
dims = 30
iterations = 1000

run_pso(num_particles, dims, iterations)