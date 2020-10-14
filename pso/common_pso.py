# __filename__: common_pso.py
#
# __description__: methods for pso
#
# __remark__:
#
# __todos__:
#
# Created by Tobias Wenzel in ~ August 2017
# Copyright (c) 2017 Tobias Wenzel
import os

import numpy as np
import sys
from pso.particle import Particle
from helper.constants import *
from helper.eval_funcs import eval_function
from helper.util import get_my_logger
from typing import List, Iterable, Tuple


class PSO:
    class __PSO:
        def __init__(self, num_particles: int = 0,
                     dims: int = 0,
                     n: int = 0):

            self.logger = get_my_logger(os.path.join(__file__ + 'log'))
            self.num_particles = num_particles
            self.dims = dims
            self.n = n
            self.swarm = [Particle(n, dims) for _ in range(num_particles)]
            self.optimum = 0.0
            self.error_thres = 0.0
            self.best_global_solution = 0.0

            self.c1 = self.c2 = 1.494  # constant for update of weights

            self.start_global_update = 0
            self.stop_global_update = 0
            self.v_brake = 0.0

            self.func_name = ""
            self.func = None

            self.error_rates = []
            self.evaluations = []
            self.iteration = 0
            self.lowest_speed = 0.01
            self.best_global_solution = sys.maxsize
            self.best_global_point = np.zeros(self.dims)
            self.max_vel = np.zeros(self.dims)
            # add random speedup -> that's actually not random @todo
            self.rand_speed_factor = self.dims ** 2
            self.v_brake = self.n / (self.n / self.n * 2)  # to slow down the swarm
            self.chaos_flag = False
            self.brake_flag = False

        def run(self, num_runs: int = 0, verbose: bool = True):
            self._run_pso(num_runs=num_runs, verbose=verbose)

        def init_evaluation_array(self, num_runs):
            """
                save evaluations for later plotting/ analysis.

            :param num_runs:
            :return:
            """
            self.evaluations = np.zeros((num_runs, self.num_particles, self.dims))

        def add_evaluation(self, array: np.ndarray):
            self.iteration += 1
            self.evaluations[self.iteration, :] = array

        def set_eval_function(self, func_name: str):
            assert func_name in eval_function
            self.func_name = func_name
            self.func = eval_function[func_name]

        def set_global_update_frame(self, start: float = 0.4,
                                    end: float = 0.8,
                                    num_runs: int = 0):
            """
            first+last iteration of global update
            """
            self.start_global_update = int(start * num_runs)
            self.stop_global_update = int(end * num_runs)

        def update_velocity(self, particle: Particle, i: int):
            r = np.random.ranf(self.dims)
            particle.v = self.ws[i] * particle.v + r[0] * self.c1 \
                         * (particle.best_point - particle.x)
            if i > self.start_global_update:
                # start global update not in the beginning, but after a
                # set interval to increase diversity
                particle.v += self.c2 * r[1] * (self.best_global_point - particle.x)
            p_vel_abs = np.sum(np.abs(particle.v))
            if p_vel_abs < self.lowest_speed:
                # don't fall asleep...
                particle.v *= r
            # slow down particles and multiply for extra rand
            if self.brake_flag:
                particle.v = np.array([min(max(v, -self.v_brake),
                                           self.v_brake) for v in particle.v] * r)
            # update highest velocity
            if np.sum(np.abs(self.max_vel)) < p_vel_abs:
                self.max_vel = particle.v

        def _run_pso(self,
                     num_runs: int = 0,
                     verbose: bool = False):
            """
            :param verbose:
            :param num_runs:
            :return:
            :description:
            This is a 'common' implementation of the PSO, only with some
            extensions. Meaning the algorithm runs a set number of iterations or
            if the sd of the swarm is less than a predefined value (this is relatively
            arbitrary, in this case the square root of the product(self.n * self.dims),
            which seemed to be a reasonable idea).
            Every iteration consists mainly of
            two steps: evaluation the field at a given place for every particle and updating
            it's position according to their personal best solution and the best solution
            found in the swarm. Every particle has a position in space and a vector defining
            it's velocity. As indicated, if a solution is better, it's updated.
            In this implementation you can set the relative time, when to start taking into
            account the global best solution. This can be done to give pers. best solutions
            a greater impact to not miss global minima. For more complex problems you should
            consider using H-PSO or even PH-HPSO. Some more things I changed about the original
            algorithm:
             - if the velocity is to low, speed up with a rand_speed_factor, correlating with the
             dimension of the problem.
             - don't exceed a maximum speed.
             - chaos for velocity
             - limit the search space to the given problem (-n,n)
            Some of them could be left out or extended or could be set in motion in a more elegant
            way. Some of the characteristics, e.g. the sum(dev) of the swarm I found in the paper
            or the script for swarm intelligence, others were found in an try-and-error way. I
            just want to illustrate the main principles for myself and everybody interested.

            The visualisation objects have to be initialized in a main method and passed as
            arguments. For more information about the classes look at 'my_visualisation.py'

            >>> pso = PSO(num_particles=5, dims=3, n=10)
            >>> pso.set_global_update_frame(start=0.2, end=0.9, num_runs=20)
            >>> pso.set_eval_function('rastrigin')
            >>> pso.run(10)
            """
            div_tolerance = np.sqrt(self.n * self.dims)  # relativly arbitary
            self.init_evaluation_array(num_runs)
            self.ws = np.linspace(0.9, 0.4, num_runs)  # decreasing weights
            i = 0
            while i < num_runs:
                array = np.zeros((len(self.swarm), self.dims))
                j = 0
                for particle in self.swarm:
                    f_n = self.func(particle.x)
                    # update particles + global solution
                    if f_n < particle.best_solution:
                        particle.best_solution = f_n
                        particle.best_point = particle.x
                    if f_n < self.best_global_solution:
                        self.best_global_solution = f_n
                        self.best_global_point = particle.x

                    self.update_velocity(particle, i)
                    self.update_position(particle)

                    # add particle to vis-array
                    array[j] = particle.x
                    if self.chaos_flag:
                        self.chaos_for_velocity(particle, i)
                    j += 1

                # append array for later animation
                self.add_evaluation(array)
                self.error_rates.append(np.sqrt((self.optimum - self.best_global_solution) ** 2))
                div = self.diversity_check(i)
                if i > int(num_runs-(num_runs / 4)) and np.sum(div) < div_tolerance:
                    if verbose:
                        self.logger.info(f'\nstop at iteration {i} of planned {num_runs}.')
                    break
                i += 1
            if verbose:
                self.logger.info(f"\nbest point {self.best_global_point} with solution {self.best_global_solution}."  )

        def diversity_check(self, i: int) -> float:
            div = 0.0
            if self.start_global_update < i < self.stop_global_update:
                div = self.sd()
                if np.sum(div) < (self.n / 2) * 2:
                    self.chaos_flag = True
                    self.brake_flag = False
                else:
                    self.chaos_flag = False
                    self.brake_flag = True
            return div

        def chaos_for_velocity(self, particle: Particle, i: int):
            r1 = np.random.ranf(self.dims)
            if np.sum(r1) < np.sum(particle.v):
                particle.v *= r1 * self.rand_speed_factor
                ri = np.random.randint(particle.dims)
                # constant factor to keep the chaos realistic
                particle.x[ri] = r1[0] * ((2 * self.n) - self.n) * self.ws[i] * 0.45

        def update_position(self, particle: Particle):
            # set maximum (so particles can't escape area)
            particle.x = particle.x + particle.v
            for d in range(self.dims):
                particle.x[d] = min(max(particle.x[d], -self.n), self.n)

        def sd(self) -> np.ndarray:
            if len(self.swarm) == 0:
                return np.array([0.0])
            div = np.var([p.x for p in self.swarm])
            return (1 / len(self.swarm)) * div

    instance = None

    def __init__(self, *args, **kwargs):
        if not PSO.instance:
            PSO.instance = PSO.__PSO(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def __str__(self):
        res = ""
        res += colon_line+"\n"+ half_space_line+"PSO\n"+colon_line+"\n"
        res += "number of particles \t"+str(self.num_particles)+"\n"
        res += "dims \t"+str(self.dims) + "\n"
        res += "n\t" + str(self.n)
        return res
