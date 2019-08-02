import numpy as np
import sys
import operator
from pso.particle import Particle
from helper.constants import *


class PSO:
    class __PSO:
        def __init__(self, num_particles: int = 0,
                     dims: int = 0,
                     n: int = 0):

            self.num_particles = num_particles
            self.dims = dims
            self.n = n # feldausdehnung
            self.swarm = [Particle(n, dims) for _ in range(num_particles)]
            self.optimum = 0.0
            self.error_thres = 0.0
            self.best_global_solution = 0.0

            self.c1 = self.c2 = 1.494  # constant for update of weights

            self.start_global_update = 0
            self.stop_global_update = 0

            self.func_name = ""

            self.error_rates = []
            self.evaluations = []
            self.iteration = 0

        def init_evaluation_array(self, num_runs):
            """
                save evaluations for later plotting/ analysis.

            :param num_runs:
            :return:
            """
            self.evaluations = np.zeros((num_runs, self.num_particles, self.dims))

        def add_evaluation(self, array: np.ndarray):
            """
                add one iteration

            :param array:
            :return:
            """
            self.iteration += 1
            self.evaluations[self.iteration, :] = array


        def set_func_name(self, func_name: str):
            self.func_name = func_name

        def set_global_update_frame(self, start: float = 0.4,
                                    end: float = 0.8,
                                    num_runs: int = 0):
            """
            first+last iteration of global update
            """
            self.start_global_update = int(start * num_runs)
            self.stop_global_update = int(end * num_runs)

        def get_sd_of_ps(self, swarm: list) -> np.ndarray:
            """

            :param swarm:
            :return:
            """
            assert type(swarm) is list
            if len(self.swarm) == 0:
                return np.array([0.0])
            div = np.var([p.x for p in self.swarm])
            return (1 / len(swarm)) * div

        def run_pso(self,
                    num_runs: int = 0):
            """
            :param target_array:
            :param create_vis:
            :param show_error_vis:
            :param show_2dvis:
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

            """
            self.init_evaluation_array(num_runs)
            self.ws = np.linspace(0.9, 0.4, num_runs)  # decreasing weights
            # add random speedup -> that's actually not random @todo
            rand_speed_factor = self.dims ** 2
            v_brake = self.n / (self.n / self.n * 2)  # to slow down the swarm
            div_flag = True
            div_tolerance = np.sqrt(self.n * self.dims)  # relativly arbitary
            lowest_speed = 0.01
            self.best_global_solution = sys.maxsize
            best_global_point = np.zeros(self.dims)
            max_vel = np.zeros(self.dims)

            chaos_flag = False
            brake_flag = False
            i = 0
            while i < num_runs:
                array = np.zeros((len(self.swarm), self.dims))
                j = 0
                for particle in self.swarm:
                    f_n = particle.evalfunc_dicct[self.func_name]()
                    # update particles + global solution
                    if f_n < particle.best_solution:
                        particle.best_solution = f_n
                        particle.best_point = particle.x
                    if f_n < self.best_global_solution:
                        self.best_global_solution = f_n
                        best_global_point = particle.x
                    """
                        UPDATE: velocity
                    """
                    r = np.random.ranf(self.dims)
                    particle.v = self.ws[i] * particle.v + r[0] * self.c1 \
                                                           * (particle.best_point - particle.x)
                    if i > self.start_global_update:
                        # start global update not in the beginning, but after a
                        # set interval to increase diversity
                        particle.v += self.c2 * r[1] * (best_global_point - particle.x)
                    p_vel_abs = np.sum(np.abs(particle.v))
                    if p_vel_abs < lowest_speed:
                        # don't fall asleep...
                        particle.v *= r

                    # slow down particles and multiply for extra rand
                    if brake_flag:
                        particle.v = np.array([min(max(v, -v_brake),
                                                   v_brake) for v in particle.v] * r)

                    # update highest velocity
                    if np.sum(np.abs(max_vel)) < p_vel_abs :
                        max_vel = particle.v
                    """
                        UPDATE: position
                    """
                    particle.x = particle.x + particle.v
                    # set maximum (so particles can't escape area)
                    for d in range(self.dims):
                        particle.x[d] = min(max(particle.x[d], -self.n), self.n)

                    # add particle to vis-array
                    array[j] = particle.x
                    """
                        EXTRAS: FOR NEXT ITERATION
                    """
                    #  distance-based neighborhood
                    # indices = get_cluster_particles(particle, 0.1, swarm)
                    # for index in indices:
                    #    swarm[index].x *= swarm[index].v*(-1)*r*ws[i]

                    # chaos for velocity.
                    if chaos_flag:
                        r1 = np.random.ranf(self.dims)
                        if np.sum(r1) < np.sum(particle.v):
                            particle.v = r1 * particle.v * max_vel ** rand_speed_factor
                            ri = np.random.randint(particle.dims)
                            # constant factor to keep the chaos realistic
                            particle.x[ri] = r1[0] * ((2 * self.n) - self.n) * self.ws[i] * 0.45
                    j += 1

                # append array for later animation
                self.add_evaluation(array)

                """
                    CHECKS: Are done after every complete iteration.
                """
                # calculate diversity of particles
                # comparison-values are picked rather arbitary
                div = 0.0
                if div_flag:
                    # here i would like to have some dynamics - but not at the
                    # beginning
                    if self.start_global_update < i < self.stop_global_update:
                        div = self.get_sd_of_ps(self.swarm)
                        if np.sum(div) < (self.n / 2) * 2:
                            chaos_flag = True
                            brake_flag = False
                        else:
                            chaos_flag = False
                            brake_flag = True

                self.error_rates.append(np.sqrt((self.optimum - self.best_global_solution) ** 2))

                if i > int(num_runs-(num_runs / 4)) and np.sum(div) < div_tolerance:
                    # stop when it's very low.
                    print('f\nstop at iteration {i} of planned {num_runs}.')
                    break

                i += 1
            print(f"\nbest point ", best_global_point, "with solution {self.best_global_solution}."  )
            # print('diversity of swarm:\t',np.sum(get_sd_of_ps(swarm)))

        """
            brute force methods to get particles in (measurable)
            neighborhood. this is very cost-inefficient. hpso might be 
            much better.
        """
        def too_narrow(self, p1: Particle,
                       p2: Particle,
                       radius: float) -> bool:
            """
            in radius?
            :param p1:
            :param p2:
            :param radius:
            :return:
            """
            if np.sum(np.abs(p1.x - p2.x)) <= radius:
                return True
            else:
                return False

        def get_cluster_particles(self, c_particle: Particle,
                                  radius) -> list:
            """

            :param c_particle:
            :param radius:
            :param swarm:
            :return:
            """
            if len(self.num_particles) == 0:
                # should never happen.
                return []
            indices = []
            for i, particle in zip(range(self.num_particles), self.swarm):
                if c_particle is particle:
                    continue

                if self.too_narrow(c_particle, particle, radius):
                    indices.append(i)
            return indices

        def get_n_neighbour_particles(self,
                                      c_particle: Particle, n=3):
            """
            used to get n next particles-> get best solution for local winner
            (TODO)
            :param c_particle:
            :param n:
            :param swarm:
            :return:
            """
            if self.num_particles== 0:
                return
            particle_dists = {}
            for i, particle in zip(range(len(self.swarm)), self.swarm):
                particle_dists[i] = np.sum(np.abs(c_particle.x - particle.x))
            particle_dists = sorted(particle_dists.items(), key=operator.itemgetter(1))
            ## TODO wie rum wird sortiert?
            return particle_dists.keys()[:3]

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