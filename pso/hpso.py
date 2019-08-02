
import numpy as np
from pso.common_pso import PSO
from pso.tree import *
from helper.constants import *
from pso.particle import Particle


class HPSO(PSO):
    class __HPSO(PSO):
        def __init__(self,
                     num_particles: int = 0,
                     dims: int = 0,
                     n: int = 0):
            height = 3
            self.num_leafs = num_particles
            # calculate the number of children needed, if height= 3
            num_children = int(np.floor(-1/2 + np.sqrt(1/4+(num_particles-1))))

            super().__init__(num_particles=num_particles,
                           dims=dims, n=n)
            self.tree = Tree(num_children=num_children,
                             height=height,
                             num_leafs=self.num_leafs,
                             n=n, dims=dims)
            self.max_weigth = 0.729
            self.min_weigth = 0.4
            self.set_level_weights(self.tree.root, method="decr")



        def set_level_weights(self,
                              node: Node,
                              method: str="decr"):
            """
            recursive function, that sets level weights according to method.
            decr. is default, meaning vHPSO.
            :param node:
            :param array:
            :param method:
            :return:
            """
            def weight_at_level_decr(level):
                return ((self.max_weigth - self.min_weigth) / self.tree.height)\
                       * level + self.min_weigth

            def weight_at_level_incr(level):
                return ((self.min_weigth - self.max_weigth) / self.tree.height)\
                       * level + self.max_weigth

            if node:
                if method == "decr":
                    node.weight = weight_at_level_decr(node.level)
                elif method == "incr":
                    node.weight = weight_at_level_incr(node.level)
                else:
                    print("weight-method not possible.", method)
                    return
                for child in node.children:
                    self.set_level_weights(child, method)

        def pers_best_recursive_update(self, node: Node):
            """

            :param node:
            :return:
            """
            if node:
                f_n = node.particle.evalfunc_dicct[self.func_name]()
                if f_n < node.particle.best_solution:
                    if np.shape(node.particle.x):
                        node.particle.best_solution = f_n
                for child in node.children:
                    self.pers_best_recursive_update(child)

        def update_lbest_recursive(self, node: Node):
            """
            recursively updates nodes of the the. the root node with level == 0
            is only influenced by the global best point. the rest is influenced
            by the best position found by their parent node. to add information
            about the importance of the particle we use a weight depending of the
            level:
                weight_at_level_decr -> root gets self.min_weight  = ^HPSO
                weight_at_level_incr -> root gets self.max_weigth  = vHPSO
                also possible: simple decreasing weights
            for more information see set_level_weights.
            :param node:
            :return:
            """
            if node:
                r = np.random.ranf(self.tree.dims)
                if node.level == 0:
                    weight = self.max_weigth
                else:
                    # kann man auch in node schieben
                    weight = node.weight #self.weight_at_level_incr(node.level)
                node.particle.v = weight * node.particle.v + r[0] * self.c1 * (
                    node.particle.best_point - node.particle.x)
                if node.level != 0:
                    node.particle.v += self.c2 * r[1] * (
                        node.parent.particle.best_point - node.particle.x)
                node.particle.x = node.particle.x + node.particle.v
                for child in node.children:
                    self.update_lbest_recursive(child)

        def get_points_of_tree_particles(self, array):
            """
            similar to get_array_of_tree, but not recursive and limited
            in respect to the height of the tree. this only returns the
            points, not the velocity, best solutions etc. for visualisation.
            :return:
            """
            array[0] = self.tree.root.particle.x
            i = 1
            for local_best in self.tree.root.children:
                array[i] = local_best.particle.x
                i += 1
                for sub_local in local_best.children:
                    array[i] = sub_local.particle.x
                    i += 1
                    for leaf in sub_local.children:
                        array[i] = leaf.particle.x
                        i += 1

        def get_array_of_tree(self, node, array=[]):
            """
            recursive function, that appends particles to the array, as it descends
            the tree.
            :param node:
            :param array:
            :return:
            """
            if node:
                array.append(node.particle.best_solution)
                for child in node.children:
                    self.get_array_of_tree(child, array)

        def get_sd(self) -> float:
            """
            returns the standard deviation of the particles in the
            tree structure. to achieve this, i first call get_array_of_tree
            to get the particles as an array.

            :return:
            """
            array = []
            self.get_array_of_tree(self.tree.root, array)
            return (1 / self.num_leafs) * np.var(array)

        def run_hpso(self,
                     num_runs: int = 100):
            """
            This is a basic implementation of the H-PSO-Algorithm. It's derived from PSO
            and has a tree-object. I figure this can be done way more elegantly, but then again:
            it's just for illustration. Most of the main-functions are called in recursive
            fashion. Since the tree-height shouldn't exceed 2 or max 3 levels, this can be
            implemented otherwise (not recursive).
            The particles are hold in a tree-structure: the particle with the global best
            solution should be in the root, local best solutions are kept in the lower levels
            and leafs (although they are just 'normal' particles). Every parent has a direct
            impact on it's child. First every particle checks it's current solution and updates
            it's personal best if it's better. Then the particles swapp their places. This is
            done in the following way: if the solution of a child is better (i.e. lower) than
            it's parent, they change places. This is done top-bottom.
            After, the particles change their position and speed according to their local and
            personal best solution. The weight changes with the level of the particle,
            ^HPSO and vHPSO are possible: vHPSO meaning the weight decreases towards the root.
            Unlike the PSO, I only added one addition, namely to
            stop if the diversity of the swarm is below a tolerance.

            So far there is no implementation of PH-PSO. A first try can be found in update_hpso.
            The visualisation is done like in the PSO class.

            :param num_runs:
            :return:
            """
            self.init_evaluation_array(num_runs)

            div_tolerance = np.sqrt(self.n * self.dims)**2

            for i in range(num_runs):
                self.pers_best_recursive_update(self.tree.root)
                self.tree.swap_top_down_breadth_first(self.tree.root)
                self.update_lbest_recursive(self.tree.root)  # vHPSO

                array = np.zeros((self.num_particles, self.dims))
                self.get_points_of_tree_particles(array)
                self.add_evaluation(array)

                self.error_rates.append(self.tree.root.particle.best_solution)

                if i > int(num_runs-(num_runs / 2)):
                    div = self.get_sd()
                    if np.sum(div) < div_tolerance:
                        # stop when it's very low.
                        print(f'\nstop at iteration {i} of planned {num_runs}.')
                        break

        def print_hpso_best_solutions(self, node: Node):
            """
            :param: node
            :return:
            """
            if node:
                points = node.level*'\t'
                print("\t"+points+str(node.particle.best_solution))
                for child in node.children:
                    self.print_hpso_best_solutions(child)

        def update_ph_pso(self, node: Node):
            """"
            ph-pso: reset leafs, randomize local-best
            in this step, the tree is "one"
            """

            if node:
                if node.level <= 1:
                    pass
                elif node.level == 2:
                    r = np.random.ranf(self.tree.dims)
                    # 're-randomizing', could be not enough.
                    node.particle.x *= 0.479 * r
                elif node.level == 3:
                    node.particle = Particle(n=self.tree.n,
                                             dims=self.tree.dims)
                else:
                    return
                for child in node.children:
                    self.update_ph_pso(child)

    instance = None

    def __init__(self,
                 num_particles=0,
                 dims=0,
                 n=0):
        super().__init__(num_particles=num_particles, dims=dims, n=n)
        if not HPSO.instance:
            HPSO.instance = HPSO.__HPSO(num_particles,
                                        dims, n)

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def __str__(self):
        res = ""
        res += colon_line+"\n"+ half_space_line+"H-PSO\n"+colon_line+"\n"
        res += "number of particles \t"+str(self.num_particles)+"\n"
        res += "height \t"+ str(self.tree.height) +"\n"
        res += "num children \t"+str(self.tree.num_children)+"\n"
        res += "dims \t"+str(self.dims) +"\n"
        res += "n\t"+ str(self.n) +"\n"
        res += "function name\t"+self.func_name
        return res
