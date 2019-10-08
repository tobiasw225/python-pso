# __filename__: tree.py
#
# __description__: methods for hierachical pso
#
# __remark__:
#
# __todos__:
#
# Created by Tobias Wenzel in August 2017
# Copyright (c) 2017 Tobias Wenzel

from pso.particle import Particle

class Node:
    def __init__(self, parent, level, n=10, dim=2):
        """
            Simple Implementation of a Tree Node. The Nodes are supposed
            to stay at their level, only the particle is exchanged for simplicity.

        :param data:
        :param parent:
        :param level:
        :param: n feldausweitung
        """
        self.level = level
        self.children = []
        self.parent = parent
        self.particle = Particle(n, dim)
        self.weight = 1 # to adjust weight vHPSO, ^HPSO


class Tree:
    def __init__(self,
                 num_children: int,
                 height: int,
                 n: int,
                 dims: int,
                 num_leafs: int):
        """
            Implementation of a tree for HPSO. Contains methods for
            inserting and creating nodes and swapping the particles.

        :param num_children:
        :param height:
        :param n:
        :param dims:
        :param num_leafs:
        """
        self.num_children = num_children
        self.height = height
        self.num_leafs = num_leafs
        self.n = n
        self.dims = dims
        self.root = None
        self.make_hpso_tree()

    def create_node(self, parent, level):
        """

        :param parent:
        :param level:
        :return:
        """
        return Node(parent,level,
                    self.n, self.dims)

    def make_hpso_tree(self):
        """

        :return:
        """
        self.root = self.insert(self.root)
        for i in range(1, self.num_leafs):
            self.insert(self.root)
        self.swap_top_down_breadth_first(self.root)

    def swap(self, node):
        """

        :param node:
        :return:
        """
        temp = node.particle
        for i in range(self.num_children):
            try:
                child = node.children[i]
                if child.particle.best_solution <= temp.best_solution:
                    node.particle = child.particle
                    child.particle = temp
                    temp = node.particle

            except IndexError:
                break

    def swap_top_down_breadth_first(self, node):
        """
        :return:
        """
        if node:
            self.swap(node)
            for child in node.children:
                self.swap_top_down_breadth_first(child)

    def insert(self, node):
        """
        die insert-methode dient nur dem zufälligem
        Aufbau des Baums. Das heißt hier braucht noch __nichts sortiert__!
        werden.

        :param node:
        :param data:
        :return:
        """

        if node is None:
            return self.create_node(parent=node, level=0)

        if len(node.children) < self.num_children:
            node.children.append(self.create_node(parent=node,
                                                  level=node.level+1))
            return node

        elif (len(node.children) == self.num_children)\
                and (node.level+1 < self.height):
            inserted = False
            for child in node.children:
                if self.insert(child) is not False:
                    inserted = True
                    break

            if not inserted:
                self.insert(node.parent)
            else:
                return node
        else:
            return False

    def points_of_tree_particles(self, array):
        """
        similar to get_array_of_tree, but not recursive and limited
        in respect to the height of the tree. this only returns the
        points, not the velocity, best solutions etc. for visualisation.

        :return:
        """
        array[0] = self.root.particle.x
        i = 1
        for local_best in self.root.children:
            array[i] = local_best.particle.x
            i += 1
            for sub_local in local_best.children:
                array[i] = sub_local.particle.x
                i += 1
                for leaf in sub_local.children:
                    array[i] = leaf.particle.x
                    i += 1
