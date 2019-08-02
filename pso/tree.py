# __filename__: tree.py
#
# __description__: methods for hierachical pso
#
# __remark__:
#
# __todos__: hpso
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
    def __init__(self, num_children=0, height=0, n=0, dims=0, num_leafs=0):
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
        if dims and n:
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
                    #node, child = child, node
                    # -> will ich nicht, da level ja gleich bleiben!

                    node.particle = child.particle
                    child.particle = temp

                    temp = node.particle

            except IndexError:
                break

    def swap_top_down_breadth_first(self, node):
        """
        hier wird dann getauscht.
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

    def get_split_trees(self, node):
        """
        should be recursive, not finished!
        :param node:
        :return:
        """
        print("@todo")
        return
        tree = Tree(num_children=self.num_children, height=self.height)
        tree.insert(node)
        for child in node.children:
            tree.insert(child)


def main():
    ## H-PSO wie PSO und nach Zeit wird getauscht?
    ## ansonsten würde ich doch bei jeder Verbesserung tauschen, oder?
    ##
    root = None
    num_children =3
    height = 2
    tree = Tree(num_children=num_children, height=height, n=10, dims=2)
    num_leafs = num_children**height + num_children+1

    random_values = np.random.rand(num_leafs)
    root = tree.insert(root, random_values[0])

    for i in range(1, len(random_values)):
        tree.insert(root, random_values[i])

    root = tree.swap_top_down_breadth_first(root)

    print('gb', root.data)
    for child in root.children:
        print('\tlb', child.data)
        for grand_child in child.children:
            print('\t\t',grand_child.data)
            for fuck in grand_child.children:
                print('\t\t\t',fuck.data)

if __name__ == '__main__':
    main()