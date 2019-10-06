import matplotlib.pyplot as plt

import time

from vis.ScatterVisualizer import ScatterVisualizer


class Particle2DVis(ScatterVisualizer):

    def __init__(self, n: float, num_runs: int, interactive: bool = True,
                 x_limit: float = 0.0, y_limit: float = 0.0,
                 offset: float = 0.0, colour_bar: bool = True):
        """

        :param n:
        :param num_runs:
        :param interactive:
        :param xlim:
        :param ylim:
        :param offset:
        :param colour_bar:
        """
        super().__init__(interactive=interactive, xlim=x_limit,
                         ylim=y_limit, offset=offset, log_scale=False)
        self.set_yx_lim([-n, n], [-n, n])
        self.num_runs = num_runs
        self.colour_bar = colour_bar
        self.colour_bar_set = False
        self.eval_steps = None
        self.bg_function = None

    def set_background_function(self, background_function):
        """

        :param background_function:
        :return:
        """
        self.bg_function = background_function

    def animate(self, solution):
        """

        :param solution:
        :return:
        """
        plt.imshow(self.bg_function, extent=[-self.bg_function.shape[1] / 2.,
                                             self.bg_function.shape[1] / 2.,
                                             -self.bg_function.shape[0] / 2.,
                                             self.bg_function.shape[0] / 2.],
                   cmap='viridis')

        if self.colour_bar and not self.colour_bar_set:
            plt.colorbar()
            point_size = 2.5
            self.my_plot.set_sizes([point_size] * len(solution))
            self.colour_bar_set = True

        self.my_plot.set_offsets(solution)
        self.fig.canvas.draw()


