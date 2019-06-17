import matplotlib.pyplot as plt

import time

from vis.ScatterVisualizer import ScatterVisualizer

class Particle_2DVis(ScatterVisualizer):

    class __Particle_2DVis(ScatterVisualizer):
        """
        scatter-plot with points -> pso
        """
        def __init__(self, n, num_runs, interactive=False,
                     xlim=0 ,ylim=0, offset=0, colour_bar=True):
            super().__init__(interactive=interactive, xlim=xlim,
                             ylim=ylim, offset=offset, log_scale=False,
                             sexify=False)
            self.set_yx_lim([-n, n], [-n, n])
            self.num_runs = num_runs
            self.sleep_interval = .01
            self.colour_bar = colour_bar

        def set_point_size(self, point_size=2.5):
            self.my_plot.set_sizes([point_size] * len(self.target_array))

        def set_data(self, target_array, vals, t_m):
            """

            :param target_array:
            :param vals:
            :param t_m:
            :return:
            """
            self.target_array = target_array
            self.vals = vals
            self.t_m = t_m
            self.set_point_size()

        def plot_contours(self):
            """

            :return:
            """
            if len(self.vals ) >1 and len(self.t_m ) >1:
                CS = plt.contour(self.vals, self.vals, self.t_m, colors='black')
                plt.clabel(CS, inline=1, fontsize=6)

        def animate(self):
            """

            :return:
            """
            plt.imshow(self.t_m, extent=[-self.t_m.shape[1] / 2., self.t_m.shape[1] / 2.,
                                         -self.t_m.shape[0] / 2., self.t_m.shape[0] / 2.],
                       cmap='viridis')
            if self.colour_bar:
                plt.colorbar()
            i = 0
            while i < self.num_runs:
                self.my_plot.set_offsets(self.target_array[i, :])
                self.fig.canvas.draw()
                time.sleep(self.sleep_interval)
                i += 1

    instance = None

    def __init__(self, n, num_runs, interactive=True, xlim=0,ylim=0, offset=0, colour_bar=True):
        if not Particle_2DVis.instance:
            Particle_2DVis.instance = Particle_2DVis.__Particle_2DVis(n=n, num_runs=num_runs,
                                                                   interactive=interactive,
                                                                   xlim=xlim,ylim=ylim, colour_bar=colour_bar,
                                                                   offset=offset)
    def __getattr__(self, name):
        return getattr(self.instance, name)

