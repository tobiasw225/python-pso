# __filename__: main.py
#
# __description__: implementation of Particle Swarm Optimizer (PSO)
#  and Hierarchical PSO (HPSO). For more information go to run_-method.
#
# __remark__: HPSO can be used 'normal' -> pass weights (e.g. decreasing) or vHPSO or ^HPSO
# see personal_best_particles_update
#
# __todos__:
"""
@todo plot error at the end. -> da gibts noch einen bug
@todo constrainsts cf. pso (field-constraints?)
(@todo PH-HPSO)
"""

#
# Created by Tobias Wenzel in August 2017
# Copyright (c) 2017 Tobias Wenzel

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import time

from helper.background_function import *
from vis.PSOVisualization import Particle2DVis

from pso.common_pso import PSO
from pso.hpso import HPSO


if __name__ == '__main__':

    num_particles = 33
    num_runs = 100
    dims = 2
    use_hpso = True

    show_vis = False
    func_name = 'rastrigin'
    n = 10
    show_error_vis = False

    if use_hpso:
        pso = HPSO(num_particles=num_particles,
                   dims=dims,
                   n=n)
        pso.weight_function = "decr"

    else:
        pso = PSO(num_particles=num_particles,
                  dims=dims,
                  n=n)
        pso.set_global_update_frame(start=0.2, end=0.9, num_runs=num_runs)

    pso.set_eval_function(func_name)
    pso.run(num_runs)
    pso.print_hpso_best_solutions(pso.tree.root) # todo
    errors = pso.error_rates
    evaluation_steps = pso.evaluations

    if show_error_vis:
        df = pd.DataFrame(list(zip(np.arange(len(errors)), errors)), columns=['error','iteration'])
        sns.lineplot(x="iteration", y="error", data=df)
        plt.show()

    if show_vis:
        vis = Particle2DVis(n=n, num_runs=num_runs)
        background_function = generate_2d_background(func_name, n)
        vis.set_background_function(background_function)

        for i in range(num_runs):
            # not stopping ?
            vis.animate(solution=evaluation_steps[i, :])
            time.sleep(.05)

