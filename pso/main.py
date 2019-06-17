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

import sys
import pickle

from background_function import *
sys.path.append('/home/tobias/mygits/python_vis/pso')
# error+2d vis
from vis.PSOVisualization import Particle_2DVis

from pso.common_pso import PSO
from pso.hpso import HPSO




if __name__ == '__main__':

    num_particles = 13#int(input("num particles") or 13)
    num_runs = 100#int(input("num runs") or 100)
    dims = int(input("num dims") or 2)
    use_hpso = True

    show_vis = True
    func_name = 'hoelder_table'
    n = 10  # ausweitung des feldes
    create_vis = False
    show_error_vis = False
    num_children = 5
    height = 2 # please don't change

    num_particles = num_children ** height + num_children + 1

    target_array = None
    if (dims == 3 or dims ==2) and show_vis:
        target_array = np.zeros((num_runs, num_particles, dims))

    if use_hpso:
        hpso = HPSO(num_particles=num_particles,
                    dims=dims,
                    n=n, num_children=num_children, height=height)
        hpso.set_func_name(func_name=func_name)
        print(hpso)

        hpso.run_hpso(target_array, num_runs=num_runs,show_vis=show_vis,
                      show_error_vis=show_error_vis)


        hpso.get_hpso_best_solutions(hpso.tree.root)
    else:
        pso = PSO(num_particles=num_particles,
                  dims=dims,
                  n=n)
        pso.set_func_name(func_name)
        print(pso)
        pso.set_global_update_frame(start=0.2, end=0.9, num_runs=num_runs)
        pso.run_pso(target_array=target_array,
                    create_vis=create_vis, show_error_vis=show_error_vis,
                    show_2dvis=show_vis, num_runs=num_runs)

    if show_vis and dims==2:
        vis2d = Particle_2DVis(n=n, num_runs=num_runs)
        values, t_m = background_function[func_name]()
        vis2d.set_data(target_array, values, t_m)
        vis2d.plot_contours()
        vis2d.set_point_size(3.5)
        vis2d.animate()
    elif create_vis and dims==3:
        print("write data to file. navigate to 3dvis-script!")
        with open("/home/tobias/Dokumente/trajectories3d", 'wb') as fout:
             pickle.dump(target_array, fout)

    #create_video(source_path="/home/tobias/Bilder/pso/*.pdf", frame_rate=4,
    #             output_file="/home/tobias/Bilder/pso/pso.mp4", pdf=True, num_runs=num_runs)