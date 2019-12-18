# Standard PSO 2011
# Translated from the C version by
# Dr. Mahamed G.H. Omran (omran.m@gust.edu.kw) 7-May-2011

# Modified and improved by: Maurice Clerc
# 2011-05-23
# 2018-04 (test options)
# Python implementation by: Gonzalo Vodanovic (gvodanovic@unc.edu.ar)
# 2019-12

from SearchSpace import SearchSpace
from Particle import Particle
import numpy as np
# TODO: Use numba optimization


class SPSO2011:
    def __init__(self, swarm_size, swarm_dimension, inertial_weight,
                 cognitive_acc, social_acc, neighbors,
                 lower_bound, upper_bound, normalize, graph, filter_param):
        self.swarm_size = swarm_size
        self.swarm_dimension = swarm_dimension
        self.inertial_weight = inertial_weight
        self.cognitive_acc = cognitive_acc
        self.social_acc = social_acc
        self.neighbors = neighbors
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.normalize = normalize
        self.graph = graph
        self.filter_param = filter_param

    def optimize(self, iterations):
        # Init the particles with position and velocity
        # Create the search space
        search_space = SearchSpace(self.swarm_size,
                                   self.swarm_dimension,
                                   self.inertial_weight,
                                   self.cognitive_acc,
                                   self.social_acc,
                                   self.neighbors,
                                   self.lower_bound,
                                   self.upper_bound,
                                   self.normalize,
                                   self.graph,
                                   self.filter_param)

        particles_vector = [Particle(self.swarm_dimension) for _ in range(search_space.swarm_size)]
        search_space.particles = particles_vector
        search_space.particles_initialization()
        search_space.calc_fitness()
        [search_space.particles[i].update_personal_best() for i in range(search_space.swarm_size)]
        search_space.update_global_best()
        iters = 0
        if search_space.graph:
            search_space.graph_particles(0)
        for iteration_index in range(iterations):
            search_space.move_particles()
            [search_space.particles[i].update_personal_best() for i in range(search_space.swarm_size)]
            search_space.calc_fitness()
            search_space.update_global_best()
            if search_space.graph:
                search_space.graph_particles(iteration_index + 1)
            if search_space.check_stop_criteria():
                break
            iters += 1
        print("The best solution is: ", np.round(search_space.gbest_position))
        print("\n Number of iterations: ", iters)
        print("\n Best fitness: ", search_space.gbest_value)

        [fitness, converged, cap, f0, d, g] = search_space.calc_filter(search_space.gbest_position)
        result = [fitness, iters, converged] + cap + [f0, d, g]
        # result = search_space.calc_filter(search_space.gbest_position)
        print(result)
        return result
# Bibliography:
# http://clerc.maurice.free.fr/pso/random_topology.pdf
# http://mat.uab.cat/~alseda/MasterOpt/SPSO_descriptions.pdf
