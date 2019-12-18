# Developed by: Maurice Clerc (May 2011)
# Python implementation by: Gonzalo Vodanovic - Dic 2019 - gvodanovic@unc.edu.ar

import numpy as np
import utils
from numpy import linalg as la
import matplotlib.pyplot as plt
# TODO: Use numba optimization


class SearchSpace:
    def __init__(self, swarm_size, swarm_dimension, inertial_weight,
                 cognitive_acc, social_acc, neighbors,
                 lower_bound, upper_bound, normalize, graph, filter_param):
        # Attributes
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
        # Class Variables
        self.particles = []
        self.gbest_value = float('inf')
        self.gbest_position = np.empty(self.swarm_dimension)
        self.informant_probability = 1 - (1 - 1 / self.swarm_size) ** self.neighbors
        self.informers_matrix = np.empty((self.swarm_size, self.swarm_size))
        if self.graph:
            self.graph_x, self.graph_y = np.meshgrid(np.arange(-1.2, 1.2, 0.1), np.arange(-1.2, 1.2, 0.1))
            self.graph_z = self.graph_x ** 2 + self.graph_y ** 2 + 1
        # TODO: Should be an attributes
        self.uniform = False
        self.max_error = 0.05
        self.fitness_max_error = 10.0
        self.filter_param = filter_param

    def particles_initialization(self):
        if self.normalize:
            x_min = list(np.zeros(self.swarm_dimension))
            x_max = list(np.ones(self.swarm_dimension))
        else:
            x_min = self.lower_bound
            x_max = self.upper_bound
        for p in self.particles:
            p.position = [utils.alea(x_min[i], x_max[i]) for i in range(self.swarm_dimension)]
            p.velocity = [(utils.alea((x_min[i] - p.position[i]),
                                      (x_max[i] - p.position[i])))
                          for i in range(self.swarm_dimension)]

    def calc_filter(self, position):
        [design_d, design_g, design_f0, design_fs] = self.filter_param
        [c1, c2, c3, c4, ca, cb] = np.round(position)
        ca = ca * 16
        cb = cb * 16
        cap = [c1, c2, c3, c4, ca, cb]
        converged = True
        if (ca * cb) - ((c4 * c3) / 2.0) - ((c2 * c3) / 4.0) > 0.0:
            d = (c4 * np.sqrt(c3 / c2)) / (np.sqrt((ca * cb) - ((c4 * c3) / 2.0) - ((c2 * c3) / 4.0)))
            f0 = ((design_fs / (2.0 * np.pi)) *
                  np.sqrt(c2 * c3)) / (np.sqrt((ca * cb) - ((c4 * c3) / 2.0) - ((c2 * c3) / 4.0)))
            g = c1 / c2
            # TODO: Create function for check_errors
            error_d = np.abs((d - design_d) / design_d)
            if error_d > self.max_error:
                error_d = 10.0
                converged = False
            error_g = np.abs((g - design_g) / design_g)
            if error_g > self.max_error:
                error_g = 10.0
                converged = False
            error_f0 = np.abs((f0 - design_f0) / design_f0)
            if error_f0 > self.max_error:
                error_f0 = 10.0
                converged = False
            fitness = error_d + error_g + error_f0
        else:
            fitness = 30.0
            d = np.inf
            f0 = np.inf
            g = np.inf
            converged = False

        return [fitness, converged, cap, f0, d, g]

    def calc_fitness(self):
        for p in self.particles:
            if self.normalize:
                position = [(p.position[i] * (self.upper_bound[i] - self.lower_bound[i]) + self.lower_bound[i])
                            for i in range(self.swarm_dimension)]
            else:
                position = p.position

            [fitness, converged, cap, f0, d, g] = self.calc_filter(position)
            p.fitness = fitness

    def build_informers_matrix(self):  # TODO: Add if(stagn)
        informers_matrix = np.identity(self.swarm_size)  # Each particle informs itself
        # TODO: Improve syntaxis and optimize this two for loops
        for s in range(self.swarm_size):  # Each particle (column) informs at most K other at random
            for r in range(self.swarm_size):
                if r != s and utils.alea(0, 1) < self.informant_probability:
                    informers_matrix[s, r] = 1
        return informers_matrix

    def find_best_informant(self, informers_matrix, p):
        min_value = np.inf
        for s in range(self.swarm_size):
            if informers_matrix[s, p] == 1:
                if self.particles[s].fitness <= min_value:
                    min_value = self.particles[s].fitness
                    g_best = s
        return g_best

    def check_boundaries(self, particle_index, dimension_index):
        if not self.normalize:
            x_min = self.lower_bound[dimension_index]
            x_max = self.upper_bound[dimension_index]
        else:
            x_min = 0
            x_max = 1

        if self.particles[particle_index].position[dimension_index] > x_max:
            self.particles[particle_index].position[dimension_index] = x_max
            self.particles[particle_index].velocity[dimension_index] = \
                -0.5 * self.particles[particle_index].velocity[dimension_index]

        if self.particles[particle_index].position[dimension_index] < x_min:
            self.particles[particle_index].position[dimension_index] = x_min
            self.particles[particle_index].velocity[dimension_index] = \
                -0.5 * self.particles[particle_index].velocity[dimension_index]

    # Modify the search_space class best values
    def update_global_best(self):
        particle_fitness = np.zeros(self.swarm_size)
        for index in range(self.swarm_size):
            particle_fitness[index] = self.particles[index].fitness
        g = int(np.argmin(particle_fitness))
        if particle_fitness[g] < self.gbest_value:
            self.gbest_value = particle_fitness[g]
            if self.normalize:
                self.gbest_position = [(self.particles[g].position[i] *
                                        (self.upper_bound[i] - self.lower_bound[i]) + self.lower_bound[i])
                                       for i in range(self.swarm_dimension)]
            else:
                self.gbest_position = self.particles[g].position

    def move_particles(self):
        self.informers_matrix = self.build_informers_matrix()

        for particle_index in range(self.swarm_size):
            g_best = self.find_best_informant(self.informers_matrix, particle_index)
            gravity_center = np.empty(self.swarm_dimension)
            for dimension_index in range(self.swarm_dimension):
                # define a point p' on x-p, beyond p
                p_x_p = (self.particles[particle_index].position[dimension_index] +
                         self.cognitive_acc * (self.particles[particle_index].pbest_position[dimension_index] -
                                               self.particles[particle_index].position[dimension_index]))
                # % ... define a point g' on x-g, beyond g
                p_x_l = (self.particles[particle_index].position[dimension_index] +
                         self.social_acc * (self.particles[g_best].pbest_position[dimension_index] -
                                            self.particles[particle_index].position[dimension_index]))
                if g_best == particle_index:
                    gravity_center[dimension_index] = 0.5 * (self.particles[particle_index].position[dimension_index] +
                                                             p_x_p)
                else:
                    sw = 1.0 / 3.0
                    gravity_center[dimension_index] = sw * (self.particles[particle_index].position[dimension_index] +
                                                            p_x_p + p_x_l)

            rad = np.float64(la.norm(gravity_center - self.particles[particle_index].position))
            x_p = utils.alea_sphere(self.swarm_dimension, rad, self.uniform) + gravity_center

            for dimension_index in range(self.swarm_dimension):  # FIXME: Should be another Method?
                self.particles[particle_index].velocity[dimension_index] = \
                    self.inertial_weight * self.particles[particle_index].velocity[dimension_index] + \
                    x_p[dimension_index] - self.particles[particle_index].position[dimension_index]
                # The result is v(t+1)
                self.particles[particle_index].position[dimension_index] = \
                    self.particles[particle_index].position[dimension_index] + \
                    self.particles[particle_index].velocity[dimension_index]

                self.check_boundaries(particle_index, dimension_index)

    def check_stop_criteria(self):
        if self.gbest_value < self.fitness_max_error:
            return True
        else:
            return False

    def graph_particles(self, iteration):
        fig, ax = plt.subplots()
        ax.contourf(self.graph_x, self.graph_y, self.graph_z)
        p_index = 0
        for p in self.particles:
            if self.normalize:
                position = [(p.position[i] * (self.upper_bound[i] - self.lower_bound[i]) + self.lower_bound[i])
                            for i in range(self.swarm_dimension)]
            else:
                position = p.position
            ax.plot([position[0]], [position[1]],
                    markerfacecolor='k', markeredgecolor='k', marker='o', markersize=5)
            ax.plot(0.0, 0.0,
                    markerfacecolor='w', markeredgecolor='w', marker='o', markersize=5)
            for s in range(self.swarm_size):
                if self.informers_matrix[s, p_index] == 1:
                    if self.normalize:
                        position_informer = [(self.particles[s].position[i] *
                                             (self.upper_bound[i] - self.lower_bound[i]) + self.lower_bound[i])
                                             for i in range(self.swarm_dimension)]
                    else:
                        position_informer = self.particles[s]

                    if (position_informer[0] - position[0]) != 0.0 and (position_informer[1] - position[1]) != 0.0:
                        ax.arrow(position[0], position[1], position_informer[0] - position[0],
                                 position_informer[1] - position[1], shape='full', lw=0,
                                 length_includes_head=True, head_width=.05)

            p_index += 1
        ax.plot([self.gbest_position[0]], [self.gbest_position[1]],
                markerfacecolor='r', markeredgecolor='r', marker='o', markersize=5)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        if iteration < 10:
            plt.savefig('./graph/particle_positions_0' + str(iteration) + '.png')
        else:
            plt.savefig('./graph/particle_positions_' + str(iteration) + '.png')
        plt.close(fig)