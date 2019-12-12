# Developed by: Maurice Clerc (May 2011)
# Python implementation by: Gonzalo Vodanovic - Dic 2019 - gvodanovic@unc.edu.ar

import numpy as np


class Particle:
    def __init__(self, swarm_dimension):
        self.position = np.empty(swarm_dimension)
        self.velocity = np.empty(swarm_dimension)
        self.fitness = np.inf
        self.pbest_position = self.position
        self.pbest_fitness = float('inf')

    # Modify the particle class best values
    # TODO: Analyse and improve performance of this method
    def update_personal_best(self):
        if self.fitness <= self.pbest_fitness:
            self.pbest_position = self.position
            self.pbest_fitness = self.fitness
