# Developed by: Maurice Clerc (May 2011)
# Python implementation by: Gonzalo Vodanovic - Dic 2019 - gvodanovic@unc.edu.ar

import spso2011
import numpy as np

swarm_size = 10
swarm_dimension = 2
inertial_weight = 1/(2*np.log(2.0))
cognitive_acc = 0.5 + np.log(2.0)
social_acc = 0.5 + np.log(2.0)
lower_bound = [-1.0, -1.0]
upper_bound = [1.0, 1.0]
normalize = True
neighbors = 3
graph = True

pso_optimizer = spso2011.SPSO2011(swarm_size,
                                  swarm_dimension,
                                  inertial_weight,
                                  cognitive_acc,
                                  social_acc,
                                  neighbors,
                                  lower_bound,
                                  upper_bound,
                                  normalize,
                                  graph)

pso_optimizer.optimize(50)
