# Developed by: Maurice Clerc (May 2011)
# Python implementation by: Gonzalo Vodanovic - Dic 2019 - gvodanovic@unc.edu.ar

import spso2011
import numpy as np
import timeit
from multiprocessing import Pool
import pandas as pd

swarm_size = 30
swarm_dimension = 6
inertial_weight = 1/(2*np.log(2.0))
cognitive_acc = 1
social_acc = 3
lower_bound = [1, 1, 1, 1, 1, 1]
upper_bound = [31, 31, 31, 31, 2, 2]
normalize = True
neighbors = 3
graph = False
max_iterations = 300
n_threads = 4

threads = [max_iterations for i in range(n_threads)]
max_error = 0.05

f0fsArray = pd.read_csv("f0_fs_array.csv", names=["f0", "fs"], skiprows=1)
spec_list_G = [0.5, 0.75, 1.0, 2.0, 4.0, 8.0, 16.0]
spec_list_d = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
spec_list_f = f0fsArray['f0']
spec_list_fs = f0fsArray['fs']
paramCombination_list = [(a, b, spec_list_f[c], spec_list_fs[c]) for a in spec_list_d for b in spec_list_G
                         for c in range(len(spec_list_f))]

pool = Pool(n_threads)
pso_filter_LP = pd.DataFrame(columns=['design_f0', 'design_d', 'design_G', 'fitness', 'iterations', 'converged',
                                      'C1', 'C2', 'C3', 'C4', 'CA', 'CB', 'calc_f0', 'calc_d', 'calc_G'])
start = timeit.default_timer()
counter = 0
for spec in paramCombination_list:
    print(spec)
    pso_optimizer = spso2011.SPSO2011(swarm_size,
                                      swarm_dimension,
                                      inertial_weight,
                                      cognitive_acc,
                                      social_acc,
                                      neighbors,
                                      lower_bound,
                                      upper_bound,
                                      normalize,
                                      graph,
                                      spec,
                                      max_error)
    for i in range(25):
        result = pool.map(pso_optimizer.optimize, threads)
        for thread in range(n_threads):
            pso_filter_LP.loc[counter] = [spec[2], spec[0], spec[1]] + result[thread]
            counter += 1


stop = timeit.default_timer()
print('Time: ', stop - start)

pso_filter_LP.to_csv('pso_filter_LP.csv')
