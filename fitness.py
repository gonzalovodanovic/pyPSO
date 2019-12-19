import numpy as np
import numba as nb


@nb.jit(nopython=True)
def calc_filter_parameters(position, filter_param, max_error):
    [design_d, design_g, design_f0, design_fs] = filter_param
    cap = [np.float64(np.round(position[i])) for i in range(len(position))]
    cap[4] = np.float64(cap[4] * 16.0)
    cap[5] = np.float64(cap[5] * 16.0)
    converged = 1.0
    if (cap[4] * cap[5]) - ((cap[3] * cap[2]) / 2.0) - ((cap[1] * cap[2]) / 4.0) > 0.0:
        d = (cap[3] * np.sqrt(cap[2] / cap[1])) / (np.sqrt((cap[4] * cap[5]) - ((cap[3] * cap[2]) / 2.0) - ((cap[1] * cap[2]) / 4.0)))
        f0 = ((design_fs / (2.0 * np.pi)) *
              np.sqrt(cap[1] * cap[2])) / (np.sqrt((cap[4] * cap[5]) - ((cap[3] * cap[2]) / 2.0) - ((cap[1] * cap[2]) / 4.0)))
        g = cap[0] / cap[1]
        error_d = np.abs((d - design_d) / design_d)
        if error_d > max_error:
            error_d = 10.0
            converged = 0.0
        error_g = np.abs((g - design_g) / design_g)
        if error_g > max_error:
            error_g = 10.0
            converged = 0.0
        error_f0 = np.abs((f0 - design_f0) / design_f0)
        if error_f0 > max_error:
            error_f0 = 10.0
            converged = 0.0
        fitness = np.float64(error_d + error_g + error_f0)
    else:
        fitness = np.float64(30.0)
        d = np.float64(np.inf)
        f0 = np.float64(np.inf)
        g = np.float64(np.inf)
        converged = 0.0
    return fitness, converged, cap, f0, d, g


def calc_fitness(particles, filter_param, upper_bound, lower_bound, normalize, swarm_dimension, max_error):
    fitness = np.zeros(len(particles), dtype=np.float64)
    converged = np.zeros(len(particles))
    f0 = np.zeros(len(particles), dtype=np.float64)
    d = np.zeros(len(particles), dtype=np.float64)
    g = np.zeros(len(particles), dtype=np.float64)

    cap = np.zeros([len(particles), 6], dtype=np.float64)
    for p in range(len(particles)):
        if normalize:
            position = [(particles[p].position[i] * (upper_bound[i] - lower_bound[i]) + lower_bound[i])
                        for i in range(swarm_dimension)]
        else:
            position = particles[p].position
        fitness[p], converged[p], cap[p], f0[p], d[p], g[p] = calc_filter_parameters(np.float64(position), np.float64(filter_param), np.float64(max_error))
    return fitness
