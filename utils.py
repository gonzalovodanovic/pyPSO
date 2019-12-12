# Developed by: Maurice Clerc (May 2011)
# Python implementation by: Gonzalo Vodanovic - Dic 2019 - gvodanovic@unc.edu.ar

import random
import math
import numpy as np
from numpy import linalg as LA


def alea(a, b):
    # Random real number between a and b
    return a + (random.random() * (b - a))
    # return a + (0.75 * (b - a))


def alea_normal(mean, std_dev):
    # Use the polar form of the Box-Muller transformation to obtain a pseudo
    # random number from a Gaussian distribution

    w = 2
    while w >= 1:
        x1 = (2.0 * alea(0, 1)) - 1.0
        x2 = (2.0 * alea(0, 1)) - 1.0
        w = (x1**2) + (x2**2)

    w = math.sqrt((-2.0 * math.log(w)) / w)
    y1 = x1 * w

    if alea(0, 1) < 0.5:
        y1 = (-1) * y1

    return (y1 * std_dev) + mean


def alea_sphere(D, radius, unif):
    # Random point in a hypersphere
    # Put  a random point inside the hypersphere S(0,radius) (center 0, radius 1).
    # unif=1 => Uniform distribution
    #      0 => non uniform (denser around the centre)

    # --------- Step 1. Direction
    x = np.empty(D)
    for j in range(D):
        x[j] = alea_normal(0, 1)

    # Get the 2-norm of the alea vector
    l = LA.norm(x, 2)

    # --------- Step 2. Random radius
    if not unif:
        r = alea(0, 1)
    elif unif:
        r = alea(0, 1)**(1/D)
#    print(r, radius, x, l)
    return r*radius*x/l


