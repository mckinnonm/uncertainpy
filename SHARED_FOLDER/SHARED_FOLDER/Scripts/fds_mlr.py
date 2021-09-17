from __future__ import division
from math import pi
import numpy as np
import pymc3 as mc
import external_fds
import tga_exp

def y_mean(theta):
 
    chid = 'test1'

    hr = 60,
    A = [6.14, 10**theta[0], 10**theta[3], 10**theta[6]],
    E = [2.35e4, theta[1], theta[4], theta[7]],
    nu = [0, theta[2], theta[5], theta[8]],
    rho = [520, 520*nu[1], 520*nu[1]*nu[2], 520*nu[1]*nu[2]*nu[3]]

    external_fds.input(chid, hr, A, E, nu, rho)
    external_fds.run_fds(chid)
    mlr = external_fds.read_fds(chid)

    # return the numpy array
    return mlr