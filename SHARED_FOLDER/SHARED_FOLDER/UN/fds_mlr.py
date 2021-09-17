import uncertainpy as un
import chaospy as cp                
from math import pi
import numpy as np
import external_fds
import random

def fds_tga(theta_1,theta_2,theta_3,theta_4,theta_5,theta_6,theta_7,theta_8,theta_9):

    x = random.randint(1,10)
    chid = f'test{x}'

    # hr = 60,
    # A = [6.14, 10**theta[0], 10**theta[3], 10**theta[6]],
    # E = [2.35e4, 100000*theta[1], 100000*theta[4], 10000*theta[7]],
    # nu = [0, theta[2], theta[5], theta[8]],
    # rho = [520, 520*nu[1], 520*nu[1]*nu[2], 520*nu[1]*nu[2]*nu[3]]

    external_fds.input(chid, theta_1, theta_2, theta_3, theta_4, theta_5, theta_6, theta_7, theta_8, theta_9)
    external_fds.run_fds(chid)
    temp, mlr = external_fds.read_fds(chid)

    # return the numpy array
    return temp, mlr
   
# Create a model from the coffee_cup function and add labels
model = un.Model(run=fds_tga, labels=["Temperature (C)", "Total MLR (1/s)"])

# Create the distributions
A2_dist = cp.Uniform(8, 12)
A3_dist = cp.Uniform(9, 13)
A4_dist = cp.Uniform(0.001, 0.5)
E2_dist = cp.Uniform(1, 1.8)
E3_dist = cp.Uniform(1, 2.3)
E4_dist = cp.Uniform(2, 3.4)
s2_dist = cp.Uniform(0.8, 0.95)
s3_dist = cp.Uniform(0.3, 0.45)
s4_dist = cp.Uniform(0.5, 0.7)

# Define the parameter dictionary
parameters = {"theta_1": A2_dist, "theta_2": E2_dist, "theta_3": s2_dist, "theta_4": A3_dist, "theta_5": E3_dist, "theta_6": s3_dist, "theta_7": A4_dist, "theta_8": E4_dist, "theta_9": s4_dist}

# Set up the uncertainty quantification
UQ = un.UncertaintyQuantification(model=model, parameters=parameters, CPUs = 2)

# Perform the uncertainty quantification using
# polynomial chaos with point collocation (by default)
# We set the seed to easier be able to reproduce the result
data = UQ.quantify()
