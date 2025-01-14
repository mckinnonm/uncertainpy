import uncertainpy as un
import chaospy as cp                
from math import pi
import numpy as np
import external_fds
import random
import glob
import time

def fds_tga(theta_1,theta_2,theta_3,theta_4,theta_5,theta_6):

    t = random.randint(10,50)

    time.sleep(t)

    fs = glob.glob('*_tga.csv')
    files_ = [a.replace('_tga.csv','') for a in fs]
    files = [a.replace('test', '') for a in files_]
    file_nums = [int(a) for a in files]
    if file_nums:
        x_ = max(file_nums)
    else:
        x_ = 0
    x = str(x_+1)  
    chid = f'test{x}'

    # hr = 60,
    # A = [6.14, 10**theta[0], 10**theta[3], 10**theta[6]],
    # E = [2.35e4, 100000*theta[1], 100000*theta[4], 10000*theta[7]],
    # nu = [0, theta[2], theta[5], theta[8]],
    # rho = [520, 520*nu[1], 520*nu[1]*nu[2], 520*nu[1]*nu[2]*nu[3]]

    external_fds.tga_input(chid, theta_1, theta_2, theta_3, theta_4, theta_5, theta_6)
    external_fds.run_fds(chid)
    time.sleep(10)
    temp, mlr = external_fds.read_tga_fds(chid)

    # return the numpy array
    return temp, mlr

t = time.time()

# Create a model from the coffee_cup function and add labels
model = un.Model(run=fds_tga, labels=["Temperature (C)", "Total MLR (1/s)"])

# Create the distributions
A2_dist = cp.Uniform(8, 12) # 16.7
A3_dist = cp.Uniform(9, 13) # 11.1
E2_dist = cp.Uniform(1, 1.8) # 16.368
E3_dist = cp.Uniform(1, 2.3) # 16.416
s2_dist = cp.Uniform(0.8, 1.0) # 0.98
s3_dist = cp.Uniform(0.0, 0.1) # 0.002

# Define the parameter dictionary
parameters = {"theta_1": A2_dist, "theta_2": E2_dist, "theta_3": s2_dist, "theta_4": A3_dist, "theta_5": E3_dist, "theta_6": s3_dist}

# Set up the uncertainty quantification
UQ = un.UncertaintyQuantification(model=model, parameters=parameters, CPUs = 1)

# Perform the uncertainty quantification using
# polynomial chaos with point collocation (by default)
# We set the seed to easier be able to reproduce the result
data = UQ.quantify(method = 'pc')

elapsed = time.time() - t
print(f'elapsed time: {elapsed}')
