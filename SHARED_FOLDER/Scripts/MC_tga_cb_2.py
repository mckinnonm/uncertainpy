import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pymc3 as mc
import tga_exp
from theano.compile.ops import as_op
import theano.tensor as tt
import external_fds

@as_op(itypes = [tt.dvector], otypes = [tt.dvector])
def y_mean(theta):

    chid = 'test1'
    external_fds.input(chid, theta)
    external_fds.run_fds(chid)
    mlr = external_fds.read_fds(chid)

    # return the numpy array
    return mlr

with mc.Model():
	# Priors
	theta = mc.Uniform('theta', lower=[7, 1e5, 0.85, 8, 1.3e5, 0.32, 0.001, 1.2e4, 0.54], upper=[13, 1.6e5, 0.95, 14, 1.9e5, 0.42, 0.5, 2.2e4, 0.64], shape=(9,))
	sigma = mc.Uniform('sigma', lower=0., upper=10., shape=(1,))

	theta = tt.as_tensor_variable(theta)

	mlr = y_mean(theta)

	#Likelihood
	y_obs = mc.Normal('y_obs', observed=tga_exp.mlr, mu=mlr, tau=sigma**-2)

	# Configure and run MCMC simulation
	step = mc.Metropolis()
	trace = mc.sample(10, chains = 4, cores = 4)
	mc.traceplot(trace)