#!/usr/bin/env python
"""Module for setting up statistical models"""
from __future__ import division
from math import pi
import numpy as np
import pymc3 as mc
import external_fds
import tga_exp

def fds_mlr():
	with mc.Model() as model:
		# Priors
		theta = mc.Uniform('theta', lower=[7, 1e5, 0.85, 8, 1.3e5, 0.32, 0.001, 1.2e4, 0.54], upper=[13, 1.6e5, 0.95, 14, 1.9e5, 0.42, 0.5, 2.2e4, 0.64])
		sigma = mc.Uniform('sigma', lower=0., upper=10.)
		# Model
		@mc.Deterministic
		def y_mean(theta=theta):
			chid = 'test1'
			external_fds.input(chid = chid, 
				hr=60,
				A=[6.14, 10**theta[0], 10**theta[3], 10**theta[6]],
				E=[2.35e4, theta[1], theta[4], theta[7]],
				nu=[0, theta[2], theta[5], theta[8]],
				rho=[520, 520*nu[1], 520*nu[1]*nu[2], 520*nu[1]*nu[2]*nu[3]])
		
			external_fds.run_fds(chid)
			mlr = external_fds.read_fds(casename)

			# Print MLR vs. time for each iteration
			# graphics.plot_fds_mlr(mlr)

			return mlr

		# Likelihood
		y_obs = mc.Normal('y_obs', observed=tga_exp.mlr, mu=y_mean, tau=sigma**-2)

		return vars()