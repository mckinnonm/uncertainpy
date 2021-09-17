import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pymc3 as mc
import tga_exp
import theano
import theano.tensor as tt
import external_fds
import scipy
import math

def y_mean(theta):

	chid = 'test1'
	external_fds.input(chid, theta)
	external_fds.run_fds(chid)
	mlr = external_fds.read_fds(chid)

	# return the numpy array
	return mlr

def my_loglike(x, data, sigma):
	"""
	A Gaussian log-likelihood function for a model with parameters given in theta
	"""
	return -(0.5*math.log(2*math.pi))-(0.5*math.log(sigma**2))-(0.5/sigma**2)*np.sum((data - x)**2)

# define a theano Op for our likelihood function
class FDS_TGA(tt.Op):

	"""
	Specify what type of object will be passed and returned to the Op when it is
	called. In our case we will be passing it a vector of values (the parameters
	that define our model) and returning a single "scalar" value (the
	log-likelihood)
	"""
	itypes = [tt.dvector] # expects a vector of parameter values when called
	otypes = [tt.dvector] # outputs a single scalar value (the log likelihood)

	def __init__(self, ymean):
		"""
		Initialise the Op with various things that our log-likelihood function
		requires. Below are the things that are needed in this particular
		example.

		Parameters
 		----------
		loglike:
			The log-likelihood (or whatever) function we've defined
		data:
			The "observed" data that our log-likelihood function takes in
		x:
			The dependent variable (aka 'x') that our model requires
		sigma:
			The noise standard deviation that our function requires.
		"""

		# add inputs as class attributes
		self.y = ymean

	def perform(self, node, inputs, outputs):
		# the method that is used when calling the Op
		theta, = inputs  # this will contain my variables

		# call the log-likelihood function
		mlr = self.y(theta)

		outputs[0][0] = np.array(mlr) # output the log-likelihood

# Generate Model

class LogLike(tt.Op):

	itypes = [tt.dvector, tt.dscalar] # expects a vector of parameter values when called
	otypes = [tt.dscalar] # outputs a single scalar value (the log likelihood)

	def __init__(self, loglike, data, sigma): # x
		"""
		Initialise with various things that the function requires. Below
		are the things that are needed in this particular example.

		Parameters
		----------
		loglike:
			The log-likelihood (or whatever) function we've defined
		data:
			The "observed" data that our log-likelihood function takes in
		x:
			The dependent variable (aka 'x') that our model requires
		sigma:
			The noise standard deviation that out function requires.
		"""

		# add inputs as class attributes
		self.likelihood = loglike
		self.data = data

		self.fds = FDS_TGA(y_mean)

	def perform(self, node, inputs, outputs):
		# the method that is used when calling the Op
		theta, sigma, = inputs
		# call the log-likelihood function
		mlr = self.fds(theta)
		#logl = self.likelihood(theta, self.x, self.data, self.sigma)
		logl = self.likelihood(mlr, self.data, sigma)

		outputs[0][0] = np.array(logl) # output the log-likelihood

ndraws = 4000 # number of draws from the distribution
nburn = 1000 # number of "burn-in points" (which we'll discard)

data = tga_exp.mlr
logl = LogLike(my_loglike, data, sigma)

with mc.Model():
	# Priors
	theta = mc.Uniform('theta', lower=[8, 1.1e5, 0.85, 9, 1.4e5, 0.32, 0.11, 1.4e4, 0.54], upper=[12, 1.5e5, 0.95, 13, 1.8e5, 0.42, 0.5, 2.0e4, 0.64], shape=(9,))
	sigma = mc.Uniform('sigma', lower=0., upper=10.)

	theta = tt.as_tensor_variable(theta)
	sigma = tt.as_tensor_variable(sigma)

	#Likelihood

	#y_obs = mc.Normal('y_obs', observed=tga_exp.mlr, mu=mlr, tau=sigma**-2)
	y_obs = mc.DensityDist('y_obs', 
						   logl, 
						   oberved={'theta':theta,
									'sigma':sigma})

	# Configure and run MCMC simulation
	trace = mc.sample(ndraws, tune=nburn, discard_tuned_samples=True, chains = 5, cores = 1)
 
# plot the traces
mc.summary(trace).round(2)