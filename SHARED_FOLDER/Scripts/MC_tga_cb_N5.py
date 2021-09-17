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
# class FDS_TGA(tt.Op):

# 	"""
# 	Specify what type of object will be passed and returned to the Op when it is
# 	called. In our case we will be passing it a vector of values (the parameters
# 	that define our model) and returning a single "scalar" value (the
# 	log-likelihood)
# 	"""
# 	itypes = [tt.dvector] # expects a vector of parameter values when called
# 	otypes = [tt.dvector] # outputs a single scalar value (the log likelihood)

# 	def __init__(self, ymean):
# 		"""
# 		Initialise the Op with various things that our log-likelihood function
# 		requires. Below are the things that are needed in this particular
# 		example.

# 		Parameters
#  		----------
# 		loglike:
# 			The log-likelihood (or whatever) function we've defined
# 		data:
# 			The "observed" data that our log-likelihood function takes in
# 		x:
# 			The dependent variable (aka 'x') that our model requires
# 		sigma:
# 			The noise standard deviation that our function requires.
# 		"""

# 		# add inputs as class attributes
# 		self.y = ymean

# 	def perform(self, node, inputs, outputs):
# 		# the method that is used when calling the Op
# 		theta, = inputs  # this will contain my variables

# 		# call the log-likelihood function
# 		mlr = self.y(theta)

# 		outputs[0][0] = np.array(mlr) # output the log-likelihood

# Generate Model

class LogLikeWithGrad(tt.Op):

	itypes = [tt.dvector] #, tt.dscalar] # expects a vector of parameter values when called
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
		self.sigma = sigma

		#self.logpgrad = LogLikeGrad(self.likelihood, self.data)
		self.logpgrad = LogLikeGrad(self.likelihood, self.data, self.sigma)

	def perform(self, node, inputs, outputs):
		# the method that is used when calling the Op
		# theta, sigma, = inputs
		theta, = inputs
		# call the log-likelihood function
		mlr = y_mean(theta)
		#logl = self.likelihood(theta, self.x, self.data, self.sigma)
		logl = self.likelihood(mlr, self.data, self.sigma)

		outputs[0][0] = np.array(logl) # output the log-likelihood

	def grad(self, inputs, g):
		# the method that calculates the gradients - it actually returns the
		# vector-Jacobian product - g[0] is a vector of parameter values 
		theta, = inputs  # our parameters 
		return [g[0]*self.logpgrad(theta)]

class LogLikeGrad(tt.Op):

	"""
	This Op will be called with a vector of values and also return a vector of
	values - the gradients in each dimension.
	"""
	itypes = [tt.dvector]#, tt.dscalar]
	otypes = [tt.dvector]

	def __init__(self, loglike, data, sigma):
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
		self.sigma = sigma

	def perform(self, node, inputs, outputs):
		theta, = inputs
		# call the log-likelihood function
		# define version of likelihood function to pass to derivative function
		def lnlike(values):
			mlr_grad = y_mean(values)
			return self.likelihood(mlr_grad, self.data, self.sigma)

		# calculate gradients
		grads = scipy.optimize.approx_fprime(theta, lnlike, 1e-5)

		outputs[0][0] = grads

ndraws = 1000 # number of draws from the distribution
nburn = 200 # number of "burn-in points" (which we'll discard)

data = tga_exp.mlr
sigma = 0.2
#logl = LogLikeWithGrad(my_loglike, data)
logl = LogLikeWithGrad(my_loglike, data, sigma)

with mc.Model():
	# Priors
	A1 = mc.Uniform('A1', lower=8, upper=12)
	E1 = mc.Uniform('E1', lower=1.1e5, upper=1.5e5)
	nu1 = mc.Uniform('nu1', lower=0.85, upper=0.95)
	A2 = mc.Uniform('A2', lower=9, upper=13)
	E2 = mc.Uniform('E2', lower=1.4e5, upper=1.8e5)
	nu2 = mc.Uniform('nu2', lower=0.32, upper=0.42)
	A3 = mc.Uniform('A3', lower=0.01, upper=0.5)
	E3 = mc.Uniform('E3', lower=1.4e4, upper=2.0e4)
	nu3 = mc.Uniform('nu3', lower=0.54, upper=0.64)
	sigma = mc.Uniform('sigma', lower=0., upper=10.)

	theta = tt.as_tensor_variable([A1, E1, nu1, A2, E2, nu2, A3, E3, nu3])
	#sigma = tt.as_tensor_variable(sigma)

	#Likelihood

	#y_obs = mc.Normal('y_obs', observed=tga_exp.mlr, mu=mlr, tau=sigma**-2)
	#mc.DensityDist('y_obs', lambda v,y: logl(v, y), observed={'v': theta, 'y': sigma})
	mc.DensityDist('y_obs', lambda v: logl(v), observed={'v': theta})

	# Configure and run MCMC simulation
	trace = mc.sample(ndraws, tune=nburn, discard_tuned_samples=True, chains = 5, cores = 1)
 
# plot the traces
mc.summary(trace).round(2)