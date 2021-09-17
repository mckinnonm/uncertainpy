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
	return -(0.5/sigma**2)*np.sum((data - x)**2)

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

class LogLikeWithGrad(tt.Op):

	itypes = [tt.dvector] # expects a vector of parameter values when called
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
		self.func = fds
		self.likelihood = loglike
		self.data = data
		#self.x = x
		self.sigma = sigma

		# initialise the gradient Op (below)
		self.FDS = FDS_TGA(self.func)
		self.logpgrad = LogLikeGrad(self.likelihood, self.data, self.x, self.sigma)

	def perform(self, node, inputs, outputs):
		# the method that is used when calling the Op
		theta, = inputs  # this will contain my variables
 		x = self.FDS(theta)
		# call the log-likelihood function
		#logl = self.likelihood(theta, self.x, self.data, self.sigma)
		logl = self.likelihood(self.x, self.data, self.sigma)

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
	itypes = [tt.dvector]
	otypes = [tt.dvector]

	def __init__(self, loglike, data, x, sigma):
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
		self.x = x
		self.sigma = sigma

	def perform(self, node, inputs, outputs):
		theta, = inputs

		# define version of likelihood function to pass to derivative function
		def lnlike(values):
			return self.likelihood(values, self.x, self.data, self.sigma)

		# calculate gradients
		grads = scipy.optimize.approx_fprime(theta, lnlike)

		outputs[0][0] = grads

ym = FDS_TGA(y_mean)

with mc.Model():
	# Priors
	theta = mc.Uniform('theta', lower=[8, 1.1e5, 0.85, 9, 1.4e5, 0.32, 0.001, 1.4e4, 0.54], upper=[12, 1.5e5, 0.95, 13, 1.8e5, 0.42, 0.5, 2.0e4, 0.64], shape=(9,))
	sigma = mc.Uniform('sigma', lower=0., upper=10., shape=(1,))

	theta = tt.as_tensor_variable(theta)
	sigma = tt.as_tensor_variable(sigma)

	db = mc.backends.Text('test')

	#Likelihood
	data = tga_exp.mlr

	logl = LogLikeWithGrad(my_loglike, data, x, sigma)

	#y_obs = mc.Normal('y_obs', observed=tga_exp.mlr, mu=mlr, tau=sigma**-2)
	y_obs = mc.DensityDist('y_obs', lambda v: logl(v), observed={'v': theta})

	# Configure and run MCMC simulation
	trace = mc.sample(10, chains = 1, cores = 1, trace = db)