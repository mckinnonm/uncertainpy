import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pymc3 as mc
import tga_exp
import theano
import theano.tensor as tt
import external_fds

def y_mean(theta):

    chid = 'test1'
    external_fds.input(chid, theta)
    external_fds.run_fds(chid)
    mlr = external_fds.read_fds(chid)

    # return the numpy array
    return mlr

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
ym = FDS_TGA(y_mean)

basic_model = mc.Model()

with mc.Model():
	# Priors
	theta = mc.Uniform('theta', lower=[7, 1e5, 0.85, 8, 1.3e5, 0.32, 0.001, 1.2e4, 0.54], upper=[13, 1.6e5, 0.95, 14, 1.9e5, 0.42, 0.5, 2.2e4, 0.64], shape=(9,))
	sigma = mc.Uniform('sigma', lower=0., upper=10., shape=(1,))

	theta = tt.as_tensor_variable(theta)

	# db = mc.backends.Text('test')

	#Likelihood
	mlr = ym(theta)
	y_obs = mc.Normal('y_obs', observed=tga_exp.mlr, mu=mlr, tau=sigma**-2)

	# Configure and run MCMC simulation
	# trace = mc.sample(10, chains = 1, cores = 4, trace = db)
	map_estimate = mc.find_MAP(model = basic_model, method = 'powell')

	map_estimate