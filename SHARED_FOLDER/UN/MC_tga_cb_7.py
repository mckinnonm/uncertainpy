import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pymc3 as mc
import tga_exp
import theano
import theano.tensor as tt
import external_fds
import glob
import time
import arviz as az
import functools as ft
import math

class FuncCallCounter(type):
    """ A Metaclass which decorates all the methods of the 
        subclass using call_counter as the decorator
    """
    
    @staticmethod
    def call_counter(func):
        """ Decorator for counting the number of function 
            or method calls to the function or method func
        """
        def helper(*args, **kwargs):
            helper.calls += 1
            return func(*args, **kwargs)
        helper.calls = 0
        helper.__name__= func.__name__
    
        return helper
    
    
    def __new__(cls, clsname, superclasses, attributedict):
        """ Every method gets decorated with the decorator call_counter,
            which will do the actual call counting
        """
        for attr in attributedict:
            if callable(attributedict[attr]) and not attr.startswith("__"):
                attributedict[attr] = cls.call_counter(attributedict[attr])
        
        return type.__new__(cls, clsname, superclasses, attributedict)
    

class A(metaclass=FuncCallCounter):
    
	def input(self, theta0, theta1, theta2):
	
		num = x.input.calls % 4
		chid = 'test' + str(num)

		hr = 60,
		A = [6.14, 10**theta0],
		E = [2.35e4, theta1*100000],
		nu = [0, theta2],
		rho = [520, (520*theta2)]

		HEAD = '&HEAD CHID = \'' + chid + '\' /'
		MISC = '&MISC SOLID_PHASE_ONLY=.TRUE. /'
		MESH = '&MESH IJK = 3, 3, 3, XB = -0.15, 0.15, -0.15, 0.15, 0.0, 0.3 /'
		TIME = '&TIME T_END=3600., WALL_INCREMENT = 1., DT = 0.05 /'
		SPEC = '&SPEC ID = \'METHANE\' /'
		VENT = '&VENT XB = -0.05, 0.05, -0.05, 0.05, 0.0, 0.0, SURF_ID = \'SAMPLE\' /'
		SURF = '&SURF ID = \'SAMPLE\', TGA_ANALYSIS = .TRUE., TGA_HEATING_RATE = ' + str(hr) + ', MATL_ID(1,1) = \'LB\', MATL_ID(1,2) = \'MOISTURE\', MATL_MASS_FRACTION(1,:) = 0.98,0.02, THICKNESS = 0.001, CELL_SIZE_FACTOR = 0.1, STRETCH_FACTOR(1) = 1. /'
		DUMP = '&DUMP DT_DEVC = 1, SUPPRESS_DIAGNOSTICS =  /'
		TAIL = '&TAIL /'
		DEVC = []
		MATL = []
		IFILE = []

		DEVC.append('&DEVC XYZ = 0.0, 0.0, 0.0, IOR = 3, QUANTITY = \'WALL TEMPERATURE\', ID = \'temp\' /' + '\n')
		DEVC.append('&DEVC XYZ = 0.0, 0.0, 0.0, IOR = 3, QUANTITY = \'BACK WALL TEMPERATURE\', ID = \'back_temp\' /' + '\n')
		DEVC.append('&DEVC XYZ = 0.0, 0.0, 0.0, IOR = 3, QUANTITY = \'MASS FLUX\', SPEC_ID = \'METHANE\', ID=\'MF\' /' + '\n')
		DEVC.append('&DEVC XYZ = 0.0, 0.0, 0.0, IOR = 3, QUANTITY = \'WALL THICKNESS\', ID = \'thick\' /' + '\n')

		MATL.append('&MATL ID = \'MOISTURE\', N_REACTIONS = 1, A(1) = ' + str(A[0][0]) + ', E(1) = ' + str(E[0][0]) + ', EMISSIVITY = 0.9, DENSITY = 1000, SPEC_ID = \'METHANE\', NU_SPEC = ' + str(1.0) + ', CONDUCTIVITY = 0.2, SPECIFIC_HEAT = 1.0  /' + '\n')
		MATL.append('&MATL ID = \'LB\', N_REACTIONS = 1, A(1) = ' + str(A[0][1]) + ', E(1) = ' + str(E[0][1]) + ', EMISSIVITY = 0.9, DENSITY = ' + str(rho[0]) + ', SPEC_ID = \'METHANE\', NU_SPEC = ' + str(1-nu[0][1]) + ', NU_MATL = ' + str(nu[0][1]) + ', MATL_ID = \'LB_char\', CONDUCTIVITY = 0.2, SPECIFIC_HEAT = 1.0  /' + '\n')
		MATL.append('&MATL ID = \'LB_char\', EMISSIVITY = 0.9, DENSITY = ' + str(rho[1]) + ', CONDUCTIVITY = 0.2, SPECIFIC_HEAT = 1.0  /' + '\n')

		IFILE.append(HEAD)
		IFILE.append(MESH)
		IFILE.append(TIME)
		IFILE.append(MISC)
		IFILE.append(SPEC)
		IFILE.append(MATL)
		IFILE.append(SURF)
		IFILE.append(VENT)
		IFILE.append(DUMP)
		IFILE.append(DEVC)
		IFILE.append(TAIL)
	
		comp_file = chid + '.fds'
		with open(comp_file, mode = 'w', encoding='utf-8') as f:
			for lines in IFILE:
				f.write(''.join(str(line) for line in lines))
				f.write('\n')
			f.close
		return chid

def y_mean(theta0,theta1,theta2):
    
	chid = x.input(theta0,theta1,theta2)
	time.sleep(0.1)
	flag = external_fds.run_fds(chid)
	mlr, nmass = external_fds.read_fds(chid)

	# return the numpy array
	return nmass

# define a theano Op for our likelihood function
class FDS_TGA(tt.Op):

    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """
    itypes = [tt.dscalar, tt.dscalar, tt.dscalar] # expects a vector of parameter values when called
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
        theta0, theta1, theta2, = inputs  # this will contain my variables

        # call the log-likelihood function
        # mlr, nmass = self.y(theta0, theta1, theta2)
        nmass = self.y(theta0, theta1, theta2)

        outputs[0][0] = np.array(nmass) # output the log-likelihood
	
# Generate Model
x = A()
ym = FDS_TGA(y_mean)

ndraws = 1500 # number of draws from the distribution
nburn = 1500 # number of "burn-in points" (which we'll discard)

with mc.Model():
	# Priors
	theta0 = mc.Uniform('theta0', lower=5, upper=25, shape=())
	theta1 = mc.Uniform('theta1', lower=0.5, upper=5, shape=())
	theta2 = mc.Uniform('theta2', lower=0, upper=1, shape=())
						
	sigma = mc.Uniform('sigma', lower=0., upper=10., shape=(1,))

	theta0 = tt.as_tensor_variable(theta0)
	theta1 = tt.as_tensor_variable(theta1)
	theta2 = tt.as_tensor_variable(theta2)
	sigma = tt.as_tensor_variable(sigma)

	db = mc.backends.NDArray('test')

	#Likelihood
	#mlr = ym(theta)
	y_obs = mc.Normal('y_obs', observed=tga_exp.nm, mu=ym(theta0,theta1,theta2), tau=sigma**-2)

	# Configure and run MCMC simulation
	# step = mc.Metropolis(vars = [theta0, theta1, theta2], S=sigma)
	step = mc.Slice()
	trace = mc.sample(ndraws, tune=nburn, discard_tuned_samples=True, chains = 2, cores = 1)
	
trace_sum = mc.summary(trace).round(2)

trace_sum.to_csv('../Figures/fds_vars_sum.csv')

mc.traceplot(trace, var_names = "theta0")
plt.savefig('../Figures/traceplot_theta0.pdf')
plt.close()
mc.traceplot(trace, var_names = "theta1")
plt.savefig('../Figures/traceplot_theta1.pdf')
plt.close()
mc.traceplot(trace, var_names = "theta2")
plt.savefig('../Figures/traceplot_theta2.pdf')
mc.traceplot(trace, var_names = "sigma")
plt.savefig('../Figures/traceplot_sigma.pdf')
plt.close()

mc.forestplot(trace, var_names = "theta0")
plt.savefig('../Figures/forestplot_theta0.pdf')
plt.close()
mc.forestplot(trace, var_names = "theta1")
plt.savefig('../Figures/forestplot_theta1.pdf')
plt.close()
mc.forestplot(trace, var_names = "theta2")
plt.savefig('../Figures/forestplot_theta2.pdf')
mc.forestplot(trace, var_names = "sigma")
plt.savefig('../Figures/forestplot_sigma.pdf')
plt.close()

mc.autocorrplot(trace, var_names = "theta0")
plt.savefig('../Figures/autocorr_theta0.pdf')
plt.close()
mc.autocorrplot(trace, var_names = "theta1")
plt.savefig('../Figures/autocorr_theta1.pdf')
plt.close()
mc.autocorrplot(trace, var_names = "theta2")
plt.savefig('../Figures/autocorr_theta2.pdf')
mc.autocorrplot(trace, var_names = "sigma")
plt.savefig('../Figures/autocorr_sigma.pdf')
plt.close()
plt.show()

back = mc.backends.tracetab.trace_to_dataframe(trace, chains=None, varnames=None, include_transformed=False)	
back.to_csv('../Figures/test1_vars.csv')
print(x.input.calls)
