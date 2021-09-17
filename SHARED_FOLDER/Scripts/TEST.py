import numpy as np
import pymc3 as mc
import theano
import theano.tensor as tt

# for reproducibility here's some version info for modules used in this notebook
import IPython
import matplotlib.pyplot as plt

# define a theano Op for our likelihood function

def my_model(theta, x):
    """
    A straight line!

    Note:
        This function could simply be:

            m, c = thetha
            return m*x + x

        but I've made it more complicated for demonstration purposes
    """
    m, c = theta  # unpack line gradient and y-intercept

    return m*x + c


# define your really-complicated likelihood function that uses loads of external codes
def my_loglike(theta, x, data, sigma):
    """
    A Gaussian log-likelihood function for a model with parameters given in theta
    """

    model = my_model(theta, x)

    return -(0.5/sigma**2)*np.sum((data - model)**2)

class LogLike(tt.Op):

    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """
    itypes = [tt.dvector] # expects a vector of parameter values when called
    otypes = [tt.dscalar] # outputs a single scalar value (the log likelihood)
    
    def __init__(self, loglike, data, x, sigma):
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
        self.likelihood = loglike
        self.data = data
        self.x = x
        self.sigma = sigma

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        theta, = inputs  # this will contain my variables
 
        # call the log-likelihood function
        logl = self.likelihood(theta, self.x, self.data, self.sigma)

        outputs[0][0] = np.array(logl) # output the log-likelihood

# set up our data
N = 10  # number of data points
sigma = 1.  # standard deviation of noise
x = np.linspace(0., 9., N)

mtrue = 0.4  # true gradient
ctrue = 3.   # true y-intercept

truemodel = my_model([mtrue, ctrue], x)

# make data
data = sigma*np.random.randn(N) + truemodel

ndraws = 10000 # number of draws from the distribution
nburn = 2000 # number of "burn-in points" (which we'll discard)

# create our Op
logl = LogLike(my_loglike, data, x, sigma)

# use PyMC3 to sampler from log-likelihood
with mc.Model():
    # uniform priors on m and c
    m = mc.Uniform('m', lower=-10., upper=10.)
    c = mc.Uniform('c', lower=-10., upper=10.)

    # convert m and c to a tensor vector
    theta = tt.as_tensor_variable([m, c])

    # use a DensityDist (use a lamdba function to "call" the Op)
    mc.DensityDist('likelihood', lambda v: logl(v), observed={'v': theta})
    
    trace = mc.sample(ndraws, tune=nburn, discard_tuned_samples=True, chains =10, cores=1)            

# plot the traces
mc.traceplot(trace, lines=(('m', {}, [mtrue]), ('c', {}, [ctrue])))
plt.show()

mc.summary(trace).round(2)
# put the chains in an array (for later!)
samples_pymc3 = np.vstack((trace['m'], trace['c'])).T