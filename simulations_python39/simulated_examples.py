import sys
sys.path.append('..')

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from multiprocessing.pool import Pool
from typing import Generator, List, Optional, Tuple
from typing import Callable, List, TypeVar
from multiprocessing.pool import Pool
from pathlib import Path
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import mcmc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gpflow
from gpflow.config import default_float
import pandas as pd
from gpflow.utilities import print_summary
from trcd.utils import SamplingHelper
import pymc3
import arviz as az

from utilities import (create_standard_mcmc, optimize_with_scipy_optimizer,
                             fit_rbf, create_trcd_model,select_parameters, init_hyperparameters, compute_hessian,
                             HMCParameters, create_mala_mcmc)

from utils_simulations import (load_single_gene, plot_trcd_predict, generate_data)

from utils_mcmc import (run_mcmc, run_mala, analyse_samples, experiment_print, reset_parameters, dfloat)

# If you run the code on Mac M1, gpu might actually be slowing does the MCMC part
# If that's the case, try uncommenting the following line
tf.config.experimental.set_visible_devices([], 'GPU')

'''
For running MCMC set run_mcmc = True. Running MCMC might take a while.
'''
run_mcmc =  False
def dfloat(value):  # default float
    return tf.cast(value, default_float())

def dint(value):  # default float
    return tf.cast(value, gpflow.default_int())
'''
Parameters for data generation;
Change only S (splicing rate) and D (degradation rate);
'''

gene_id = 'FBgn0000490'
tr_id = 'FBtr0077775'

# Set model parameters for data generation
S = 0.3
D = 0.05 # 0.003, 0.008, 0.01, 0.02, 0.05 -- values from the simulation study
var_m = 50.5 # noise var for mRNA
var_p = 3.5  # noise var for pre-mRNA

t0, t, y, ym, yp, m = generate_data(gene_id, tr_id, S, D, var_m, var_p)
# t0 - time points in which we have observations
# t - time points for the replicates
# y - stacked generated data for mRNA and pre-mRNA
# ym, yp - mRNA and pre-mRNA observations correspondingly
# m - model for a single Gaussian process regression fitted jus ton pre-mRNA data

'''
Initialize parameter values
'''
initial_D = 0.1
initial_S = 0.3
initial_lengthscale = 33.0
initial_variance = 69.0


'''
Create transcriptional regulation model
'''

data = (dfloat(t),dfloat(y))
trcd, dict_parameters = create_trcd_model(data, initial_lengthscale, initial_variance, initial_S, initial_D, transform_base=None)
dict_parameters = select_parameters(dict_parameters,
                                    names=None)  # When `names` is None, same dictionary is returned.

## Alternatively, you can assign values of the parameters after the model is initialized  as:
#trcd.model.kernel.D.assign(initial_D)
#trcd.model.kernel.S.assign(initial_S)
#trcd.model.kernel.lengthscale.assign(initial_lengthscale)
#trcd.model.kernel.variance.assign(initial_variance)
#trcd.model.likelihood.variance_m.assign(var_m)
#trcd.model.likelihood.variance_p.assign(var_p)

'''
Optimize the model
'''

parameters = list(dict_parameters.values())
optimize_with_scipy_optimizer(trcd, parameters)  # NOTE: Updates TRCD model parameters in place!
print_summary(trcd)

parameters_estimates = list(dict_parameters.values())

var_m_noise = np.asarray(trcd.model.likelihood.variance_m).flatten()[0]
var_p_noise = np.asarray(trcd.model.likelihood.variance_p).flatten()[0]

# Plot model fit
tp_obs = np.stack((t0,t0,t0)).reshape(-1,1)
observations = (dfloat(tp_obs.reshape(-1,1)), dfloat(ym), dfloat(yp))
plot_trcd_predict('D_'+str(D), trcd, tr_id, gene_id, observations, var_m_noise, var_p_noise)

'''
Run MCMC; only runs if run_mcmc == True.
'''
if(run_mcmc == True):
    ## NOTE: Static parameters for the MCMC.
    ##
    hmc_parameters = HMCParameters(num_samples=100000, num_burnin_steps=1000)
    hmc_helper, run_chain = create_mala_mcmc(trcd, parameters, hmc_parameters)

    gpflow.config.set_default_positive_minimum(1e-6)
    step_size_mala = np.array([10**-5 , 10**-4 , 10**-3 , 0.003, 0.005, 0.001, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.5, 1])#
    parameters_vector = tf.stack(parameters)

    print(f"log posterior density at optimum: {trcd.model.log_posterior_density()}")

    for j in range(step_size_mala.shape[0]):
        step_size = step_size_mala[j]

        #try:
        experiment_print(j, "Run MCMC")
        #num_leapfrog, step_size = leapfrog_num_and_step_size[j]
        pathname = f"results mala D -{str(D)}-{gene_id}-{tr_id}-{step_size}"
        experiment_print(j, f"Save results at '{pathname}'")

        samples = run_mala(run_chain,
                           hmc_helper,
                           step_size=step_size_mala[j])

        print('Got through')

        analyse_samples(gene_id, tr_id, parameters_vector, step_size, *samples, dict_parameters=dict_parameters, pathname=pathname)
        reset_parameters(parameters, parameters_estimates)
        experiment_print(j, f"MCMC finished")
        #except:
        #    print('Experiment error, likely step size was not optimal')
