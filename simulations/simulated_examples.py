from bisect import bisect_left
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
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from multiprocessing.pool import Pool
from typing import Generator, List, Optional, Tuple
from gpflow.utilities import print_summary
import sys
sys.path.append('..')
from trcd.utils import SamplingHelper
from trcd.kernels import Kernel_mRNA
import pymc3

from utils.utilities import (create_data, load_data, load_single_gene, load_filtered_data, create_standard_mcmc,
                       optimize_with_scipy_optimizer,  fit_rbf, predict_trcd, create_trcd_model,
                        select_parameters, init_hyperparameters, compute_hessian)
from utils.utilities import (HMCParameters, create_data, create_nuts_mcmc, create_standard_mcmc, handle_pool,
                       optimize_with_scipy_optimizer, select_parameters, load_single_gene,init_hyperparameters, predict_trcd, create_mala_mcmc,
                       compute_prior_hyperparameters_variance)


#from utilities import  create_trcd_model

__all__ = ["setup_problem", "setup_full_data"]

Scalar = TypeVar("Scalar", tf.Tensor, float)
Observations = Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
Data = Tuple[tf.Tensor, tf.Tensor]
Data_p = Tuple[tf.Tensor,tf.Tensor]
FullData = Data
Observations = Tuple[tf.Tensor, tf.Tensor, tf.Tensor]

def experiment_print(step: int, msg: str):
    tf.print(f"# [Step {step}] {msg}")


def reset_parameters(parameters: List[gpflow.Parameter], values: List[tf.Tensor]):
    for p, v in zip(parameters, values):
        p.assign(v)

def load_single_gene(gene_id, tr_id) -> Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
    # data = pd.read_csv('density_data_pickedup_cluster10.txt',sep=" ")
    #data = pd.read_csv('density_data_keepdensitysumover10_withcluster.txt',sep=" ")
    data1 = pd.read_csv('../data/LB_GP_TS.csv', sep=",")
    data2 = pd.read_csv('../data/Exon_intron_counts_data_normalizedbylibrarydepthonly_20200120.txt',sep=" ")
    gene_id = gene_id
    data1 = data1[data1['FBtr'] == tr_id]
    data2 = data2[data2['FBgn'] == gene_id]
    data_m = data1[data1['FBtr'] == tr_id].iloc[0][1:31].to_numpy()
    data_p = data2[data2['FBgn'] == gene_id].iloc[0][31:61].to_numpy()#/data2[data2['FBgn'] == gene_id].iloc[0][62]

    #data_p = data_p*1000.0

    t0 = np.array((95.0,105.0,115.0,125.0,145.0,160.0,175.0,190.0,205.0,220.0))
    rep_no = 3
    t = np.hstack((t0,t0,t0))[:,None]

    tp_obs = np.asarray(t)
    ym = np.asarray(data_m, dtype=np.float)
    yp = np.asarray(data_p, dtype=np.float)
    y_full = tf.convert_to_tensor(np.vstack((ym, yp)).reshape(60, 1))
    x_full = tf.convert_to_tensor(np.vstack((tp_obs, tp_obs)).reshape(60, 1))
    full_data = (dfloat(x_full), dfloat(y_full))
    full_data_p = (dfloat(tp_obs), dfloat(yp))
    observations = (dfloat(tp_obs), dfloat(ym), dfloat(yp))
    observations_p = (dfloat(tp_obs), dfloat(yp))
    return full_data, observations, gene_id, full_data_p, observations_p


def plot_trcd_predict(cluster, trcd: tf.Module, tr_id, gene_id, observations: Observations, var_m_noise, var_p_noise, num_predict: int = 100):

    tp_obs, ym, yp = [np.array(o) for o in observations]

    xx = np.linspace(np.min(tp_obs), np.max(tp_obs), num_predict).reshape(num_predict, 1)
    xx_full = np.concatenate((xx, xx)).reshape(-1, 1)
    #
    #gpflow.config.set_default_jitter(0.5)
    try:
        #print('predicted y')
        mean, var = trcd.model.predict_y(xx_full)
    except:
        print('predicted y failed, predicted f')
        mean, var = trcd.model.predict_f(xx_full)

    mean1 = mean[0:num_predict]
    mean2 = mean[num_predict:2 * num_predict]
    var1 = var[0:num_predict]
    var2 = var[num_predict:2 * num_predict]

    # Plot predicted model
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_ylabel('mRNA', color='r', size = 30)
    ax1.plot(tp_obs.flatten(), ym.flatten(), 'rx', mew=2)
    ax1.tick_params(axis='both', which='major', labelsize=30)
    ax1.plot(xx[:, 0], mean1[:, 0], 'r', lw=2)
    ax1.fill_between(xx[:, 0],
                         mean1[:, 0] - 2 * np.sqrt(var1[:, 0])-2* np.sqrt(var_m_noise),
                         mean1[:, 0] + 2 * np.sqrt(var1[:, 0])+2* np.sqrt(var_m_noise),
                         color='red',
                         alpha=0.2)
    ax1.tick_params(axis='both', which='major', labelsize=30)
    ax1.tick_params(axis='both', which='minor', labelsize=30)
    #plt.ylim((-20.0, 60.0))

    ax2 = ax1.twinx()
    ax2.set_ylabel('pre-mRNA', color='b',  size = 30)
    ax2.set_xlabel('Time (mins)',  size = 30)


    ax2.tick_params(axis='both', which='major', labelsize=30)
    ax2.plot(tp_obs.flatten(), yp.flatten(), 'bx', mew=2)
    ax2.plot(xx[:, 0], mean2[:, 0], 'b', lw=2)
    plt.fill_between(xx[:, 0],
                     mean2[:, 0] - 2 * np.sqrt(var2[:, 0])-2* np.sqrt(var_p_noise),
                     mean2[:, 0] + 2 * np.sqrt(var2[:, 0])+2* np.sqrt(var_p_noise),
                     color='blue',
                     alpha=0.2)
    ax2.tick_params(axis='both', which='major', labelsize=30)
    ax2.tick_params(axis='both', which='minor', labelsize=30)
    #plt.ylim((-20.0, 60.0))
    ax1.set_xlabel('Time (mins)', fontsize=30)
    plt.title("Simulated data and model fit", fontsize=40)
    plt.tight_layout()
    #plt.title(cluster+str(gene_id)+' '+str(tr_id), fontsize=18)
    plt.savefig(cluster+'_'+'simulated_data'+'.png')


def dfloat(value):  # default float
    return tf.cast(value, default_float())

def dint(value):  # default float
    return tf.cast(value, gpflow.default_int())


@contextmanager
def handle_pool(pool: Pool) -> Generator[Pool, None, None]:
    try:
        yield pool
    finally:
        pool.close()
        pool.join()

Data = Tuple[tf.Tensor, tf.Tensor]
Data_p = Tuple[tf.Tensor,tf.Tensor]
Initial_D = Tuple[tf.Tensor]
Initial_S = Tuple[tf.Tensor]

Alpha_m = Tuple[tf.Tensor]
Beta_m = Tuple[tf.Tensor]
Alpha_p = Tuple[tf.Tensor]
Beta_p = Tuple[tf.Tensor]
Alpha = Tuple[tf.Tensor]
Beta = Tuple[tf.Tensor]

T_break = Tuple[tf.Tensor]

Initial_lengthscale = Tuple[tf.Tensor]
Initial_variance = Tuple[tf.Tensor]
Initial_variance_m = Tuple[tf.Tensor]
Initial_variance_p = Tuple[tf.Tensor]
Prior_lengthscale_mean = Tuple[tf.Tensor]
Prior_variance_mean = Tuple[tf.Tensor]
FullData = Data
Observations = Tuple[tf.Tensor, tf.Tensor, tf.Tensor]


def run_mcmc(run_chain_fn: Callable,
             hmc_helper: SamplingHelper,
             step_size: Scalar):
    # The `traces` object (it can be a tuple or a namedtuple) contains infromation about `log_prob`     values,
    # accept ratios and etc. E.g. you can get `traces.accepted_results.target_log_prob`.

    unconstrained_samples, traces = run_chain_fn(step_size)
    constrained_samples = hmc_helper.convert_to_constrained_values(*unconstrained_samples)

    # NOTE: not used, but useful
    # def assign_unconstrained_sample(index, helper, samples):
    #     subsample = [values[index] for values in samples]
    #     helper.assign_values(*subsample)

    if isinstance(traces.inner_results, tfp.mcmc.nuts.NUTSKernelResults):
        accept_log_probs = traces.inner_results.target_log_prob
        grads = traces.inner_results.grads_target_log_prob
    else:
        accept_log_probs = traces.inner_results.accepted_results.target_log_prob
        grads = traces.inner_results.accepted_results.grads_target_log_prob
    return unconstrained_samples, constrained_samples, accept_log_probs, grads

def run_mala(run_chain_fn: Callable,
             hmc_helper: SamplingHelper,
             step_size: Scalar):
    # The `traces` object (it can be a tuple or a namedtuple) contains infromation about `log_prob`     values,
    # accept ratios and etc. E.g. you can get `traces.accepted_results.target_log_prob`.

    unconstrained_samples = run_chain_fn(step_size)
    #print(unconstrained_samples)
    constrained_samples = hmc_helper.convert_to_constrained_values(*unconstrained_samples)

    # NOTE: not used, but useful
    # def assign_unconstrained_sample(index, helper, samples):
    #     subsample = [values[index] for values in samples]
    #     helper.assign_values(*subsample)

    #if isinstance(traces.inner_results, tfp.mcmc.nuts.NUTSKernelResults):
    #    accept_log_probs = traces.inner_results.target_log_prob
    #    grads = traces.inner_results.grads_target_log_prob
    #else:
    #    accept_log_probs = traces.inner_results.accepted_results.target_log_prob
    #    grads = traces.inner_results.accepted_results.grads_target_log_prob
    return unconstrained_samples, constrained_samples#, accept_log_probs, grads


def analyse_samples(gene_id, tr_id, parameters_vector , step_size, unconstrained_samples: List[tf.Tensor],
                    constrained_samples: List[tf.Tensor],
                    #accept_log_prob: List[tf.Tensor],
                    #grads: List[tf.Tensor],
                    dict_parameters: dict = None,
                    pathname: str = None):
    """
    Plotting and saving graphs on disk.
    """

    print('analyse_samples loaded')
    path = Path(pathname)
    path.mkdir(exist_ok=True, parents=True)

    parameter_names = list(dict_parameters.keys())
    num_parameters = len(parameter_names)

    num_samples = constrained_samples[0].shape[0]
    x = list(range(num_samples))

    print('analyse_samples loaded 1')
    figsize = (30, 20)
    fig, axes = plt.subplots(num_parameters, 2, figsize=figsize)

    for i in range(num_parameters):
        axes[i][0].plot(x, constrained_samples[i])
        axes[i][1].hist(constrained_samples[i], bins='auto', alpha=0.7, rwidth=0.95)
        axes[i][0].title.set_text(parameter_names[i])


    plt.savefig(f'{pathname}/'+str(gene_id)+'_'+str(tr_id)+'_traces.png')

    res = np.zeros((num_parameters, 2))
    for i in range(num_parameters):
        hpd = pymc3.stats.hpd(np.asarray(constrained_samples[i]))
        res[i,0] = hpd[0]
        res[i,1] = hpd[1]
    print(res)

    res = np.concatenate((np.array(parameter_names).reshape(-1,1), np.array(parameters_vector).reshape(-1,1), res), axis=1)

    print(parameter_names)
    df = pd.DataFrame(res)
    print(df)

    df.columns = ['parameter','MAP','hpd_l', 'hpd_u']

    print(df)
    df = df.round(5)
    df.to_csv(f'{pathname}/hpd'+str(gene_id)+'_'+str(tr_id)+'.csv')

    save_samples = pd.DataFrame(np.asarray(constrained_samples))
    save_samples.to_csv(f'{pathname}/samples_'+str(gene_id)+str(tr_id)+'step_size'+str(step_size)+'.csv')


def experiment_print(step: int, msg: str):
    tf.print(f"# [Step {step}] {msg}")


def reset_parameters(parameters: List[gpflow.Parameter], values: List[tf.Tensor]):
    for p, v in zip(parameters, values):
        p.assign(v)

def dfloat(value):  # default float
    return tf.cast(value, default_float())

def dint(value):  # default float
    return tf.cast(value, gpflow.default_int())


@contextmanager
def handle_pool(pool: Pool) -> Generator[Pool, None, None]:
    try:
        yield pool
    finally:
        pool.close()
        pool.join()

def fit_rbf(data, init_lengthscale, init_variance):
    k = gpflow.kernels.RBF()
    #m = gpflow.models.GPR(data, kernel=k,mean_function=gpflow.mean_functions.Constant(c=None))
    m = gpflow.models.GPR(data, kernel=k)
    m.likelihood.variance.assign(0.01)
    m.kernel.lengthscales.assign(init_lengthscale)
    m.kernel.variance.assign(init_variance)
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(m.training_loss,
                            m.trainable_variables,
                            options=dict(maxiter=100))
    return m, np.asarray(m.training_loss())

gene_id = 'FBgn0000490'#
tr_id = 'FBtr0077775'  #

data, observations, gene_id, data_p, observations_p = load_single_gene(gene_id, tr_id)

f_sample = np.asarray(data_p[1])

t0 = np.array((95.0,105.0,115.0,125.0,145.0,160.0,175.0,190.0,205.0,220.0))
rep_no = 3
t = np.hstack((t0,t0,t0))[:,None]
tp_full = np.asarray(t)


# Estimate a single GP based on pre-mRNA data
data_full = (dfloat(np.asarray(data_p[0]).reshape(-1,1)), dfloat(np.asarray(data_p[1]).reshape(-1,1)))
#print(data_full)

init_lengthscale = 10.0
init_variance = 1.0
m, lik = fit_rbf(data_full, init_lengthscale, init_variance)
print_summary(m)

xx = np.linspace(95.0, 220.0, 100).reshape(100, 1)  # test points must be of shape (N, D)

## predict mean and variance of latent GP at test points
mean, var = m.predict_f(xx)

f_sample, var = m.predict_f(t0.reshape(-1, 1))

t0 = t0.reshape(-1, 1)
S = 0.3
D = 0.05
variance = 69.0
lengthscale = 33.0
k_trcd = Kernel_mRNA(S=S, D=D, variance=variance,lengthscale=lengthscale)

xx = np.stack((t0,t0)).reshape(-1,1)

n = int(k_trcd(xx, xx).shape[0]/2)
K = k_trcd(xx, xx)

Km = K[0:n,0:n]
Kmp = K[0:n,n:2*n]
Kpm = K[n:2*n,0:n]
Kp = K[n:2*n,n:2*n]

fm_mean = Kmp@np.linalg.inv(Kp)@f_sample
fm_covar = Km-Kmp@np.linalg.inv(Kp)@Kpm

#print(fm_mean)

var_m = 50.5
var_p = 3.5

ym1 = fm_mean + np.random.normal(0.0, np.sqrt(var_m), 1)
ym2 = fm_mean + np.random.normal(0.0, np.sqrt(var_m), 1)
ym3 = fm_mean + np.random.normal(0.0, np.sqrt(var_m), 1)

yp1 = f_sample + np.random.normal(0.0, np.sqrt(var_p), 1)
yp2 = f_sample + np.random.normal(0.0, np.sqrt(var_p), 1)
yp3 = f_sample + np.random.normal(0.0, np.sqrt(var_p), 1)


### ### ### ###
# Combine the data
### ### ### ###
t = np.stack((t0,t0,t0,t0,t0,t0)).reshape(-1,1)
y = np.stack((ym1,ym2,ym3,yp1,yp2,yp3)).reshape(-1,1)

ym = np.stack((ym1,ym2,ym3)).flatten()
yp =np.stack((yp1,yp2,yp3)).flatten()

data = (dfloat(t),dfloat(y))

tp_obs = np.stack((t0,t0,t0)).reshape(-1,1)
observations = (dfloat(tp_obs.reshape(-1,1)), dfloat(ym), dfloat(yp))

initial_D = np.random.uniform(0.001,1.0,1)[0]
initial_S = np.random.uniform(0.1,1.5,1)[0]
initial_lengthscale = np.random.uniform(10.0,20.0,1)[0]
initial_lengthscale = m.kernel.lengthscales.value()
initial_variance = np.random.uniform(1.0,50.5,1)[0]
initial_variance = m.kernel.variance.value()

print('initial_variance', initial_variance)

trcd, dict_parameters = create_trcd_model(data, initial_lengthscale, initial_variance, initial_S, initial_D, transform_base=None)
dict_parameters = select_parameters(dict_parameters,
                                    names=None)  # When `names` is None, same dictionary is returned.

initial_D = 0.01
initial_S = 0.3

initial_lengthscale = 15.2
initial_variance = 60.5

trcd.model.kernel.D.assign(initial_D)
trcd.model.kernel.S.assign(initial_S)
trcd.model.kernel.lengthscale.assign(initial_lengthscale)
trcd.model.kernel.variance.assign(initial_variance)
trcd.model.likelihood.variance_m.assign(var_m)
trcd.model.likelihood.variance_p.assign(var_p)

parameters = list(dict_parameters.values())

print_summary(trcd)

optimize_with_scipy_optimizer(trcd, parameters)  # NOTE: Updates TRCD model parameters in place!

print_summary(trcd)

parameters_estimates = list(dict_parameters.values())

var_m_noise = trcd.model.likelihood.variance_m.value()
var_p_noise = trcd.model.likelihood.variance_p.value()

#trcd.model.kernel.S.assign(0.45)
plot_trcd_predict('D_'+str(D), trcd, tr_id, gene_id, observations, var_m_noise, var_p_noise)

## NOTE: Static parameters for the MCMC.
##
hmc_parameters = HMCParameters(num_samples=1000, num_burnin_steps=1000)
#hmc_helper, run_chain = create_standard_mcmc(trcd, parameters)
hmc_helper, run_chain = create_mala_mcmc(trcd, parameters, hmc_parameters)

gpflow.config.set_default_positive_minimum(1e-6)
#step_size_mala = np.array([0.02,0.04, 0.06])#
step_size_mala = np.array([10**-5 , 10**-4 , 10**-3 , 0.003, 0.005, 0.001, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.5, 1])#
#step_size_mala = np.array([0.001, 0.002, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15])#
#step_size_mala = np.array([0.03])#

parameters_vector = tf.stack(parameters)
#print_summary(trcd)
#print('observations', observations)
#exit()
print(f"log posterior density at optimum: {trcd.model.log_posterior_density()}")
exit()
#print(step_size_mala)
for j in range(step_size_mala.shape[0]):
    step_size = step_size_mala[j]

    try:
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
    except:
        print('Experiment error')
