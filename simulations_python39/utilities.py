from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from multiprocessing.pool import Pool
from typing import Generator, List, Optional, Tuple

import matplotlib.pyplot as plt
#from sklearn.preprocessing import normalize
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import mcmc

import pandas as pd
import gpflow
from gpflow.config import default_float
import sys
sys.path.append('..')
from trcd.model import TRCD
from trcd.synthetic_data import setup_full_data, setup_problem
from trcd.utils import SamplingHelper
from gpflow.utilities import print_summary

import pandas as pd

__all__ = [
    "create_data", "create_trcd_model", "HMCParameters", "optimize_with_scipy_optimizer", "create_standard_mcmc",
    "create_nuts_mcmc", "handle_pool"
]

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

def compute_hessian(model, parameters):
    """
    Computes the hessian
    """

    variables = [p.unconstrained_variable for p in parameters]

    @tf.function
    def grads():
        with tf.GradientTape() as tape1:
            tape1.watch(variables)
            with tf.GradientTape() as tape2:
                tape2.watch(variables)
                objective = -model.model.training_loss()
        # WARN: If you change this computation from constrained to unconstrained
        #       you will have to update the rest of computations respectively.
        # NOTE: grad w.r.t. constrained parameters:
        #    dy_dx = tape2.gradient(objective, parameters)
        #d2y_dx2 = tape1.gradient(dy_dx, parameters)
        # NOTE: grad w.r.t. UNconstrained parameters:
            dy_dx = tape2.gradient(objective, variables)
        d2y_dx2 = tape1.gradient(dy_dx, variables)
        return d2y_dx2

    return grads()

def init_hyperparameters(X):
    initial_lengthscale = np.random.uniform(0. , (np.max(X)-np.min(X))/10.)
    initial_variance = np.random.uniform(1., 100.)
    return initial_lengthscale, initial_variance

def distance_from_mean(model, observations, num_predict: int = 100):
    tp_obs, y = [np.array(o) for o in observations]
    t0 = np.array((95.0,105.0,115.0,125.0,145.0,160.0,175.0,190.0,205.0,220.0))
    xx = t0.reshape(-1,1)
    mean, var = model.predict_f(xx)
    mean_y = np.mean(y, axis=0)
    dist_mean = 0.1 * np.sum(((mean_y - mean[:, 0])**2))
    dist_var = 0.1 * np.sum(((mean_y - (mean[:, 0] + 2 * np.sqrt(var[:, 0])))**2))
    return dist_mean , dist_var

#################################################################################
# Functions for filtering genes
#################################################################################
def fit_rbf(data, init_lengthscale, init_variance):
    k = gpflow.kernels.RBF()
    m = gpflow.models.GPR(data, kernel=k)
    m.likelihood.variance.assign(0.01)
    m.kernel.lengthscales.assign(init_lengthscale)
    m.kernel.variance.assign(init_variance)
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(m.training_loss,
                            m.trainable_variables,
                            options=dict(maxiter=100))
    return m, np.asarray(m.training_loss())

# Alternative version
#def fit_rbf(data, init_lengthscale, init_variance):
#    k = gpflow.kernels.RBF()
#    m = gpflow.models.GPR(data, kernel=k)
#    m.likelihood.variance.assign(0.01)
#    def objective_closure():
#        return m.training_loss()
#    m.kernel.lengthscales.assign(init_lengthscale)
#    m.kernel.variance.assign(init_variance)
#    opt = gpflow.optimizers.Scipy()
#    opt_logs = opt.minimize(objective_closure,
#                            m.trainable_variables,
#                            options=dict(maxiter=100))
#    return m, np.asarray(m.training_loss())

def fit_noise(data, init_variance):
    k = gpflow.kernels.White()
    m = gpflow.models.GPR(data, kernel=k, mean_function=gpflow.mean_functions.Constant(c=None))
    m.likelihood.variance.assign(0.01)
    m.kernel.variance.assign(init_variance)
    opt = gpflow.optimizers.Scipy()
    def objective_closure():
        return m.training_loss()
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(objective_closure,
                            m.trainable_variables,
                            options=dict(maxiter=100))
    return m, np.asarray(m.training_loss())

def fit_noise_rbf(data, init_variance):
    k = gpflow.kernels.RBF()
    m = gpflow.models.GPR(data, kernel=k, mean_function=gpflow.mean_functions.Constant(c=None))
    m.likelihood.variance.assign(0.01)
    m.kernel.variance.assign(init_variance)
    m.kernel.lengthscales.assign(1000000.0)
    gpflow.utilities.set_trainable(m.kernel.lengthscales, False)
    opt = gpflow.optimizers.Scipy()
    def objective_closure():
        return m.training_loss()
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(objective_closure,
                            m.trainable_variables,
                            options=dict(maxiter=100))
    return m, np.asarray(m.training_loss())

'''
Functions for loading data;

'''


def load_data(data_i) -> Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
    i=data_i

    data = pd.read_csv('density_data_pickedup_cluster10.txt',sep=" ")
    data  = data.loc[data['cluster'] == 3]
    gene_id = data.FBgn.iloc[i]
    data_m = data.iloc[i][1:31].to_numpy() # exones
    data_p = data.iloc[i][31:61].to_numpy() # intons

    t0 = np.array((95.0,105.0,115.0,125.0,145.0,160.0,175.0,190.0,205.0,220.0))
    rep_no = 3
    t = np.hstack((t0,t0,t0))[:,None]

    #tp_all, f_all = setup_full_data()
    tp_obs = np.asarray(t)
    ym = np.asarray(data_m, dtype=np.float)
    yp = np.asarray(data_p, dtype=np.float)
    y_full = tf.convert_to_tensor(np.vstack((ym, yp)).reshape(60, 1))
    x_full = tf.convert_to_tensor(np.vstack((tp_obs, tp_obs)).reshape(60, 1))
    full_data = (dfloat(x_full), dfloat(y_full))
    #full_data_p = (dfloat(tp_obs), dfloat(yp/np.linalg.norm(yp)))
    full_data_p = (dfloat(tp_obs), dfloat(yp))
    observations = (dfloat(tp_obs), dfloat(ym), dfloat(yp))
    #observations_p = (dfloat(tp_obs), dfloat(yp/np.linalg.norm(yp)))
    observations_p = (dfloat(tp_obs), dfloat(yp))
    return full_data, observations, gene_id, full_data_p, observations_p



def load_gene_level_data(gene_id) -> Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
    data1 = pd.read_csv('../data/LB_genes_from_tr.csv',sep=",")
    data2 = pd.read_csv('../data/Exon_intron_counts_data_normalizedbylibrarydepthonly_20200120.txt',sep=" ")

    gene_id = gene_id
    data1 = data1[data1['gene'] == gene_id]
    data2 = data2[data2['FBgn'] == gene_id]

    data_m = data1[data1['gene'] == gene_id].iloc[0][1:31].to_numpy()
    data_p = data2[data2['FBgn'] == gene_id].iloc[0][31:61].to_numpy()

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


def create_data() -> Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
    tp_all, f_all = setup_full_data()
    tp_obs, ym, yp = setup_problem(tp_all, f_all)
    y_full = tf.convert_to_tensor(np.vstack((ym, yp)).reshape(20, 1))
    x_full = tf.convert_to_tensor(np.vstack((tp_obs, tp_obs)).reshape(20, 1))
    full_data = (dfloat(x_full), dfloat(y_full))
    observations = (dfloat(tp_obs), dfloat(ym), dfloat(yp))
    return full_data, observations


def load_single_gene_scaled(gene_id, tr_id) -> Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:

    data1 = pd.read_csv('../data/LB_GP_TS.csv', sep=",")
    data2 = pd.read_csv('../data/Exon_intron_counts_data_normalizedbylibrarydepthonly_20200120.txt',sep=" ")

    gene_id = gene_id
    data1 = data1[data1['FBtr'] == tr_id]
    data2 = data2[data2['FBgn'] == gene_id]
    #genes = pd.read_csv('genes_Yuliya_full.csv',sep=" ")
    data_m = data1[data1['FBtr'] == tr_id].iloc[0][1:31].to_numpy()
    data_p = data2[data2['FBgn'] == gene_id].iloc[0][31:61].to_numpy()/data2[data2['FBgn'] == gene_id].iloc[0][62]

    data_p = data_p*1000.0

    data_m = data_m/10.0
    data_p = data_p/10.0

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
    observations_m = (dfloat(tp_obs), dfloat(ym))
    return full_data, observations, gene_id, full_data_p, observations_p


def load_single_gene(gene_id, tr_id) -> Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
    data1 = pd.read_csv('../data/LB_GP_TS.csv', sep=",")
    data2 = pd.read_csv('../data/Exon_intron_counts_data_normalizedbylibrarydepthonly_20200120.txt',sep=" ")

    gene_id = gene_id
    data1 = data1[data1['FBtr'] == tr_id]
    data2 = data2[data2['FBgn'] == gene_id]
    data_m = data1[data1['FBtr'] == tr_id].iloc[0][1:31].to_numpy()
    data_p = data2[data2['FBgn'] == gene_id].iloc[0][31:61].to_numpy()/data2[data2['FBgn'] == gene_id].iloc[0][62]

    data_p = data_p*1000.0

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
    observations_m = (dfloat(tp_obs), dfloat(ym))
    return full_data, observations, gene_id, full_data_p, observations_p


def create_trcd_model(data: Data, initial_lengthscale: Initial_lengthscale, initial_variance: Initial_variance,initial_S: Initial_S,initial_D: Initial_D,
                       transform_base: Optional[tfp.bijectors.Bijector] = None, protein = False) -> Tuple[TRCD, List[gpflow.Parameter]]:
    """
    Creates a TRCD model with initial parameters
    """
    mean_var_m = np.mean(np.var(np.asarray(data[1][0:30]).reshape(3,10), axis = 0))
    mean_var_p = np.mean(np.var(np.asarray(data[1][30:60]).reshape(3,10), axis = 0))
    max_val_p = np.max(data[1][30:60])
    min_val_p = np.min(data[1][30:60])


    alpha_m, beta_m = compute_prior_hyperparameters_variance(mean_var_m, 0.2*mean_var_m)
    alpha_p, beta_p = compute_prior_hyperparameters_variance(mean_var_p, 0.2*mean_var_p)
    alpha, beta = compute_prior_hyperparameters_variance(initial_variance, 0.5*initial_variance)
    alpha_l, beta_l = compute_prior_hyperparameters_variance(initial_lengthscale, 0.5*initial_lengthscale)

    initial_variance_m = mean_var_m
    initial_variance_p = mean_var_p
    initial_variance = max_val_p

    S_prior = tfd.Gamma(concentration=dfloat(0.7), rate=dfloat(2.0))
    D_prior = tfd.Gamma(concentration=dfloat(2.0), rate=dfloat(2.0))

    variance_m_prior = tfd.Gamma(concentration=dfloat(alpha_m), rate=dfloat(beta_m))
    variance_p_prior = tfd.Gamma(concentration=dfloat(alpha_p), rate=dfloat(beta_p))

    transform = gpflow.utilities.positive if transform_base is None else transform_base

    S = gpflow.Parameter(initial_S, transform=transform(), prior=S_prior)
    D = gpflow.Parameter(initial_D, transform=transform(), prior=D_prior)


    variance_m = gpflow.Parameter(initial_variance_m, transform=transform(), prior=variance_m_prior)
    variance_p = gpflow.Parameter(initial_variance_p, transform=transform(), prior=variance_p_prior)

    set_kernel_prior = True
    if set_kernel_prior:

        variance = gpflow.Parameter(initial_variance,
                                   transform=transform(),
                                   prior=tfd.Gamma(concentration=dfloat(alpha), rate=dfloat(beta)))

        lengthscale = gpflow.Parameter(initial_lengthscale,
                                       transform=transform(),
                                       prior=tfd.Gamma(concentration=dfloat(alpha_l), rate=dfloat(beta_l)))

    else:
        variance = gpflow.Parameter(initial_variance, transform=transform())
        lengthscale = gpflow.Parameter(initial_lengthscale, transform=transform())

    parameters = OrderedDict(D=D,
                             S=S,
                             variance=variance,
                             lengthscale=lengthscale,
                             variance_m = variance_m,
                             variance_p = variance_p
                             )

    return TRCD(data, **parameters), parameters

'''
Optimization/CI functions
'''
def optimize_with_scipy_optimizer(trcd: TRCD, parameters: List[gpflow.Parameter], maxiter: int = 100):
    opt = gpflow.optimizers.Scipy()
    loss_cb = tf.function(trcd.model.training_loss, autograph=False)

    def step_cb(step: int, variables: List[tf.Tensor], values: List[tf.Tensor]):
        if step % 10 == 0:
            [variable.assign(value) for variable, value in zip(variables, values)]
            print(f"Step {step} loss={loss_cb()}")

    variables = [v.unconstrained_variable for v in parameters]
    return opt.minimize(loss_cb, variables, step_callback=step_cb, options=dict(maxiter=maxiter))

def select_parameters(dict_parameters: OrderedDict, names: Optional[List[str]] = None) -> OrderedDict:
    """
    Selects parameters from dictionary according to the names list.
    If `names` is None, same dictionary is returned.
    """
    if names is None:
        return dict_parameters
    for k in list(dict_parameters.keys()):
        if k not in names:
            dict_parameters.pop(k)
    return dict_parameters


def compute_prior_hyperparameters_variance(mean, variance):
    # mean equal to variance of empirical standard deviation
    beta = mean/variance
    alpha = mean**2/variance
    return alpha, beta


def plot_trcd_predict(cluster,  trcd: tf.Module, tr_id, gene_id, observations: Observations, var_m_noise, var_p_noise, num_predict: int = 100):

    tp_obs, ym, yp = [np.array(o) for o in observations]

    xx = np.linspace(np.min(tp_obs), np.max(tp_obs), num_predict).reshape(num_predict, 1)
    xx1 = np.linspace(np.min(tp_obs), np.max(tp_obs), num_predict).reshape(num_predict, 1)
    xx_full = np.concatenate((xx, xx1)).reshape(-1, 1)
    
    mean, var = trcd.model.predict_f(xx_full)

    mean1 = mean[0:num_predict]
    mean2 = mean[num_predict:2 * num_predict]
    var1 = var[0:num_predict]
    var2 = var[num_predict:2 * num_predict]

    # Plot predicted model
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_ylabel('mRNA', color='r', size = 22)
    ax1.plot(tp_obs.flatten(), ym.flatten(), 'rx', mew=2)
    ax1.tick_params(axis='both', which='major', labelsize=22)
    ax1.plot(xx[:, 0], mean1[:, 0], 'r', lw=2)
    ax1.fill_between(xx[:, 0],
                         mean1[:, 0] - 2 * np.sqrt(var1[:, 0])-2* np.sqrt(var_m_noise),
                         mean1[:, 0] + 2 * np.sqrt(var1[:, 0])+2* np.sqrt(var_m_noise),
                         color='red',
                         alpha=0.2)
    ##plt.ylim((-20.0, 60.0))

    ax2 = ax1.twinx()
    ax2.set_ylabel('pre-mRNA', color='b',  size = 22)
    ax2.set_xlabel('Time (mins)',  size = 22)


    ax2.tick_params(axis='both', which='major', labelsize=22)
    ax2.plot(tp_obs.flatten(), yp.flatten(), 'bx', mew=2)
    ax2.plot(xx[:, 0], mean2[:, 0], 'b', lw=2)
    plt.fill_between(xx[:, 0],
                     mean2[:, 0] - 2 * np.sqrt(var2[:, 0])-2* np.sqrt(var_p_noise),
                     mean2[:, 0] + 2 * np.sqrt(var2[:, 0])+2* np.sqrt(var_p_noise),
                     color='blue',
                     alpha=0.2)
    ##plt.ylim((-20.0, 60.0))
    plt.xlabel('Time (mins)', size = 22)
    plt.title(str(gene_id)+' '+str(tr_id), fontsize=22)
    plt.savefig(f"../output/fit_figures/"+str(cluster)+'_'+str(gene_id)+'_'+str(tr_id)+'.png')


def predict_trcd(trcd: tf.Module, gene_id, observations: Observations, num_predict: int = 100):

    tp_obs, ym, yp = [np.array(o) for o in observations]
    t0 = np.array((95.0,105.0,115.0,125.0,145.0,160.0,175.0,190.0,205.0,220.0))
    #xx = np.linspace(np.min(tp_obs), np.max(tp_obs), num_predict).reshape(num_predict, 1)
    #xx_full = np.concatenate((tp_obs, tp_obs)).reshape(-1, 1)
    xx_full = np.concatenate((t0, t0)).reshape(-1, 1)
    num_predict = t0.shape[0]
    mean, var = trcd.model.predict_f(xx_full)
    mean1 = mean[0:num_predict]
    mean2 = mean[num_predict:2 * num_predict]
    var1 = var[0:num_predict]
    var2 = var[num_predict:2 * num_predict]

    ym = ym.reshape(3,10)
    yp = yp.reshape(3,10)
    mean_ym = np.mean(ym, axis=0)
    mean_yp = np.mean(yp, axis=0)
    dist_mean = 0.05 * np.sum(((mean_ym - mean1[:, 0])**2 + (mean_yp - mean2[:, 0])**2))
    dist_var = 0.05 * np.sum(((mean_ym - (mean1[:, 0] + 2 * np.sqrt(var1[:, 0])))**2 + (mean_yp - (mean2[:, 0]+2 * np.sqrt(var2[:, 0])))**2))
    return dist_mean , dist_var


'''
MCMC functions
'''

@dataclass
class HMCParameters:
    accept_prob: tf.Tensor = field(default_factory=lambda: tf.cast(0.75, default_float()))
    adaptation_rate: tf.Tensor = field(default_factory=lambda: tf.cast(0.01, default_float()))
    num_burnin_steps: int = 50
    num_adaptation_steps: int = int(num_burnin_steps * 0.75)  # TODO(@awav) is that too much?
    num_leapfrog_steps: int = 20
    num_samples: int = 500

    # TODO: passed to the run_chain function
    # step_size = tf.cast(0.01, default_float())
    # num_leapfrog = tf.cast(20, default_float())


def create_sampling_helper(trcd, state_variables, hmc_parameters):
    hmc_parameters = HMCParameters() if hmc_parameters is None else hmc_parameters  # NOTE: Hardcoded values!
    hmc_helper = SamplingHelper(lambda: -trcd.model.training_loss(), state_variables)
    return hmc_helper, hmc_parameters


def create_nuts_mcmc(trcd, state_variables: List[gpflow.Parameter], hmc_parameters: Optional[HMCParameters] = None):
    hmc_helper, hmc_parameters = create_sampling_helper(trcd, state_variables, hmc_parameters)
    target_fn = hmc_helper.make_posterior_log_prob_fn()

    @tf.function
    def run_chain_fn(num_leapfrog_steps, step_size):
        step_size = tf.cast(step_size, default_float())
        hmc = tfp.mcmc.NoUTurnSampler(target_fn, step_size, unrolled_leapfrog_steps=num_leapfrog_steps)
        adaptive_hmc = mcmc.SimpleStepSizeAdaptation(
            hmc,
            num_adaptation_steps=hmc_parameters.num_adaptation_steps,
            target_accept_prob=hmc_parameters.accept_prob,
            adaptation_rate=hmc_parameters.adaptation_rate,
            step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(step_size=new_step_size),
            step_size_getter_fn=lambda pkr: pkr.step_size,
            log_accept_prob_getter_fn=lambda pkr: pkr.log_accept_ratio,
        )

        return mcmc.sample_chain(num_results=hmc_parameters.num_samples,
                                 num_burnin_steps=hmc_parameters.num_burnin_steps,
                                 current_state=state_variables,
                                 kernel=adaptive_hmc)

    return hmc_helper, run_chain_fn


def create_standard_mcmc(trcd: TRCD,
                         state_variables: List[gpflow.Parameter],
                         hmc_parameters: Optional[HMCParameters] = None):
    hmc_helper, hmc_parameters = create_sampling_helper(trcd, state_variables, hmc_parameters)
    target_fn = hmc_helper.make_posterior_log_prob_fn()

    @tf.function
    def run_chain_fn(num_leapfrog_steps, step_size):
        """
        Run HMC on created model
        """

        hmc = mcmc.HamiltonianMonteCarlo(target_log_prob_fn=target_fn,
                                         num_leapfrog_steps=num_leapfrog_steps,
                                         step_size=step_size)

        adaptive_hmc = mcmc.SimpleStepSizeAdaptation(hmc,
                                                     num_adaptation_steps=hmc_parameters.num_adaptation_steps,
                                                     target_accept_prob=hmc_parameters.accept_prob,
                                                     adaptation_rate=hmc_parameters.adaptation_rate)

        return mcmc.sample_chain(num_results=hmc_parameters.num_samples,
                                 num_burnin_steps=hmc_parameters.num_burnin_steps,
                                 current_state=state_variables,
                                 kernel=adaptive_hmc)

    return hmc_helper, run_chain_fn


def create_mala_mcmc(trcd: TRCD,
                         state_variables: List[gpflow.Parameter],
                         step_size,
                         hmc_parameters: Optional[HMCParameters] = None):
    hmc_helper, hmc_parameters = create_sampling_helper(trcd, state_variables, hmc_parameters)
    target_fn = hmc_helper.make_posterior_log_prob_fn()

    @tf.function
    def run_chain_fn(step_size):
        """
        Run HMC on created model
        """

        #hmc = mcmc.HamiltonianMonteCarlo(target_log_prob_fn=target_fn,
        #                                 num_leapfrog_steps=num_leapfrog_steps,
        #                                 step_size=step_size)

        #adaptive_hmc = mcmc.SimpleStepSizeAdaptation(hmc,
        #                                             num_adaptation_steps=hmc_parameters.num_adaptation_steps,
        #                                             target_accept_prob=hmc_parameters.accept_prob,
        #                                             adaptation_rate=hmc_parameters.adaptation_rate)

        mala = mcmc.MetropolisAdjustedLangevinAlgorithm(target_log_prob_fn=target_fn, step_size=step_size)

        adaptive_mala = mcmc.SimpleStepSizeAdaptation(mala,
                                                     num_adaptation_steps=int(0.75*2000),
                                                     target_accept_prob=0.75,
                                                     adaptation_rate=0.1)
                                                     #adaptation_rate=0.01)

        return mcmc.sample_chain(num_results=100000,
                                 num_burnin_steps=20000,
                                 current_state=state_variables,
                                 kernel=mala, trace_fn=None)


    return hmc_helper, run_chain_fn

def create_standard_mcmc(trcd: TRCD,
                         state_variables: List[gpflow.Parameter],
                         hmc_parameters: Optional[HMCParameters] = None):
    hmc_helper, hmc_parameters = create_sampling_helper(trcd, state_variables, hmc_parameters)
    target_fn = hmc_helper.make_posterior_log_prob_fn()

    @tf.function
    def run_chain_fn(num_leapfrog_steps, step_size):
        """
        Run HMC on created model
        """

        hmc = mcmc.HamiltonianMonteCarlo(target_log_prob_fn=target_fn,
                                         num_leapfrog_steps=num_leapfrog_steps,
                                         step_size=step_size)

        adaptive_hmc = mcmc.SimpleStepSizeAdaptation(hmc,
                                                     num_adaptation_steps=hmc_parameters.num_adaptation_steps,
                                                     target_accept_prob=hmc_parameters.accept_prob,
                                                     adaptation_rate=hmc_parameters.adaptation_rate)

        return mcmc.sample_chain(num_results=hmc_parameters.num_samples,
                                 num_burnin_steps=hmc_parameters.num_burnin_steps,
                                 current_state=state_variables,
                                 kernel=adaptive_hmc)

    return hmc_helper, run_chain_fn
