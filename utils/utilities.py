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
from trcd.model import TRCD
from trcd.model_nb import TRCD_nb
from trcd.model import m_premRNA
from trcd.synthetic_data import setup_full_data, setup_problem
from trcd.utils import SamplingHelper
from gpflow.utilities import print_summary

import pandas as pd

gpflow.default_jitter()

__all__ = [
    "create_data", "create_trcd_model", "HMCParameters", "optimize_with_scipy_optimizer", "create_standard_mcmc",
    "create_nuts_mcmc", "handle_pool"
]

Data = Tuple[tf.Tensor, tf.Tensor]
Data_p = Tuple[tf.Tensor,tf.Tensor]
Initial_D = Tuple[tf.Tensor]
Initial_S = Tuple[tf.Tensor]
Initial_lengthscale = Tuple[tf.Tensor]
Initial_variance = Tuple[tf.Tensor]
FullData = Data
Observations = Tuple[tf.Tensor, tf.Tensor, tf.Tensor]


def dfloat(value):  # default float
    return tf.cast(value, default_float())


@contextmanager
def handle_pool(pool: Pool) -> Generator[Pool, None, None]:
    try:
        yield pool
    finally:
        pool.close()
        pool.join()

#################################################################################
# Functions for filtering genes
#################################################################################
def fit_rbf(data, init_lengthscale, init_variance):
    k = gpflow.kernels.RBF()
    m = gpflow.models.GPR(data, kernel=k,mean_function=gpflow.mean_functions.Constant(c=None))
    m.likelihood.variance.assign(0.01)
    m.kernel.lengthscales.assign(init_lengthscale)
    m.kernel.variance.assign(init_variance)
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(m.training_loss,
                            m.trainable_variables,
                            options=dict(maxiter=100))
    return m, np.asarray(m.training_loss())

def fit_rbf2(data, init_lengthscale, init_variance):
    k = gpflow.kernels.RBF()
    m = gpflow.models.GPR(data, kernel=k,mean_function=gpflow.mean_functions.Constant(c=None))
    m.likelihood.variance.assign(0.01)
    m.kernel.lengthscales.assign(init_lengthscale)
    m.kernel.variance.assign(init_variance)
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(m.training_loss,
                            m.trainable_variables,
                            options=dict(maxiter=100))
    return m, np.asarray(m.training_loss())

#def fit_noise(data, init_variance):
#    k = gpflow.kernels.White()
#    m = gpflow.models.GPR(data, kernel=k, mean_function=gpflow.mean_functions.Constant(c=None))
#    m.likelihood.variance.assign(0.01)
#    m.kernel.variance.assign(init_variance)
#    opt = gpflow.optimizers.Scipy()
#    opt_logs = opt.minimize(m.training_loss,
#                            m.trainable_variables,
#                            options=dict(maxiter=100))
#    return m, np.asarray(m.training_loss())

def fit_noise(data, init_variance):
    #k = gpflow.kernels.White()
    k = gpflow.kernels.RBF()
    m = gpflow.models.GPR(data, kernel=k, mean_function=gpflow.mean_functions.Constant(c=None))
    m.likelihood.variance.assign(0.01)
    m.kernel.variance.assign(init_variance)
    m.kernel.lengthscales.assign(1000000.0)
    gpflow.utilities.set_trainable(m.kernel.lengthscales, False)
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(m.training_loss,
                            m.trainable_variables,
                            options=dict(maxiter=100))
    return m, np.asarray(m.training_loss())

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

def load_filtered_data(data_i) -> Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
    i=data_i
    # data = pd.read_csv('density_data_pickedup_cluster10.txt',sep=" ")

    #data = pd.read_csv('density_data_keepdensitysumover10_withcluster.txt',sep=" ")
    data = pd.read_csv('Exon_intron_counts_data_normalizedbylibrarydepthonly_20200120.txt',sep=" ")
    #genes = pd.read_csv('Significant_genes.csv',sep=" ")
    genes = pd.read_csv('genes_unnormalized.csv',sep=" ")
    #data  = data.loc[data['cluster'] == 3]

    #gene_id = data.FBgn.iloc[i]
    gene_id = genes['Name'].iloc[i]
    #print('GENE ID')
    #print(gene_id)
    #data_m = data.loc[data['FBgn'] == gene_id].iloc[0][1:31].to_numpy()
    #data_p = data.loc[data['FBgn'] == gene_id].iloc[0][31:61].to_numpy()

    data_m = data.loc[data['FBgn'] == gene_id].iloc[0][1:31].to_numpy()/data.iloc[i][61] # exones, mrna
    data_p = data.loc[data['FBgn'] == gene_id].iloc[0][31:61].to_numpy()/data.iloc[i][62] # intons, premrna
    data_m = data_m*1000.0
    data_p = data_p*1000.0
    #initial_S = abs(np.sum(data_m[27:29])/3 - np.sum(data_p[59:61])/3)
    initial_S = 10.0
    #initial_S = abs(np.sum(data_m[27:29])/3 - np.sum(data_p[59:61])/3)
    #print(data.loc[data['FBgn'] == gene_id])
    #print(data_p)
    #data_m = data.iloc[i][1:31].to_numpy() # exones
    #data_p = data.iloc[i][31:61].to_numpy() # intons

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
    return full_data, observations, gene_id, full_data_p, observations_p, initial_S

def load_filtered_data_unnormalized(data_i) -> Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
    i=data_i
    # data = pd.read_csv('density_data_pickedup_cluster10.txt',sep=" ")

    #data = pd.read_csv('density_data_keepdensitysumover10_withcluster.txt',sep=" ")
    data = pd.read_csv('Exon_intron_counts_data_normalizedbylibrarydepthonly_20200120.txt',sep=" ")
    #genes = pd.read_csv('Significant_genes.csv',sep=" ")
    genes = pd.read_csv('genes_unnormalized.csv',sep=" ")
    #data  = data.loc[data['cluster'] == 3]

    #gene_id = data.FBgn.iloc[i]
    gene_id = genes['Name'].iloc[i]
    #print('GENE ID')
    #print(gene_id)
    data_m = data.loc[data['FBgn'] == gene_id].iloc[0][1:31].to_numpy()/data.iloc[i][61] # exones, mrna
    data_p = data.loc[data['FBgn'] == gene_id].iloc[0][31:61].to_numpy()/data.iloc[i][62] # intons, premrna
    data_m = data_m*1000.0
    data_p = data_p*1000.0
    #print(data.loc[data['FBgn'] == gene_id])
    #print(data_p)
    #data_m = data.iloc[i][1:31].to_numpy() # exones
    #data_p = data.iloc[i][31:61].to_numpy() # intons

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


def load_sim_data() -> Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
    gene_id = 'sim_data'
    data = pd.read_csv('simulated_data.csv',sep=",")
    data_m = data['Ym'].to_numpy()
    data_p = data['Yp'].to_numpy()
    t = data['t'].to_numpy()

    tp_obs = np.asarray(t)
    ym = np.asarray(data_m, dtype=np.float)
    yp = np.asarray(data_p, dtype=np.float)
    y_full = tf.convert_to_tensor(np.vstack((ym, yp)).reshape(20, 1))
    x_full = tf.convert_to_tensor(np.vstack((tp_obs, tp_obs)).reshape(20, 1))
    full_data = (dfloat(x_full), dfloat(y_full))
    full_data_p = (dfloat(tp_obs), dfloat(yp))
    observations = (dfloat(tp_obs), dfloat(ym), dfloat(yp))
    observations_p = (dfloat(tp_obs), dfloat(yp))
    return full_data, observations, gene_id, full_data_p, observations_p

def load_sim_data_nb() -> Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
    gene_id = 'sim_data'
    #data = pd.read_csv('simulated_data_nb.csv',sep=",")
    #data = pd.read_csv('simulated_Nuha_nb.csv',sep=",")
    data = pd.read_csv('simulated_data_nb_new.csv',sep=",")
    data_m = data['Ym'].to_numpy()
    data_p = data['Yp'].to_numpy()
    t = data['t'].to_numpy()

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


def load_single_gene(gene_id) -> Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
    # data = pd.read_csv('density_data_pickedup_cluster10.txt',sep=" ")
    #data = pd.read_csv('density_data_keepdensitysumover10_withcluster.txt',sep=" ")
    data = pd.read_csv('Exon_intron_counts_data_normalizedbylibrarydepthonly_20200120.txt',sep=" ")
    gene_id = gene_id
    #data = data[data['FBgn'] == gene_id]

    #genes = pd.read_csv('filtered_genes.csv',sep=" ")


    data_m = data[data['FBgn'] == gene_id].iloc[0][1:31].to_numpy()/data[data['FBgn'] == gene_id].iloc[0][61]
    data_p = data[data['FBgn'] == gene_id].iloc[0][31:61].to_numpy()/data[data['FBgn'] == gene_id].iloc[0][62]


    data_m = data_m*1000.0
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
    return full_data, observations, gene_id, full_data_p, observations_p

def load_single_gene_nb(gene_id) -> Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
    # data = pd.read_csv('density_data_pickedup_cluster10.txt',sep=" ")
    #data = pd.read_csv('density_data_keepdensitysumover10_withcluster.txt',sep=" ")
    data = pd.read_csv('Exon_intron_counts_data_normalizedbylibrarydepthonly_20200120.txt',sep=" ")
    gene_id = gene_id
    #data = data[data['FBgn'] == gene_id]

    #genes = pd.read_csv('filtered_genes.csv',sep=" ")


    data_m = data[data['FBgn'] == gene_id].iloc[0][1:31].to_numpy()
    data_p = data[data['FBgn'] == gene_id].iloc[0][31:61].to_numpy()

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

def init_hyperparameters(X):
    initial_lengthscale = np.random.uniform(0. , (np.max(X)-np.min(X))/10.)
    initial_variance = np.random.uniform(1., 100.)
    return initial_lengthscale, initial_variance

#def init_hyperparameters_nb(X):
#    initial_lengthscale = np.random.uniform((1*(np.max(X)-np.min(X)))/100 ,(30*(np.max(X)-np.min(X)))/100)
#    initial_variance = np.random.uniform((1*(np.max(X)-np.min(X)))/100 ,(50*(np.max(X)-np.min(X)))/100)
#    init_alpha = np.random.uniform(0., 10.)
#    init_km = np.random.uniform(0., 100.)
#    return initial_lengthscale, initial_variance

def init_hyperparameters_nb(X):
    initial_lengthscale = np.random.uniform((.5*(np.max(X)-np.min(X)))/100 ,(50*(np.max(X)-np.min(X)))/100)
    initial_variance = np.random.uniform(0.,10.)
    init_alpha = np.random.uniform(0., 10.)
    init_km = np.random.uniform(0., 100.)
    return initial_lengthscale, initial_variance

#def init_hyperparameters_nb(X):
#    initial_lengthscale = np.random.uniform((.5*(np.max(X)-np.min(X)))/100 ,(50*(np.max(X)-np.min(X)))/100)
#    initial_variance = np.random.uniform(0.,10.)
#    init_alpha = p.random.uniform(0., 5.)
#    init_km = np.random.uniform(0., 100.)
#    return initial_lengthscale, initial_variance

def create_trcd_model_with_mean(data: Data, initial_lengthscale: Initial_lengthscale, initial_variance: Initial_variance,initial_S: Initial_S,
                      transform_base: Optional[tfp.bijectors.Bijector] = None) -> Tuple[TRCD, List[gpflow.Parameter]]:
    """
    Creates a TRCD model with initial parameters
    """
    S_prior = tfd.Gamma(concentration=dfloat(10.0), rate=dfloat(5.))
    D_prior = tfd.Gamma(concentration=dfloat(2.0), rate=dfloat(2.))

    variance_m_prior = tfd.Gamma(concentration=dfloat(10.0), rate=dfloat(5.))
    variance_p_prior = tfd.Gamma(concentration=dfloat(1.0), rate=dfloat(5.))

    transform = gpflow.utilities.positive if transform_base is None else transform_base

    S = gpflow.Parameter(initial_S, transform=transform(), prior=S_prior)
    D = gpflow.Parameter(1.1, transform=transform(), prior=D_prior)
    variance_m = gpflow.Parameter(10.0, transform=transform(), prior=variance_m_prior)
    variance_p = gpflow.Parameter(1.0, transform=transform(), prior=variance_p_prior)
    set_kernel_prior = True



    b0_prior = tfd.Gamma(concentration=dfloat(1.0), rate=dfloat(1.))
    b_prior = tfd.Gamma(concentration=dfloat(1.0), rate=dfloat(1.))
    x0_prior = tfd.Gamma(concentration=dfloat(1.0), rate=dfloat(1.))
    mp_prior = tfd.Gamma(concentration=dfloat(1.0), rate=dfloat(1.))
    delay_prior = tfd.Gamma(concentration=dfloat(1.0), rate=dfloat(1.))


    b0 = gpflow.Parameter(1.0, transform=transform(), prior=b0_prior)
    b = gpflow.Parameter(1.0, transform=transform(), prior=b_prior)
    #self.D = positive_parameter(D)
    x0 = gpflow.Parameter(1.0, transform=transform(), prior=x0_prior)
    mp = gpflow.Parameter(1.0, transform=transform(), prior=mp_prior)

    m0 = gpflow.Parameter(1.0, transform=transform(), prior=m0_prior)
    delay = gpflow.Parameter(1.0, transform=transform(), prior=delay_prior)


    #set_kernel_prior = False
    if set_kernel_prior:
        variance = gpflow.Parameter(initial_variance,
                                    transform=transform(),
                                    prior=tfd.Gamma(concentration=dfloat(50.0), rate=dfloat(10.)))
        lengthscale = gpflow.Parameter(initial_lengthscale,
                                       transform=transform(),
                                       prior=tfd.Gamma(concentration=dfloat(10.0), rate=dfloat(2.)))
    else:
        variance = gpflow.Parameter(initial_variance, transform=transform())
        lengthscale = gpflow.Parameter(initial_lengthscale, transform=transform())

    parameters = OrderedDict(D=D,
                             S=S,
                             variance=variance,
                             lengthscale=lengthscale,
                             variance_m = variance_m,
                             variance_p = variance_p,
                             m0 = m0,
                             b0 = b0,
                             x0 = x0,
                             mp = mp,
                             b = b,
                             delay = delay
                             )

    return TRCD(data, **parameters), parameters

def create_trcd_model_nb(data: Data, initial_lengthscale: Initial_lengthscale, initial_variance: Initial_variance,initial_S: Initial_S,
                      transform_base: Optional[tfp.bijectors.Bijector] = None) -> Tuple[TRCD, List[gpflow.Parameter]]:
    """
    Creates a TRCD model with initial parameters
    """
    S_prior = tfd.Gamma(concentration=dfloat(10.0), rate=dfloat(5.))
    D_prior = tfd.Gamma(concentration=dfloat(2.0), rate=dfloat(2.))
    alpha_prior = tfd.Gamma(concentration=dfloat(1.0), rate=dfloat(1.))

    #variance_m_prior = tfd.Gamma(concentration=dfloat(10.0), rate=dfloat(5.))
    #variance_p_prior = tfd.Gamma(concentration=dfloat(1.0), rate=dfloat(5.))

    transform = gpflow.utilities.positive if transform_base is None else transform_base

    S = gpflow.Parameter(initial_S, transform=transform(), prior=S_prior)
    D = gpflow.Parameter(1.1, transform=transform(), prior=D_prior)

    #variance_m = gpflow.Parameter(10.0, transform=transform(), prior=variance_m_prior)
    #variance_p = gpflow.Parameter(1.0, transform=transform(), prior=variance_p_prior)
    #alpha = gpflow.Parameter(10.0, transform=transform(), prior=alpha_prior)
    set_kernel_prior = True
    #set_kernel_prior = False
    if set_kernel_prior:

        variance = gpflow.Parameter(initial_variance,
                                    transform=transform(),
                                    prior=tfd.Gamma(concentration=dfloat(50.0), rate=dfloat(10.)))
        lengthscale = gpflow.Parameter(initial_lengthscale,
                                       transform=transform(),
                                       prior=tfd.Gamma(concentration=dfloat(10.0), rate=dfloat(2.)))
    else:
        variance = gpflow.Parameter(initial_variance, transform=transform())
        lengthscale = gpflow.Parameter(initial_lengthscale, transform=transform())

    #variance_x_white = gpflow.Parameter(std_m, transform=transform(), prior=variance_x_white_prior)
    #variance_f_white = gpflow.Parameter(std_p, transform=transform(), prior=variance_f_white_prior)

    parameters = OrderedDict(D=D,
                             S=S,
                             variance=variance,
                             lengthscale=lengthscale
                             #alpha = alpha
                             )

    return TRCD_nb(data, **parameters), parameters

def create_trcd_model_nb_gaussian(data: Data, initial_lengthscale: Initial_lengthscale, initial_variance: Initial_variance,initial_S: Initial_S,
                      transform_base: Optional[tfp.bijectors.Bijector] = None) -> Tuple[TRCD, List[gpflow.Parameter]]:
    """
    Creates a TRCD model with initial parameters
    """
    S_prior = tfd.Gamma(concentration=dfloat(10.0), rate=dfloat(5.))
    D_prior = tfd.Gamma(concentration=dfloat(2.0), rate=dfloat(2.))
    #D_prior = tfd.LogisticNormal(dfloat(0.01), dfloat(0.5))
    #alpha_prior = tfd.Gamma(concentration=dfloat(2.0), rate=dfloat(2.))

    #variance_m_prior = tfd.Gamma(concentration=dfloat(10.0), rate=dfloat(5.))
    #variance_p_prior = tfd.Gamma(concentration=dfloat(1.0), rate=dfloat(5.))

    transform = gpflow.utilities.positive if transform_base is None else transform_base

    S = gpflow.Parameter(initial_S, transform=transform(), prior=S_prior)
    D = gpflow.Parameter(1.1, transform=transform(), prior=D_prior)

    #variance_m = gpflow.Parameter(10.0, transform=transform(), prior=variance_m_prior)
    #variance_p = gpflow.Parameter(1.0, transform=transform(), prior=variance_p_prior)
    #alpha = gpflow.Parameter(1.0, transform=transform(), prior=alpha_prior)
    set_kernel_prior = True
    #set_kernel_prior = False
    if set_kernel_prior:

        variance = gpflow.Parameter(initial_variance,
                                    transform=transform(),
                                    prior=tfd.Gamma(concentration=dfloat(50.0), rate=dfloat(10.)))
        lengthscale = gpflow.Parameter(initial_lengthscale,
                                       transform=transform(),
                                       prior=tfd.Gamma(concentration=dfloat(10.0), rate=dfloat(2.)))
    else:
        variance = gpflow.Parameter(initial_variance, transform=transform())
        lengthscale = gpflow.Parameter(initial_lengthscale, transform=transform())

    #variance_x_white = gpflow.Parameter(std_m, transform=transform(), prior=variance_x_white_prior)
    #variance_f_white = gpflow.Parameter(std_p, transform=transform(), prior=variance_f_white_prior)

    parameters = OrderedDict(D = D,
                             S = S,
                             variance = variance,
                             lengthscale = lengthscale
                             #variance_m = variance_m,
                             #variance_p = variance_p
                             )

    return TRCD_nb(data, **parameters), parameters


def create_trcd_model(data: Data, initial_lengthscale: Initial_lengthscale, initial_variance: Initial_variance,initial_S: Initial_S,
                      transform_base: Optional[tfp.bijectors.Bijector] = None) -> Tuple[TRCD, List[gpflow.Parameter]]:
    """
    Creates a TRCD model with initial parameters
    """
    S_prior = tfd.Gamma(concentration=dfloat(2.0), rate=dfloat(2.))
    D_prior = tfd.Gamma(concentration=dfloat(2.0), rate=dfloat(2.0))

    variance_m_prior = tfd.Gamma(concentration=dfloat(10.0), rate=dfloat(5.))
    variance_p_prior = tfd.Gamma(concentration=dfloat(1.0), rate=dfloat(5.))

    transform = gpflow.utilities.positive if transform_base is None else transform_base

    S = gpflow.Parameter(initial_S, transform=transform(), prior=S_prior)
    D = gpflow.Parameter(1.1, transform=transform(), prior=D_prior)
    variance_m = gpflow.Parameter(10.0, transform=transform(), prior=variance_m_prior)
    variance_p = gpflow.Parameter(1.0, transform=transform(), prior=variance_p_prior)
    set_kernel_prior = True
    #set_kernel_prior = False
    if set_kernel_prior:

        variance = gpflow.Parameter(initial_variance,
                                    transform=transform(),
                                    prior=tfd.Gamma(concentration=dfloat(50.0), rate=dfloat(10.)))
        #lengthscale = gpflow.Parameter(initial_lengthscale,
        #                               transform=tfp.bijectors.Invert(transform()),
        #                               prior=tfd.Gamma(concentration=dfloat(1.0), rate=dfloat(1.)))
        lengthscale = gpflow.Parameter(initial_lengthscale,
                                       transform=transform(),
                                       prior=tfd.Gamma(concentration=dfloat(12.0), rate=dfloat(2.)))
    else:
        variance = gpflow.Parameter(initial_variance, transform=transform())
        lengthscale = gpflow.Parameter(initial_lengthscale, transform=transform())

    #variance_x_white = gpflow.Parameter(std_m, transform=transform(), prior=variance_x_white_prior)
    #variance_f_white = gpflow.Parameter(std_p, transform=transform(), prior=variance_f_white_prior)

    parameters = OrderedDict(D=D,
                             S=S,
                             variance=variance,
                             lengthscale=lengthscale,
                             variance_m = variance_m,
                             variance_p = variance_p
                             )

    return TRCD(data, **parameters), parameters

def create_trcd_model_sim_data(data: Data,  initial_lengthscale: Initial_lengthscale, initial_variance: Initial_variance,initial_S: Initial_S,
                      transform_base: Optional[tfp.bijectors.Bijector] = None) -> Tuple[TRCD, List[gpflow.Parameter]]:
    """
    Creates a TRCD model with initial parameters
    """

    #std_m = np.mean(tf.math.reduce_std((tf.reshape(data[1][0:30],[3,10])),axis=1))
    #std_p = np.mean(tf.math.reduce_std((tf.reshape(data[1][30:60],[3,10])),axis=1))
    #print('Standard deviations')
    #print(std_m)
    #print(std_p)
    #std_m = 1.0
    #std_p = 1.0
    #std_m = 10.0
    #std_p = 10.0
    #initial_lengthscale = 10.00


    #S_prior = tfd.InverseGamma(concentration=dfloat(1.0), scale=dfloat(1.0))
    #S_prior = tfd.Logistic(loc=dfloat(0.05*std_m), scale=dfloat(std_m))
    S_prior = tfd.Gamma(concentration=dfloat(1.0), rate=dfloat(1.))
    #S_prior = tfd.Normal(loc=dfloat(15.0), scale=dfloat(5.0))
    #S_prior = tfd.QuantizedDistribution(distribution=tfp.distributions.LogitNormal(loc=dfloat(0.0), scale=dfloat(2.0)), low = dfloat(5.0), high = dfloat(25.0))
    #S_prior = tfd.HalfNormal(scale=dfloat(25.0))
    #S_prior = None
    #S_prior = tfd.Uniform(low=dfloat(0.0), high=dfloat(25.))
    #S_prior = None
    D_prior = tfd.Gamma(concentration=dfloat(1.0), rate=dfloat(2.))
    #D_prior = tfd.QuantizedDistribution(distribution=tfp.distributions.LogitNormal(loc=dfloat(0.0), scale=dfloat(2.0)), low = dfloat(0.1), high = dfloat(2.0))
    #D_prior = None
    #D_prior = tfd.Uniform(low=dfloat(0.0), high=dfloat(3.))
    variance_m_prior = tfd.Gamma(concentration=dfloat(1.0), rate=dfloat(1.))
    variance_p_prior = tfd.Gamma(concentration=dfloat(1.0), rate=dfloat(1.))

    #variance_m_prior = tfd.QuantizedDistribution(distribution=tfp.distributions.LogitNormal(loc=dfloat(0.0), scale=dfloat(2.0)), low = dfloat(1.0), high = dfloat(20.0))
    #variance_p_prior = tfd.QuantizedDistribution(distribution=tfp.distributions.LogitNormal(loc=dfloat(0.0), scale=dfloat(2.0)), low = dfloat(0.1), high = dfloat(20.0))


    #variance_m_prior = tfd.Normal(loc=dfloat(std_m), scale=dfloat(1.0))
    #variance_p_prior = tfd.Normal(loc=dfloat(std_p), scale=dfloat(1.0))
    #variance_m_prior = tfd.LogitNormal(loc=dfloat(0.05*std_m), scale=dfloat(std_m))
    #variance_p_prior = tfd.LogitNormal(loc=dfloat(0.05*std_p), scale=dfloat(std_p))

    #variance_m_prior = tfd.Logistic(loc=dfloat(0.05*std_m), scale=dfloat(std_m))
    #variance_p_prior = tfd.Logistic(loc=dfloat(0.05*std_m), scale=dfloat(std_m))
    #variance_m_prior = tfd.HalfNormal(scale=dfloat(1.0))
    #variance_p_prior = tfd.HalfNormal(scale=dfloat(1.0))
    #variance_m_prior = None
    #variance_m_prior = None
    #variance_x_white_prior = tfd.Logistic(loc=dfloat(0.05*std_m), scale=dfloat(std_m))
    #variance_f_white_prior = tfd.Logistic(loc=dfloat(0.05*std_p), scale=dfloat(std_p))


    #D_prior = None
    #variance_x_white_prior = None
    #variance_f_white_prior = None

    transform = gpflow.utilities.positive if transform_base is None else transform_base

    S = gpflow.Parameter(initial_S, transform=transform(), prior=S_prior)
    D = gpflow.Parameter(1.1, transform=transform(), prior=D_prior)
    variance_m = gpflow.Parameter(1.0, transform=transform(), prior=variance_m_prior)
    variance_p = gpflow.Parameter(1.0, transform=transform(), prior=variance_p_prior)
    set_kernel_prior = True
    #set_kernel_prior = False
    if set_kernel_prior:
        #variance = gpflow.Parameter(initial_variance,
        #                            transform=transform(),
        #                            prior=tfd.InverseGamma(concentration=dfloat(1.0), scale=dfloat(1.)))
        #variance = gpflow.Parameter(initial_variance,
        #                            transform=transform(),
        #                            prior=tfd.Normal(loc=dfloat(prior_variance_mean), scale=dfloat(5.0)))
        #variance = gpflow.Parameter(initial_variance,
        #                            transform=transform(),
        #                            prior=tfd.LogNormal(loc=dfloat(0.05*std_p**2), scale=dfloat(std_p**2)))
        variance = gpflow.Parameter(initial_variance,
                                    transform=transform(),
                                    prior=tfd.Gamma(concentration=dfloat(1.0), rate=dfloat(1.0)))
        #variance = gpflow.Parameter(initial_variance,
        #                            transform=transform(),
        #                            prior = tfd.QuantizedDistribution(distribution=tfp.distributions.LogitNormal(loc=dfloat(0.0), scale=dfloat(2.0)), low = dfloat(1.0), high = dfloat(70.0)))
        #variance = gpflow.Parameter(initial_variance,
        #                            transform=transform(),
        #                            prior=tfd.LogitNormal(loc=dfloat(0.0), scale=dfloat(1.0)))
        #lengthscale = gpflow.Parameter(initial_lengthscale,
        #                               transform=transform(),
        #                               prior=tfd.Uniform(low=dfloat(10.0), high=dfloat(70.0)))
        #lengthscale = gpflow.Parameter(initial_lengthscale,
        #                               transform=transform(),
        #                               prior = tfd.QuantizedDistribution(distribution=tfp.distributions.LogitNormal(loc=dfloat(0.0), scale=dfloat(2.0)), low = dfloat(10.0), high = dfloat(50.0)))
        #lengthscale = gpflow.Parameter(initial_lengthscale,
        #                               transform=transform(),
        #                               prior=tfd.LogitNormal(loc=dfloat(0.0), scale=dfloat(0.1)))

        #lengthscale = gpflow.Parameter(initial_lengthscale,
        #                               transform=transform(),
        #                               prior=tfd.Logistic(loc=dfloat(10.0), scale=dfloat(1.0)))
        #lengthscale = gpflow.Parameter(initial_lengthscale,
        #                               transform=transform(),
        #                               prior=tfd.InverseGaussian(loc=dfloat(15.0), concentration=dfloat(1.0)))
        lengthscale = gpflow.Parameter(initial_lengthscale,
                                       transform=transform(),
                                       prior=tfd.Gamma(concentration=dfloat(1.0), rate=dfloat(1.)))
        #lengthscale = gpflow.Parameter(initial_lengthscale,
        #                               transform=transform(),
        #                               prior=tfd.InverseGamma(concentration=dfloat(1.0), scale=dfloat(10.0)))
        #lengthscale = gpflow.Parameter(initial_lengthscale,
        #                               transform=transform(),
        #                               prior=tfd.Normal(loc=dfloat(10.0), scale=dfloat(5.0)))
        #lengthscale = gpflow.Parameter(initial_variance,
        #                            transform=transform(),
        #                            prior=tfd.LogNormal(loc=dfloat(2.0), scale=dfloat(1.0)))
    else:
        variance = gpflow.Parameter(initial_variance, transform=transform())
        lengthscale = gpflow.Parameter(initial_lengthscale, transform=transform())

    #variance_x_white = gpflow.Parameter(std_m, transform=transform(), prior=variance_x_white_prior)
    #variance_f_white = gpflow.Parameter(std_p, transform=transform(), prior=variance_f_white_prior)

    parameters = OrderedDict(D=D,
                             S=S,
                             variance=variance,
                             lengthscale=lengthscale,
                             variance_m = variance_m,
                             variance_p = variance_p
                             )

    return TRCD(data, **parameters), parameters


def create_premRNA_model(data: Data, lengthscale: float, variance: float,
                      transform_base: Optional[tfp.bijectors.Bijector] = None) -> Tuple[TRCD, List[gpflow.Parameter]]:
    """
    Creates a standard GPR for estimation of the GP for premRNA
    """
    #lengthscale_prior = tfd.Uniform(low=dfloat(0.0), high=dfloat(20.0))
    std_p = np.mean(tf.math.reduce_std((tf.reshape(data[1][0:30],[3,10])),axis=1))
    transform = gpflow.positive if transform_base is None else transform_base

    lengthscale_prior = prior=tfd.Uniform(low=dfloat(10.0), high=dfloat(70.0))
    variance_prior = tfd.InverseGamma(concentration=dfloat(1.0), scale=dfloat(1.0))
    #lengthscale_prior = tfd.InverseGamma(concentration=dfloat(1.0), scale=dfloat(1.0))
    #variance_prior = tfd.Logistic(loc=dfloat(0.05*std_p), scale=dfloat(std_p))

    # linear_variance_prior = tfd.Normal(loc=dfloat(0.), scale=dfloat(10.))
    # bias_variance_prior = tfd.Normal(loc=dfloat(0.), scale=dfloat(10.))

    rbf_kernel = gpflow.kernels.RBF()

    rbf_kernel.lengthscale = gpflow.Parameter(
        5.0,
        prior=lengthscale_prior,
        transform=transform())

    rbf_kernel.variance = gpflow.Parameter(
        1.0,
        prior=variance_prior,
        transform=transform())

    k = rbf_kernel
    model = gpflow.models.GPR(data, kernel=k)

    un_lengthscale = rbf_kernel.lengthscale
    un_variance = rbf_kernel.variance

    state_variables = [un_lengthscale, un_variance]

    return model, state_variables


def optimize_with_scipy_optimizer_nb(trcd_nb: TRCD_nb, parameters: List[gpflow.Parameter], maxiter: int = 100):
    opt = gpflow.optimizers.Scipy()
    loss_cb = tf.function(trcd_nb.model.training_loss, autograph=False)

    def step_cb(step: int, variables: List[tf.Tensor], values: List[tf.Tensor]):
        if step % 10 == 0:
            [variable.assign(value) for variable, value in zip(variables, values)]
            print(f"Step {step} loss={loss_cb()}")

    variables = [v.unconstrained_variable for v in parameters]
    return opt.minimize(loss_cb, variables, step_callback=step_cb, options=dict(maxiter=maxiter))

def optimize_with_scipy_optimizer(trcd: TRCD, parameters: List[gpflow.Parameter], maxiter: int = 100):
    opt = gpflow.optimizers.Scipy()
    loss_cb = tf.function(trcd.model.training_loss, autograph=False)

    def step_cb(step: int, variables: List[tf.Tensor], values: List[tf.Tensor]):
        if step % 10 == 0:
            [variable.assign(value) for variable, value in zip(variables, values)]
            print(f"Step {step} loss={loss_cb()}")

    variables = [v.unconstrained_variable for v in parameters]
    return opt.minimize(loss_cb, variables, step_callback=step_cb, options=dict(maxiter=maxiter))


def optimize_premRNA(model: m_premRNA, parameters: List[gpflow.Parameter], maxiter: int = 100):
    opt = gpflow.optimizers.Scipy()
    loss_cb = lambda: model.neg_log_marginal_likelihood()
    #loss_cb = tf.function(m_premRNA.neg_log_marginal_likelihood, autograph=False)
    def step_cb(step: int, variables: List[tf.Tensor], values: List[tf.Tensor]):
        if step % 10 == 0:
            [variable.assign(value) for variable, value in zip(variables, values)]
            print(f"Step {step} loss={loss_cb()}")
    variables = [v.unconstrained_variable for v in parameters]
    return opt.minimize(loss_cb, variables, step_callback=step_cb, options=dict(maxiter=100))

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

def predict_trcd_sim_data(trcd: tf.Module, gene_id, observations: Observations, num_predict: int = 100):

    tp_obs, ym, yp = [np.array(o) for o in observations]
    t0 = tp_obs.copy()
    #xx = np.linspace(np.min(tp_obs), np.max(tp_obs), num_predict).reshape(num_predict, 1)
    #xx_full = np.concatenate((tp_obs, tp_obs)).reshape(-1, 1)
    xx_full = np.concatenate((t0, t0)).reshape(-1, 1)
    num_predict = t0.shape[0]
    mean, var = trcd.model.predict_f(xx_full)
    mean1 = mean[0:num_predict]
    mean2 = mean[num_predict:2 * num_predict]
    var1 = var[0:num_predict]
    var2 = var[num_predict:2 * num_predict]

    #dist_mean = (ym - (mean1[:, 0]+2 * np.sqrt(var1[:, 0])))**2 + ((yp - (mean2[:, 0]+2 * np.sqrt(var2[:, 0]))))**2
    #dist_var = ym - (mean1[:, 0]+2 * np.sqrt(var1[:, 0])) + (yp - (mean2[:, 0]+2 * np.sqrt(var2[:, 0])))
    #ym = ym.reshape(3,10)
    #yp = yp.reshape(3,10)
    mean_ym = np.mean(ym)
    mean_yp = np.mean(yp)
    dist_mean = 0.05 * np.sum(((mean_ym - mean1[:, 0])**2 + (mean_yp - mean2[:, 0])**2))
    dist_var = 0.05 * np.sum(((mean_ym - (mean1[:, 0] + 2 * np.sqrt(var1[:, 0])))**2 + (mean_yp - (mean2[:, 0]+2 * np.sqrt(var2[:, 0])))**2))
    return dist_mean , dist_var


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

    #dist_mean = (ym - (mean1[:, 0]+2 * np.sqrt(var1[:, 0])))**2 + ((yp - (mean2[:, 0]+2 * np.sqrt(var2[:, 0]))))**2
    #dist_var = ym - (mean1[:, 0]+2 * np.sqrt(var1[:, 0])) + (yp - (mean2[:, 0]+2 * np.sqrt(var2[:, 0])))
    ym = ym.reshape(3,10)
    yp = yp.reshape(3,10)
    mean_ym = np.mean(ym, axis=0)
    mean_yp = np.mean(yp, axis=0)
    dist_mean = 0.05 * np.sum(((mean_ym - mean1[:, 0])**2 + (mean_yp - mean2[:, 0])**2))
    dist_var = 0.05 * np.sum(((mean_ym - (mean1[:, 0] + 2 * np.sqrt(var1[:, 0])))**2 + (mean_yp - (mean2[:, 0]+2 * np.sqrt(var2[:, 0])))**2))
    return dist_mean , dist_var

def plot_trcd_predict(trcd: tf.Module, gene_id, observations: Observations, num_predict: int = 100):

    tp_obs, ym, yp = [np.array(o) for o in observations]

    #xx = np.linspace(np.min(tp_obs), np.max(tp_obs), num_predict).reshape(num_predict, 1)
    xx = np.linspace(np.min(tp_obs), np.max(tp_obs), num_predict).reshape(num_predict, 1)
    xx_full = np.concatenate((xx, xx)).reshape(-1, 1)
    mean, var = trcd.model.predict_f(xx_full)
    mean1 = mean[0:num_predict]
    mean2 = mean[num_predict:2 * num_predict]
    var1 = var[0:num_predict]
    var2 = var[num_predict:2 * num_predict]

    # Plot predicted model
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_ylabel('mRNA', color='r')
    ax1.plot(tp_obs.flatten(), ym.flatten(), 'rx', mew=2)
    ax1.plot(xx[:, 0], mean1[:, 0], 'r', lw=2)
    ax1.fill_between(xx[:, 0],
                         mean1[:, 0] - 2 * np.sqrt(var1[:, 0]),
                         mean1[:, 0] + 2 * np.sqrt(var1[:, 0]),
                         color='red',
                         alpha=0.2)
    ax2 = ax1.twinx()
    ax2.set_ylabel('premRNA', color='b')
    ax2.plot(tp_obs.flatten(), yp.flatten(), 'bx', mew=2)
    ax2.plot(xx[:, 0], mean2[:, 0], 'b', lw=2)
    plt.fill_between(xx[:, 0],
                     mean2[:, 0] - 2 * np.sqrt(var2[:, 0]),
                     mean2[:, 0] + 2 * np.sqrt(var2[:, 0]),
                     color='blue',
                     alpha=0.2)
    plt.title(str(gene_id), fontsize=18)
    plt.savefig(str(gene_id)+'.png')
def plot_trcd_predict_nb(trcd: tf.Module, gene_id, observations: Observations, num_predict: int = 100):

    tp_obs, ym, yp = [np.array(o) for o in observations]

    #xx = np.linspace(np.min(tp_obs), np.max(tp_obs), num_predict).reshape(num_predict, 1)
    xx = np.linspace(np.min(tp_obs), np.max(tp_obs), num_predict).reshape(num_predict, 1)
    xx_full = np.concatenate((xx, xx)).reshape(-1, 1)
    mean, var = trcd.model.predict_f(xx_full)
    mean1 = mean[0:num_predict]
    mean2 = mean[num_predict:2 * num_predict]
    var1 = var[0:num_predict]
    var2 = var[num_predict:2 * num_predict]

    # Plot predicted model
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_ylabel('mRNA', color='r')
    ax1.plot(tp_obs.flatten(), ym.flatten(), 'rx', mew=2)
    ax1.plot(xx[:, 0], mean1[:, 0], 'r', lw=2)
    ax1.fill_between(xx[:, 0],
                         mean1[:, 0] - 2 * np.sqrt(var1[:, 0]),
                         mean1[:, 0] + 2 * np.sqrt(var1[:, 0]),
                         color='red',
                         alpha=0.2)
    ax2 = ax1.twinx()
    ax2.set_ylabel('premRNA', color='b')
    ax2.plot(tp_obs.flatten(), yp.flatten(), 'bx', mew=2)
    ax2.plot(xx[:, 0], mean2[:, 0], 'b', lw=2)
    plt.fill_between(xx[:, 0],
                     mean2[:, 0] - 2 * np.sqrt(var2[:, 0]),
                     mean2[:, 0] + 2 * np.sqrt(var2[:, 0]),
                     color='blue',
                     alpha=0.2)
    plt.title(str(gene_id), fontsize=18)
    plt.savefig(str(gene_id)+'_nb'+'.png')

def plot_trcd_predict_posterior_samples(trcd: tf.Module, gene_id, observations: Observations, num_predict: int = 100):

    tp_obs, ym, yp = [np.array(o) for o in observations]
    #num_predict = 1000
    xx = np.linspace(np.min(tp_obs), np.max(tp_obs), num_predict).reshape(num_predict, 1)
    xx_full = np.concatenate((xx, xx)).reshape(-1, 1)
    mean, var = trcd.model.predict_f(xx_full)
    num_samples = 1
    samples = trcd.model.predict_f_samples(xx_full, num_samples)
    mean1 = mean[0:num_predict]
    mean2 = mean[num_predict:2 * num_predict]
    var1 = var[0:num_predict]
    var2 = var[num_predict:2 * num_predict]

    # Plot predicted model
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_ylabel('mRNA', color='r')
    ax1.plot(tp_obs.flatten(), ym.flatten(), 'rx', mew=2)
    ax1.plot(xx[:, 0], mean1[:, 0], 'r', lw=2)
    ax1.fill_between(xx[:, 0],
                         mean1[:, 0] - 2 * np.sqrt(var1[:, 0]),
                         mean1[:, 0] + 2 * np.sqrt(var1[:, 0]),
                         color='red',
                         alpha=0.2)
    print(samples[:, 0:num_predict, 0].numpy().T.shape)
    ax1.plot(xx, samples[:,0:num_predict, 0].numpy().T, 'r', linewidth=.5)

    ax2 = ax1.twinx()
    ax2.set_ylabel('premRNA', color='b')
    ax2.plot(tp_obs.flatten(), yp.flatten(), 'bx', mew=2)
    ax2.plot(xx[:, 0], mean2[:, 0], 'b', lw=2)
    ax2.plot(xx, samples[:,num_predict:(2*num_predict), 0].numpy().T, 'b', linewidth=.5)
    plt.fill_between(xx[:, 0],
                     mean2[:, 0] - 2 * np.sqrt(var2[:, 0]),
                     mean2[:, 0] + 2 * np.sqrt(var2[:, 0]),
                     color='blue',
                     alpha=0.2)
    plt.title(str(gene_id), fontsize=18)
    plt.savefig(str(gene_id)+'_tf2_'+'.png')

def plot_premRNA_predict(m_premRNA: m_premRNA, gene_id, observations: Observations, num_predict: int = 100):

    tp_obs, yp = [np.array(o) for o in observations]

    xx = np.linspace(np.min(tp_obs), np.max(tp_obs), num_predict).reshape(num_predict, 1)
    mean, var = m_premRNA.predict_y(xx)
    mean1 = mean[0:num_predict]
    var1 = var[0:num_predict]

    # Plot predicted model
    plt.figure(figsize=(12, 6))
    plt.plot(tp_obs.flatten(), yp.flatten(), 'bx', mew=2)
    plt.plot(xx[:, 0], mean1[:, 0], 'b', lw=2)

    plt.fill_between(xx[:, 0],
                     mean1[:, 0] - 2 * np.sqrt(var1[:, 0]),
                     mean1[:, 0] + 2 * np.sqrt(var1[:, 0]),
                     color='blue',
                     alpha=0.2)
    plt.savefig(str(gene_id)+'prem_RNA'+'.png')
    #plt.show()
