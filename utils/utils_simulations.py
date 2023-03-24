from typing import Callable, List, TypeVar
from typing import Generator, List, Optional, Tuple
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from gpflow.config import default_float

from trcd.kernels import Kernel_mRNA

import sys
sys.path.append('..')
from utils.utilities import fit_rbf

def dfloat(value):  # default float
    return tf.cast(value, default_float())

def dint(value):  # default float
    return tf.cast(value, gpflow.default_int())

Scalar = TypeVar("Scalar", tf.Tensor, float)
Observations = Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
Data = Tuple[tf.Tensor, tf.Tensor]
Data_p = Tuple[tf.Tensor,tf.Tensor]
FullData = Data
Observations = Tuple[tf.Tensor, tf.Tensor, tf.Tensor]

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


def load_single_gene(gene_id, tr_id) -> Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
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
    '''
    Plot predictions for trcd model;
    Resulted plots include uncertainty to f and y.
    '''
    tp_obs, ym, yp = [np.array(o) for o in observations]

    xx = np.linspace(np.min(tp_obs), np.max(tp_obs), num_predict).reshape(num_predict, 1)
    xx_full = np.concatenate((xx, xx)).reshape(-1, 1)
    ## TO DO: See if there is a more neat way for predicting through using trcd.model.predict_y() function
    #
    #gpflow.config.set_default_jitter(0.5)
    #try:
    #    #print('predicted y')
    #mean, var = trcd.model.predict_y(xx_full)
    #except:
    #    print('predicted y failed, predicted f')
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

def generate_data(gene_id, tr_id, S, D, var_m, var_p):
    # S -- splicing rate;
    # D -- degradationl;
    # variance -- related to amplitude of underlying pre-mRNA;
    # lengthscale -- smoothness of underlying pre-mRNA.

    '''
    Generate the data
    '''

    data, observations, gene_id, data_p, observations_p = load_single_gene(gene_id, tr_id)

    f_sample = np.asarray(data_p[1])

    t0 = np.array((95.0,105.0,115.0,125.0,145.0,160.0,175.0,190.0,205.0,220.0))
    rep_no = 3
    t = np.hstack((t0,t0,t0))[:,None]
    tp_full = np.asarray(t)

    '''
    Estimate a single Gaussian process on pre-mRNA data
    '''
    # pre-mRNA data; data_p[0] -- input time points; data_p[1] -- observations.
    data_full = (dfloat(np.asarray(data_p[0]).reshape(-1,1)), dfloat(np.asarray(data_p[1]).reshape(-1,1)))

    # initialize hyperparameters
    init_lengthscale = 10.0
    init_variance = 1.0
    m, lik = fit_rbf(data_full, init_lengthscale, init_variance)

    # Print summary of the model
    #print_summary(m)

    ## predict mean and variance of latent GP at test points
    xx = np.linspace(95.0, 220.0, 100).reshape(100, 1)  # test points must be of shape (N, D)
    mean, var = m.predict_f(xx)
    f_sample, var = m.predict_f(t0.reshape(-1, 1))
    t0 = t0.reshape(-1, 1)

    # Define parameters for data generation

    variance = 69.0    # these parameters are close to single GP on pre-mRNA
    lengthscale = 33.0 #

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

    return (t0, t , y, ym, yp, m)
