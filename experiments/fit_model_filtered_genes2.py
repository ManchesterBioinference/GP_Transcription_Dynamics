from typing import Tuple, TypeVar

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

import gpflow
from gpflow.config import default_float
from gpflow.utilities import print_summary
import sys
sys.path.append('..')
from utils.utilities import (create_data, load_data, load_single_gene, create_standard_mcmc, create_trcd_model,
                       optimize_with_scipy_optimizer,  fit_rbf, predict_trcd,
                       plot_trcd_predict, select_parameters, init_hyperparameters, compute_hessian)

gpflow.config.set_default_float(np.float64)  # noqa

Scalar = TypeVar("Scalar", tf.Tensor, float)
FullData = Tuple[tf.Tensor, tf.Tensor]
Observations = Tuple[tf.Tensor, tf.Tensor, tf.Tensor]

def dfloat(value):  # default float
    return tf.cast(value, default_float())

def main():
    np.random.seed(100)
    tf.random.set_seed(100)

    data1 = pd.read_csv('../data/LB_GP_TS.csv', sep=",")
    data2 = pd.read_csv('../data/Exon_intron_counts_data_normalizedbylibrarydepthonly_20200120.txt',sep=" ")


    t0 = np.array((95.0,105.0,115.0,125.0,145.0,160.0,175.0,190.0,205.0,220.0))
    rep_no = 3
    t = np.hstack((t0,t0,t0))[:,None]

    t0 = np.array((95.0,105.0,115.0,125.0,145.0,160.0,175.0,190.0,205.0,220.0))
    #names_transcripts =  pd.read_csv('gene_transcript_cl_10.csv', sep=",")
    #names_transcripts =  pd.read_csv('../data/zygotic_tr_95_genes.csv', sep=";")
    names_transcripts = pd.read_csv('LB_zyg95_clusters_degradation.csv', sep=",")
    #names_transcripts = names_transcripts.dropna()
    n_genes = len(names_transcripts.index)
    #n_genes = 5

    recorded_genes = []
    recorded_transcripts = []
    all_genes = []
    all_transcripts = []
    #parameters_estimates_all_genes = np.zeros((n_genes,6*3))

    parameters_estimates_all_genes = np.zeros((n_genes,6*3))

    n_genes = 10

    #for i in range(n_genes):
    for i in range(2):
        #gene_id = names_transcripts['FBgn'].iloc[i]
        #tr_id = names_transcripts['FBtr'].iloc[i]

        #gene_id = 'FBgn0266129'
        #tr_id = 'FBtr0072458'
        gene_id = names_transcripts['gene_id'].iloc[i]
        tr_id = names_transcripts['tr_id'].iloc[i]
        all_genes.append(gene_id)
        all_transcripts.append(tr_id)

        try:
            data, observations, gene_id, data_p, observations_p = load_single_gene(gene_id, tr_id)

            print('Data found')

            '''
            Fit GP regression on pre-mRNA
            '''

            # Estimate a single GP based on pre-mRNA data
            data_full = (dfloat(np.asarray(data_p[0]).reshape(-1,1)), dfloat(np.asarray(data_p[1]).reshape(-1,1)))
            #print(data_full)

            init_lengthscale = 10.0
            init_variance = ((np.max(np.asarray(data_p[1]))-np.min(np.asarray(data_p[1])))/2)**2
            m, lik = fit_rbf(data_full, init_lengthscale, init_variance)

            print_summary(m)

            '''
            Fit trcd model
            '''
            # Heruistic initialization of hyperparameters: init_hyperparameters(t0)
            # Check if the distance from the predicted GP mean is not too far away from mean of observations

            num_fail = 0

            while(num_fail<20):
                try:

                    if(num_fail==0):
                        initial_lengthscale = m.kernel.lengthscales.value()
                        initial_variance = m.kernel.variance.value()

                        initial_S = 0.5
                        initial_D = 0.1
                    else:
                        #initial_lengthscale = np.random.uniform(10.,30.0,1)[0]
                        #initial_variance = np.random.uniform(m.kernel.variance.value()*0.01,m.kernel.variance.value(),1)[0]
                        initial_S = np.random.uniform(0.001,1.0,1)[0]
                        initial_D = np.random.uniform(0.001,0.5,1)[0]

                    trcd, dict_parameters = create_trcd_model(data, initial_lengthscale, initial_variance, initial_S, initial_D, transform_base=None)
                    dict_parameters = select_parameters(dict_parameters,
                                                        names=None)  # When `names` is None, same dictionary is returned.
                    # NOTE: WARNING! The order of parameters is quite important here,
                    # as we pass them around and use same order for plotting and setting titles for plots.
                    # For that reason we use ordered dictionary.
                    parameters = list(dict_parameters.values())
                    # NOTE: Updates TRCD model parameters in place!
                    res = optimize_with_scipy_optimizer(trcd, parameters)

                    print_summary(trcd)

                    # Compute confidence intervals with Fisher infomation matrix
                    try:
                        fisher = -np.array(compute_hessian(trcd, parameters))
                        inv_fisher = np.linalg.inv(np.diag(fisher))

                        ci = 1.96 * np.sqrt(inv_fisher)
                        ci = ci.diagonal()
                        parameters_vector = tf.stack(parameters)
                        #ci changed since parameters should be positive
                        ci_lower = parameters_vector * np.exp(-ci)
                        ci_upper = parameters_vector * np.exp( ci)
                    except:
                        parameters_vector = tf.stack(parameters)
                        #ci changed since parameters should be positive
                        ci_lower = np.nan
                        ci_upper = np.nan

                    parameters_estimates_all_genes[i,:] = np.concatenate(( parameters_vector, ci_lower, ci_upper), axis=0)

                    #Results = pd.DataFrame({
                    #   'name': pd.Categorical(dict_parameters.keys()),
                    #   'value': parameters_vector,
                    #   'ci_lower': ci_lower,
                    #   'ci_upper': ci_upper
                    # })
                    #Results.to_csv('output/'+str(gene_id)+'_'+str(tr_id) +'.csv',index=False)
                    var_m_noise = trcd.model.likelihood.variance_m.value()
                    var_p_noise = trcd.model.likelihood.variance_p.value()
                    plot_trcd_predict(1,trcd, tr_id, gene_id, observations, var_m_noise, var_p_noise)
                    print('Finished')
                    recorded_genes.append(gene_id)
                    recorded_transcripts.append(tr_id)
                    num_fail = 30
                except:
                    num_fail = num_fail + 1

        except:
            print('data were not found')

    parameters_estimates_all_genes = pd.DataFrame(parameters_estimates_all_genes)
    parameters_estimates_all_genes.columns = ['D', 'S', 'variance', 'lengthscale', 'variance_m', 'variancve_p',
                                              'D_95_l', 'S_95_l', 'variance_95_l', 'lengthscale_95_l', 'variance_m_95_l', 'variancve_p_95_l',
                                              'D_95_u', 'S_95_u', 'variance_95_u', 'lengthscale_95_u', 'variance_m_95_u', 'variancve_p_95_u']

    print(parameters_estimates_all_genes)
    parameters_estimates_all_genes['gene_id'] = all_genes
    parameters_estimates_all_genes['tr_id'] = all_transcripts

    parameters_estimates_all_genes.to_csv("../output/parameters_estimates_all_genes_remaining_final.csv")


if __name__ == "__main__":
    main()
