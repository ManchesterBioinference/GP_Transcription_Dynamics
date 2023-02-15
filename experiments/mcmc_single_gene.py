from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Callable, List, TypeVar

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import pymc3
from gpflow.config import default_float

import gpflow
from gpflow.utilities import print_summary
import sys
sys.path.append('..')
from trcd.utils import SamplingHelper
from utils.utilities import (HMCParameters, create_nuts_mcmc, create_standard_mcmc, create_trcd_model, handle_pool,
                       optimize_with_scipy_optimizer, select_parameters, load_single_gene,init_hyperparameters, predict_trcd, create_mala_mcmc,
                       plot_trcd_predict, compute_prior_hyperparameters_variance, fit_rbf)

gpflow.config.set_default_float(np.float64)  # noqa

Scalar = TypeVar("Scalar", tf.Tensor, float)

def dfloat(value):  # default float
    return tf.cast(value, default_float())

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
    path = Path(pathname)
    path.mkdir(exist_ok=True, parents=True)

    parameter_names = list(dict_parameters.keys())
    num_parameters = len(parameter_names)

    num_samples = constrained_samples[0].shape[0]
    x = list(range(num_samples))
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


def main():
    np.random.seed(100)
    tf.random.set_seed(100)


    initial_lengthscale = 20.0
    initial_variance =1.0
    data1 = pd.read_csv('../data/LB_GP_TS.csv', sep=",")
    data2 = pd.read_csv('../data/Exon_intron_counts_data_normalizedbylibrarydepthonly_20200120.txt',sep=" ")

    t0 = np.array((95.0,105.0,115.0,125.0,145.0,160.0,175.0,190.0,205.0,220.0))
    rep_no = 3
    t = np.hstack((t0,t0,t0))[:,None]
    t0 = np.array((95.0,105.0,115.0,125.0,145.0,160.0,175.0,190.0,205.0,220.0))

    D = []
    S = []
    m_est = []
    good_gene = []

    df = pd.DataFrame({
       'name': [],
       'value': [],
       'ci_lower': [],
       'ci_upper': []})

    ## Choose the index for the gene_id/tr_id
    # names_transcripts =  pd.read_csv('../data/int_c5_zygc1.csv', sep=",")
    # i=0
    # gene_id = names_transcripts['gene_id'].iloc[i]
    # tr_id = names_transcripts['zygotic_transcript'].iloc[i]

    # specify gene and transcription id
    gene_id = 'FBgn0015773'
    tr_id = 'FBtr0308337'

    #gene_id = "FBgn0000395"
    #tr_id = "FBtr0071610"

    data, observations, gene_id, data_p, observations_p = load_single_gene(gene_id, tr_id)

    '''
    Fit GP regression on pre-mRNA
    '''

    # Estimate a single GP based on pre-mRNA data
    data_full = (dfloat(np.asarray(data_p[0]).reshape(-1,1)), dfloat(np.asarray(data_p[1]).reshape(-1,1)))
    #print(data_full)

    init_lengthscale = 10.0
    init_variance = ((np.max(np.asarray(data_p[1]))-np.min(np.asarray(data_p[1])))/2)**2
    m, lik = fit_rbf(data_full, init_lengthscale, init_variance)

    '''
    Fit trcd
    '''

    initial_lengthscale = m.kernel.lengthscales.value()
    initial_variance = m.kernel.variance.value()

    initial_S = 0.5
    initial_D = 0.1

    trcd, dict_parameters = create_trcd_model(data, initial_lengthscale, initial_variance, initial_S,initial_D, transform_base=None)

    dict_parameters = select_parameters(dict_parameters,
                                        names=None)  # When `names` is None, same dictionary is returned.

    # Choose to sample only for subset of parameters
    #dict_parameters = select_parameters(dict_parameters,
    #                                    names=['D', 'varaince', 'lengthscale', 'variance_m', 'variance_p'])  # When `names` is None, same dictionary is returned.

    ## WARN: The order of parameters is quite important here,
    ## as we pass them around and use same order for plotting and setting titles for plots.
    ## For that reason we use ordered dictionary.
    parameters = list(dict_parameters.values())
    initial_values = [p.read_value() for p in parameters]

    optimize_with_scipy_optimizer(trcd, parameters)
    parameters_vector = tf.stack(parameters)

    print_summary(trcd)

    ## NOTE: Static parameters for the MCMC.
    ##
    hmc_parameters = HMCParameters(num_samples=1000, num_burnin_steps=1000)
    #hmc_helper, run_chain = create_standard_mcmc(trcd, parameters)
    hmc_helper, run_chain = create_mala_mcmc(trcd, parameters, hmc_parameters)
    #hmc_helper, run_chain = create_nuts_mcmc(trcd, parameters, hmc_parameters=hmc_parameters)

    ## NOTE: Parameter grid for MCMC. Replace these values
    # NOTE: Train model with BFGS optimizer and print optimal values
    #
    optimize_with_scipy_optimizer(trcd, parameters)  # NOTE: Updates TRCD model parameters in place!
    var_m_noise = trcd.model.likelihood.variance_m.value()
    var_p_noise = trcd.model.likelihood.variance_p.value()
    plot_trcd_predict(1, trcd, tr_id, gene_id, observations, var_m_noise, var_p_noise)

    print(f"log posterior density at optimum: {trcd.model.log_posterior_density()}")

    # dict_parameters = select_parameters(dict_parameters,
    #                                     names=['S', 'D'])  # When `names` is None, same dictionary is returned.
    # parameters = list(dict_parameters.values())
    # initial_values = [p.read_value() for p in parameters]
    # parameters_vector = tf.stack(parameters)


    gpflow.config.set_default_positive_minimum(1e-6)
    step_size_mala = np.array([10**-5 , 10**-4 , 10**-3 , 0.003, 0.005, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.5, 1])#
    #step_size_mala = np.array([ 0.0008, 0.0009, 0.001, 0.002, 0.003, 0.004])#
    print(step_size_mala)
    for j in range(step_size_mala.shape[0]):
        step_size = step_size_mala[j]

        try:
            experiment_print(j, "Run MCMC")
            #num_leapfrog, step_size = leapfrog_num_and_step_size[j]
            pathname = f"../output/mcmc_results/results mala- {gene_id}-{tr_id}-{step_size}"
            experiment_print(j, f"Save results at '{pathname}'")

            samples = run_mala(run_chain,
                               hmc_helper,
                               step_size=step_size_mala[j])

            analyse_samples(gene_id, tr_id, parameters_vector, step_size, *samples, dict_parameters=dict_parameters, pathname=pathname)
            #reset_parameters(parameters, initial_values)
            experiment_print(j, f"MCMC finished")
        except:
            print('Experiment error')


if __name__ == "__main__":
    main()
