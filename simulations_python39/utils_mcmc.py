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
import arviz as az

import gpflow
from gpflow.utilities import print_summary
from gpflow.config import default_float

from trcd.utils import SamplingHelper
#from utilities_dpp import (HMCParameters, create_data, create_nuts_mcmc, create_standard_mcmc, create_trcd_model, handle_pool,
#                       optimize_with_scipy_optimizer, select_parameters, load_single_gene,init_hyperparameters, predict_trcd, create_mala_mcmc,
#                       plot_trcd_predict, compute_prior_hyperparameters_variance)

gpflow.config.set_default_float(np.float64)  # noqa

Scalar = TypeVar("Scalar", tf.Tensor, float)

def experiment_print(step: int, msg: str):
    tf.print(f"# [Step {step}] {msg}")


def reset_parameters(parameters: List[gpflow.Parameter], values: List[tf.Tensor]):
    for p, v in zip(parameters, values):
        p.assign(v)


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

def accrate(mcmc_trace):
    N = len(mcmc_trace)
    count = 0
    for i in range(1,N):
        if (mcmc_trace[i] == mcmc_trace[i-1]):
            count +=0
        else:
            count +=1
    return count/N


def analyse_samples(gene_id, tr_id, parameters_vector , step_size, unconstrained_samples: List[tf.Tensor],
                    constrained_samples: List[tf.Tensor],
                    #accept_log_prob: List[tf.Tensor],
                    #grads: List[tf.Tensor],
                    dict_parameters: dict = None,
                    pathname: str = None):
    """
    Plotting and saving graphs on disk;
    Saving samples and df with HPD summaries.
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

    plt.savefig(f'{pathname}/'+str(gene_id)+'_'+str(tr_id)+'_step_size_'+str(step_size)+'_traces.png')
    print(step_size)
    print(str(step_size))

    acceptance_rate = []
    par_names = []
    for i in range(num_parameters):
        par_names.append(parameter_names[i])
        acceptance_rate.append(accrate(np.array(constrained_samples[i])))

    acc_result = np.concatenate((np.array(par_names).reshape(-1,1), np.array(acceptance_rate).reshape(-1,1)), axis=1)
    df_acc = pd.DataFrame(acc_result)
    df_acc.columns = ['parameter','acceptance_rate']

    print(df_acc)
    df_acc.to_csv(f'{pathname}/accrate_'+str(gene_id)+'_'+str(tr_id)+'_step_size_'+str(step_size)+'.csv')


    res = np.zeros((num_parameters, 2))
    for i in range(num_parameters):
        #hpd = pymc3.stats.hpd(np.asarray(constrained_samples[i]))
        hpd = az.hdi(np.asarray(constrained_samples[i]), hdi_prob=0.95)
        res[i,0] = hpd[0]
        res[i,1] = hpd[1]

    res = np.concatenate((np.array(parameter_names).reshape(-1,1), np.array(parameters_vector).reshape(-1,1), res), axis=1)

    df = pd.DataFrame(res)


    df.columns = ['parameter','MAP','hpd_l', 'hpd_u']

    print(df)
    df = df.round(5)
    df.to_csv(f'{pathname}/hpd_'+str(gene_id)+'_'+str(tr_id)+'_step_size'+str(step_size)+'.csv')

    save_samples = pd.DataFrame(np.asarray(constrained_samples))
    save_samples.to_csv(f'{pathname}/samples_'+str(gene_id)+str(tr_id)+'_step_size_'+str(step_size)+'.csv')


def experiment_print(step: int, msg: str):
    tf.print(f"# [Step {step}] {msg}")


def reset_parameters(parameters: List[gpflow.Parameter], values: List[tf.Tensor]):
    for p, v in zip(parameters, values):
        p.assign(v)


# TODO: it doesn't work yet:
# def run_multiple_chains(run_chain_args, hmc_static_parameters):
#     def run_chain_in_process(model, parameters, step_size: float, num_leapfrog: int):
#         hmc_helper, run_chain = create_nuts_mcmc(model, parameters, hmc_parameters=hmc_static_parameters)
#         return run_mcmc(run_chain,
#                         hmc_helper,
#                         step_size=step_size,
#                         num_leapfrog_steps=num_leapfrog)
#     with handle_pool(Pool(processes=cpu_count())) as pool:
#         return pool.map(run_chain_in_process, run_chain_args)
