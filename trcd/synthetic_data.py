from bisect import bisect_left

import numpy as np

import gpflow

__all__ = ["setup_problem", "setup_full_data"]


def generate_xm(tp, f_sample, D: np.ndarray, S: np.ndarray):
    """
    Estimated mRNA level (x^m) given parameters B,S,D
    """
    delta = tp[2] - tp[1]
    integral_pt1 = (f_sample * np.exp(D * tp)) * delta
    integral_pt2 = np.cumsum(integral_pt1, axis=1)
    integral_final = integral_pt2 * S * np.exp(-D * tp)
    return integral_final


def observed_tp_indices(tp, length, t0):
    """
    Searchers for points in the time grid closest to the observed time points.
    """
    return [bisect_left(tp, value_t) for idx_t, value_t in enumerate(t0)]


def setup_full_data(num_tp: int = 100):
    tp_origin = np.linspace(100,220, num_tp)  # grid for GP
    tp_origin = tp_origin[:, None]
    lengthscale = 0.5 * max(tp_origin)  # set lengthscale

    kernel = gpflow.kernels.RBF(variance=10.0, lengthscales=lengthscale)  # set kernel
    Kff = kernel.K(tp_origin)  # cov function for f
    mean = np.zeros(Kff.shape[0])

    f_origin = np.random.multivariate_normal(mean, Kff, 1).T  # sample f with zero mean and cov K_f

    return tp_origin, f_origin  # [N, 1], [N, 1]


def setup_problem(tp_full: np.ndarray, f_full: np.ndarray, length: int = 10):

    # Observed time points
    inds_obs = np.linspace(20, tp_full.shape[0] - 1, length).astype(int)
    tp_obs = tp_full[inds_obs].copy()

    # Set parameters of the model to simulate the data
    B = 0.0
    S = 1.0
    D = 2.0
    l = 5.0

    # Standard deviation of the noise in mRNA and pre-mRNA respectively
    stdev_m = 0.5
    stdev_p = 0.5

    # Simulate mRNA based on sampled GP
    f_sample = f_full.copy()
    x_all = generate_xm(tp_full, f_sample, D, S)

    # Take only observed points out of sampled data points
    indices = observed_tp_indices(tp_full, length, tp_obs)  # Get indexes for the observed points
    x = x_all[indices]
    y = x + np.random.normal(0, stdev_m, length)[:, None]

    ym = y.copy()
    yp = f_full[indices] + np.random.normal(0, stdev_p, 10)[:, None]

    return tp_obs, ym, yp

    # NOTE: From tf_trcd_multi.py changes
    # #y = x + np.random.normal(0, stdev_m, length)[:, None]
    # cov_m = [[stdev_m, 0, 0], [0, stdev_m, 0], [0, 0, stdev_m]]
    # y = np.repeat(x, 3).reshape(3, 10) + np.random.multivariate_normal([0, 0, 0], cov_m, 10).T

    # ym = y.copy()

    # cov_p = [[stdev_p, 0, 0], [0, stdev_p, 0], [0, 0, stdev_p]]
    # yp = np.repeat(f_full[indices], 3).reshape(3, 10) + np.random.multivariate_normal([0, 0, 0], cov_p, 10).T

    # ym = ym.flatten()
    # yp = yp.flatten()
    # tp_obs = np.repeat(tp_obs, 3)

    #    y = x + np.random.normal(0, stdev_m, length)[:, None]

    # ym = y.copy()
    # yp = f_full[indices] + np.random.normal(0, stdev_p, 10)[:, None]

    # return tp_obs, ym, yp
