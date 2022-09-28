import numpy as np
import tensorflow as tf
import gpflow
from gpflow.likelihoods import ScalarLikelihood


def positive_parameter(value: tf.Tensor):
    if isinstance(value, (tf.Variable, gpflow.Parameter)):
        return value
    return gpflow.Parameter(value, transform=gpflow.utilities.positive())


class Gaussian_test(ScalarLikelihood):
    r"""
    The Gaussian likelihood is appropriate where uncertainties associated with the data are
    believed to follow a normal distribution, with constant variance.

    Very small uncertainties can lead to numerical instability during the
    optimization process. A lower bound of 1e-6 is therefore imposed on the likelihood variance
    by default.
    """

    def __init__(self, variance_m=1.0, variance_p=1.0, variance_lower_bound=1e-6, **kwargs):
        super().__init__(**kwargs)
        #self.variance = Parameter(variance, transform=positive(lower=variance_lower_bound))
        #self.variance_m = positive_parameter(variance_m)
        self.variance_m = positive_parameter(variance_m)
        self.variance_p = positive_parameter(variance_p)
    def _scalar_log_prob(self, F, Y):
        n=Y.shape[0] // 2
        return logdensities.gaussian(Y[0:n], F, self.variance_m) + logdensities.gaussian(Y[n:(2*n)], F, self.variance_p)

    def _conditional_mean(self, F):  # pylint: disable=R0201
        return tf.identity(F)

    def _conditional_variance(self, F):
        n = tf.shape(F)//2
        s_m = tf.fill([n], self.likelihood.variance_m)
        s_p = tf.fill([n], self.likelihood.variance_p)
        return  tf.concat([s_m, s_p], axis=0)

    def _predict_mean_and_var(self, Fmu, Fvar):
        n = tf.shape(F)//2
        s_m = tf.fill([n], self.likelihood.variance_m)
        s_p = tf.fill([n], self.likelihood.variance_p)

        return tf.identity(Fmu), Fvar + self.variance_m

    def _predict_density(self, Fmu, Fvar, Y):
        n=Y.shape[0] // 2
        return logdensities.gaussian(Y[0:n], Fmu, Fvar + self.variance_m) + logdensities.gaussian(Y[n:(2*n)], Fmu, Fvar + self.variance_p)

    def _variational_expectations(self, Fmu, Fvar, Y):
        return (
            -0.5 * np.log(2 * np.pi)
            - 0.5 * tf.math.log(self.variance_m)
            - 0.5 * ((Y - Fmu) ** 2 + Fvar) / self.variance_m
                 ) #+
            #(
            #    -0.5 * np.log(2 * np.pi)
            #    - 0.5 * tf.math.log(self.variance_m)
            #    - 0.5 * (( - Fmu) ** 2 + Fvar) / self.variance_m
            #         )
