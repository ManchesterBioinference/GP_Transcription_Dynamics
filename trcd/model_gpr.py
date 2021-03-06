import numpy as np
import tensorflow as tf
from bisect import bisect_left
import gpflow
from gpflow.kernels import Kernel
from gpflow.logdensities import multivariate_normal
from gpflow.likelihoods import Likelihood
from gpflow.mean_functions import MeanFunction
from gpflow.models import BayesianModel
from gpflow.models import GPR
from gpflow.config import default_float
from gpflow.base import Parameter
from gpflow.mean_functions import Zero
from gpflow.models.model import MeanAndVariance
from gpflow.models.training_mixins import InternalDataTrainingLossMixin, InputData
from typing import Optional, Tuple

np.random.seed(100)
tf.random.set_seed(100)


Data = Tuple[tf.Tensor, tf.Tensor]
MeanAndVariance = Tuple[tf.Tensor, tf.Tensor]

def positive_parameter(value: tf.Tensor):
    if isinstance(value, (tf.Variable, gpflow.Parameter)):
        return value
    return gpflow.Parameter(value, transform=gpflow.utilities.positive())


class GPR_test(gpflow.models.GPModel, InternalDataTrainingLossMixin):
    r"""
    Gaussian Process Regression.

    This is a vanilla implementation of GP regression with a Gaussian
    likelihood.  Multiple columns of Y are treated independently.

    The log likelihood of this model is sometimes referred to as the 'log
    marginal likelihood', and is given by

    .. math::
       \log p(\mathbf y \,|\, \mathbf f) =
            \mathcal N(\mathbf{y} \,|\, 0, \mathbf{K} + \sigma_n \mathbf{I})
    """

    def __init__(
        self,
        data,
        kernel,
        likelihood,
        mean_function: Optional[MeanFunction] = None,
        noise_variance: float = 1.0):

        #likelihood = gpflow.likelihoods.Gaussian(noise_variance)
        _, y_data = data
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=y_data.shape[-1])
        self.data = data
        #self.likelihood = Likelihood
    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.log_marginal_likelihood()

    def log_marginal_likelihood(self):
        r"""
        Computes the log marginal likelihood.

        .. math::
            \log p(Y | \theta).

        """
        likelihood = self.likelihood
        x, y = self.data
        K = self.kernel(x)
        num_data = x.shape[0]
        k_diag = tf.linalg.diag_part(K)
        s_diag_m = tf.fill([num_data//2], self.likelihood.variance_m)
        s_diag_p = tf.fill([num_data//2], self.likelihood.variance_p)
        s_diag = tf.concat([s_diag_m, s_diag_p], axis=0)
        ks = tf.linalg.set_diag(K, k_diag + s_diag)
        L = tf.linalg.cholesky(ks)
        m = self.mean_function(x)

        # [R,] log-likelihoods for each independent dimension of Y
        log_prob = multivariate_normal(y, m, L)
        return tf.reduce_sum(log_prob) #+ tf.math.log(self.likelihood.variance_p)

    def predict_f(self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False) -> MeanAndVariance:
        r"""
        This method computes predictions at X \in R^{N \x D} input points
        .. math::
            p(F* | Y)
        where F* are points on the GP at new data points, Y are noisy observations at training data points.
        """
        X_data, Y_data = self.data
        err = Y_data - self.mean_function(X_data)

        kmm = self.kernel(X_data)
        knn = self.kernel(Xnew, full_cov=full_cov)
        kmn = self.kernel(X_data, Xnew)

        num_data = X_data.shape[0]
        s_m = tf.fill([num_data//2], self.likelihood.variance_m)
        s_p = tf.fill([num_data//2], self.likelihood.variance_p)
        s_mp = tf.concat([s_m, s_p], axis=0)
        s = tf.linalg.diag(s_mp)
        #s = tf.linalg.diag(tf.fill([num_data], self.likelihood.variance))

        conditional = gpflow.conditionals.base_conditional
        f_mean_zero, f_var = conditional(
            kmn, kmm + s, knn, err, full_cov=full_cov, white=False
        )  # [N, P], [N, P] or [P, N, N]
        f_mean = f_mean_zero + self.mean_function(Xnew)
        return f_mean, f_var



def dfloat(value):  # default float
    return tf.cast(value, default_float())
