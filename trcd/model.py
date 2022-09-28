from typing import Optional, Tuple

import numpy as np

import tensorflow as tf
import gpflow
from .utils import positive_parameter
from .kernels import Kernel_mRNA, White_mRNA#, Kernel_mRNA_discontinuous
#from .mean_functions import MeanFunction_mRNA

from .model_gpr import GPR_test
from .likelihood import Gaussian_test
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

__all__ = ["TRCD"]


Data = Tuple[tf.Tensor, tf.Tensor]
#T_break = Tuple[tf.Tensor]

class TRCD(tf.Module):
    def __init__(self,
                 data: Data,
                 S: float = 0.3,
                 D: float = 0.2,
                 variance: float = 1.0,
                 variance_m: float = 10.0,
                 variance_p: float = 1.0,
                 lengthscale: float = 15.0,
                 #variance_x_white: float = 0.25,
                 #variance_f_white: float = 0.25,
                 #m0: float = 0.0,
                 #b0: float = 0.0,
                 #b: float = 0.0,
                 #x0: float = 0.0,
                 #mp: float = 0.0,
                 #use_mrna_mean_function: bool = False,
                 #use_delay: bool = True):
                 ):
        super().__init__()

        self.D = positive_parameter(D)
        #self.delay = positive_parameter(delay)

        # Define mean and kernel
        #kernel_1 = Kernel_mRNA(S, self.D, variance, lengthscale)
        #kernel_2 = White_mRNA(variance_x_white, variance_f_white)
        #kernel = kernel_1 + kernel_2
        #k = kernel_1 + kernel_2

        k = Kernel_mRNA(S, self.D, variance, lengthscale)

        #if use_mrna_mean_function:
        #    mean_function = MeanFunction_mRNA(m0, b0, b, x0, mp, self.D, delay)
        #else:
        mean_function = None

        # here we would just use GPR2
        #self.model = gpflow.models.GPR(data, kernel, mean_function=mean_function)
        self.model = GPR_test(data, kernel = k, likelihood=Gaussian_test(variance_m, variance_p))

    def objective(self, data: Tuple[tf.Tensor, tf.Tensor]):
        return self.model.neg_log_marginal_likelihood(data)

    def call(self, x_input, data):
        return self.model.predict_y(x_input, data)


class m_premRNA(tf.Module):
    def __init__(self,
                 data: Data):
        super().__init__()

        # Define mean and kernel
        k = gpflow.kernels.RBF()
        self.model = gpflow.models.GPR(data, kernel=k, mean_function=None)

    def objective(self, data: Tuple[tf.Tensor, tf.Tensor]):
        return self.model.neg_log_marginal_likelihood(data)

    def call(self, x_input, data):
        return self.model.predict_y(x_input, data)
