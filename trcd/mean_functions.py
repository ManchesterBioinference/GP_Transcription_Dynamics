import tensorflow as tf

import gpflow

from .utils import positive_parameter


__all__ = ["MeanFunction_mRNA"]


class MeanFunction_mRNA(gpflow.mean_functions.MeanFunction):
    def __init__(self, m0, b0, b, x0, mp, D, delay):
        super().__init__()
        self.m0 = positive_parameter(m0)
        self.b0 = positive_parameter(b0)
        self.b = positive_parameter(b)
        self.D = positive_parameter(D)
        self.x0 = positive_parameter(x0)
        self.mp = positive_parameter(mp)
        self.delay = positive_parameter(delay)

    def __call__(self, x_input: tf.Tensor):
        n = x_input.shape[0] // 2
        x_mrna = x_input[0:n]
        ## x_premRNA = x_input[0:n]
        mean_pre_mrna = tf.fill([n], tf.squeeze(self.mp))
        a = self.m0 * tf.exp(-self.D * (x_mrna - self.x0))
        b = (self.b0 / self.D) * (1.0 - tf.exp(-self.D * x_mrna))
        c = (self.b * self.mp / self.D) * (1.0 - tf.exp(-self.D * (x_mrna - self.delay)))
        #mean_mrna = a + b + c  # TODO: Should it be used below?
        mean_mrna = b
        return tf.concat([mean_pre_mrna, mean_pre_mrna], axis=0)[:, None]
