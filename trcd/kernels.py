from typing import Optional

import numpy as np
import tensorflow as tf

import gpflow

from .utils import positive_parameter


__all__ = ["Kernel_mRNA", "White_mRNA"]


class Kernel_mRNA(gpflow.kernels.Kernel):
    def __init__(self, S: tf.Tensor, D: tf.Tensor, variance: tf.Tensor, lengthscale: tf.Tensor):
        super().__init__()
        self.S = positive_parameter(S)
        self.D = positive_parameter(D)
        self.variance = positive_parameter(variance)
        self.lengthscale = positive_parameter(lengthscale)

    def K(self, a_input, b_input: Optional[tf.Tensor] = None, presliced: bool = False):
        """
        Args:
            a_input: [N, 1] Tensor.
            b_input: [M, 1] Tensor.
        Returns:
        """

        if b_input is None:
            b_input = a_input

        assert a_input.shape[0] % 2 == 0
        assert b_input.shape[0] % 2 == 0

        n = a_input.shape[0] // 2
        m = b_input.shape[0] // 2

        D = self.D
        S = self.S
        l = self.lengthscale
        v = self.variance

        gamma = 0.5 * D * l
        sqrt_pi = tf.convert_to_tensor(np.sqrt(np.pi), dtype=l.dtype)
        half_lengthscale_sqrt_pi = 0.5 * sqrt_pi * l

        xx_const = S**2 * half_lengthscale_sqrt_pi
        xf_const = S * half_lengthscale_sqrt_pi

        if n == m:
            x1 = a_input[0:n]
            x2 = tf.linalg.adjoint(b_input[0:m])

            x1_f = a_input[n:(2*n+1)]
            x2_f = tf.linalg.adjoint(b_input[m:(2*m+1)])

        else:
            x1 = tf.broadcast_to(a_input[0:n], [n, m])
            x2_tmp = tf.linalg.adjoint(b_input[0:m])
            x2 = tf.broadcast_to(x2_tmp, [n, m])

            x1_f = tf.broadcast_to(a_input[n:(2*n+1)], [n, m])
            x2_tmp_f = tf.linalg.adjoint(b_input[m:(2*m+1)])
            x2_f = tf.broadcast_to(x2_tmp_f, [n, m])

        # Defines h(t',t)
        def h_tt(t, t_prime):
            """ Computes h(t, t') """
            a = 0.5 * tf.math.exp(gamma**2) / D
            diff_t = t - t_prime
            #b = tf.math.exp(-D * diff_t)
            b = tf.math.exp(-D * diff_t)
            c = tf.math.erf(diff_t / l - gamma) + tf.math.erf(t_prime / l + gamma)
            #d = tf.math.exp(-D * (t + 1))
            d = tf.math.exp(-D * (t + t_prime)) # changes based on file derivation_asynchronous4.pdf
            e = tf.math.erf(t / l - gamma) + tf.math.erf(gamma)
            return a * (b * c - d * e)

        def k_xf(t, t_prime):
            diff_t = t - t_prime
            a = tf.math.exp(-D * diff_t)
            b = tf.math.erf(diff_t / l - gamma) + tf.math.erf(t_prime / l + gamma)
            return xf_const * tf.math.exp(gamma**2) * a * b

        kxx = xx_const * (h_tt(x1, x2) + h_tt(x2, x1))
        kxf = k_xf(x1, x2)
        kfx = k_xf(x2, x1)
        kff = v * tf.math.exp(-0.5 * ((x1_f - x2_f) / l)**2)

        # Combine four blocks together
        output_n = 2 * kxx.shape[-2]
        output_m = 2 * kxx.shape[-1]
        k = tf.transpose([[kxx, kxf], [kfx, kff]], [0, 2, 1, 3])
        return tf.reshape(k, [output_n, output_m])

    def K_diag(self, a_input: tf.Tensor, presliced: bool = False):
        assert a_input.shape[0] % 2 == 0

        v = self.variance
        l = self.lengthscale
        D = self.D
        S = self.S

        n = a_input.shape[0] // 2
        pi = tf.convert_to_tensor(np.pi, dtype=l.dtype)
        xx_const = 0.5 * S**2 * tf.math.sqrt(pi) * l
        gamma = 0.5 * D * l

        x = a_input[0:n]

        a = 0.5 * tf.math.exp(gamma**2) / D
        b = tf.math.erf(-gamma) + tf.math.erf(x / l + gamma)
        c = tf.math.exp(-2.0 * D * x)
        d = tf.math.erf(x / l - gamma) + tf.math.erf(gamma)

        upper_block = tf.reshape(xx_const * 2.0 * a * (b - c * d), [-1])
        lower_block = tf.fill([n], tf.squeeze(v))

        return tf.concat([upper_block, lower_block], axis=0)


class White_mRNA(gpflow.kernels.Kernel):
    def __init__(self, variance_x_white: tf.Tensor, variance_f_white: tf.Tensor):
        super().__init__()
        self.variance_x_white = positive_parameter(variance_x_white)
        self.variance_f_white = positive_parameter(variance_f_white)

    def K(self, a_input: tf.Tensor, b_input: tf.Tensor = Optional[tf.Tensor], presliced: bool = False):
        if b_input is None:
            n = a_input.shape[0] // 2
            x = tf.fill([n], tf.squeeze(self.variance_x_white))
            f = tf.fill([n], tf.squeeze(self.variance_f_white))
            xf = tf.concat([x, f], axis=0)
            return tf.linalg.diag(xf)

        n = a_input.shape[0]
        m = b_input.shape[0]
        return tf.zeros([n, m], dtype=a_input.dtype)

    def K_diag(self, a_input: tf.Tensor, presliced: bool = False):
        n = a_input.shape[0] // 2
        x = tf.fill([n], tf.squeeze(self.variance_x_white))
        f = tf.fill([n], tf.squeeze(self.variance_f_white))
        return tf.concat([x, f], axis=0)
