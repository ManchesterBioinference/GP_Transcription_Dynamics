from typing import Optional, Tuple

import numpy as np

import tensorflow as tf
import gpflow
from .utils import positive_parameter
from .kernels import Kernel_mRNA, White_mRNA#, Kernel_mRNA_discontinuous
from .mean_functions import MeanFunction_mRNA

from .model_gpr import GPR_test
from .likelihood import Gaussian_test
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

__all__ = ["TRCD"]


Data = Tuple[tf.Tensor, tf.Tensor]
#T_break = Tuple[tf.Tensor]

class Kernel_mRNA_discontinuous(gpflow.kernels.Kernel):
    def __init__(self, S: tf.Tensor, t_break, D0: tf.Tensor, D1: tf.Tensor,  variance: tf.Tensor, lengthscale: tf.Tensor):
        super().__init__()
        self.S = positive_parameter(S)
        self.t_break = t_break
        self.D0 = positive_parameter(D0)
        self.D1 = positive_parameter(D1)
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

        #print('t_break', t_break.numpy())
        #t_break = int(t_break.numpy())
        t_break = int(self.t_break)

        assert a_input.shape[0] % 2 == 0
        assert b_input.shape[0] % 2 == 0

        n = a_input.shape[0] // 2
        m = b_input.shape[0] // 2

        ##t_break = self.t_break
        D0 = self.D0
        D1 = self.D1
        S = self.S
        l = self.lengthscale
        v = self.variance

        print('t_break', t_break)


        gamma0 = 0.5 * D0 * l
        gamma1 = 0.5 * D1 * l

        #diff_t = t - t_prime

        #gamma_mat = np.zeros((diff_t.shape))
        #gamma_mat[0:t_break,0:t_break] = 0.5 * D0 * l
        #gamma_mat[t_breakDmat.shape[0],t_break:Dmat.shape[1]] = 0.5 * D1 * l

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


            #a0 = 0.5 * tf.math.exp(gamma**2) / D0
            #a1 = 0.5 * tf.math.exp(gamma**2) / D1
            print('t_break in h(t,t)', t_break)
            diff_t = t - t_prime
            #Dmat = np.zeros((diff_t.shape))
            Dmat = tf.zeros(diff_t.shape, tf.float64)
            # !!! Dmat = tf.slice((Dmat, [range(0:t_break),range(0:t_break)]))
            print('Dmat', Dmat)
            print(tf.assign(Dmat[0:t_break,0:t_break] , [[D0]] ))
            Dmat[0:t_break,0:t_break] = D0
            print(Dmat)
            Dmat[t_break:Dmat.shape[0],t_break:Dmat.shape[1]] = D1

            gamma_mat = np.zeros((diff_t.shape))
            gamma_mat[0:t_break,0:t_break] = 0.5 * D0 * l
            gamma_mat[t_breakDmat.shape[0],t_break:Dmat.shape[1]] = 0.5 * D1 * l

            a_mat = np.zeros((diff_t.shape))
            a_mat[0:t_break,0:t_break] = D0
            a_mat[t_break:a_mat.shape[0],t_break:a_mat.shape[1]] = D1

            #b = tf.math.exp(-D * diff_t)
            b = tf.math.exp(-Dmat * diff_t)
            c = tf.math.erf(diff_t / l - gamma_mat) + tf.math.erf(t_prime / l + gamma_mat)
            #d = tf.math.exp(-D * (t + 1))
            d = tf.math.exp(-Dmat * (t + t_prime)) # changes based on file derivation_asynchronous4.pdf
            e = tf.math.erf(t / l - gamma_mat) + tf.math.erf(gamma_mat)
            return a * (b * c - d * e)

        def k_xf(t, t_prime):
            diff_t = t - t_prime
            Dmat = np.zeros((diff_t.shape))
            Dmat[0:t_break,0:t_break] = D0
            Dmat[t_break:Dmat.shape[0],t_break:Dmat.shape[1]] = D1

            gamma_mat = np.zeros((diff_t.shape))
            gamma_mat[0:t_break,0:t_break] = 0.5 * D0 * l
            gamma_mat[t_breakDmat.shape[0],t_break:Dmat.shape[1]] = 0.5 * D1 * l

            a = tf.math.exp(-D_mat * diff_t)
            b = tf.math.erf(diff_t / l - gamma_mat) + tf.math.erf(t_prime / l + gamma_mat)
            return xf_const * tf.math.exp(gamma_mat**2) * a * b

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
        D0 = self.D0
        D1 = self.D1
        S = self.S
        t_break = int(self.t_break)

        n = a_input.shape[0] // 2
        pi = tf.convert_to_tensor(np.pi, dtype=l.dtype)
        xx_const = 0.5 * S**2 * tf.math.sqrt(pi) * l
        gamma = 0.5 * D * l

        gamma_mat = np.zeros((x.shape))
        gamma_mat[0:t_break] = 0.5 * D0 * l
        gamma_mat[t_break:x.shape[0]] = 0.5 * D1 * l

        Dmat = np.zeros((x.shape))
        Dmat[0:t_break,0:t_break] = D0
        x_shape = x.shape[0]
        Dmat[t_break:x_shape,t_break:x_shape] = D1

        x = a_input[0:n]

        a = 0.5 * tf.math.exp(gamma_mat**2) / D
        b = tf.math.erf(-gamma_mat) + tf.math.erf(x / l + gamma_mat)
        c = tf.math.exp(-2.0 * D * x)
        d = tf.math.erf(x / l - gamma_mat) + tf.math.erf(gamma_mat)

        upper_block = tf.reshape(xx_const * 2.0 * a * (b - c * d), [-1])
        lower_block = tf.fill([n], tf.squeeze(v))

        return tf.concat([upper_block, lower_block], axis=0)


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

class Kernel_mRNA_delay(gpflow.kernels.Kernel):
    def __init__(self, S: tf.Tensor, D: tf.Tensor, variance: tf.Tensor, lengthscale: tf.Tensor, delay: tf.Tensor):
        super().__init__()
        self.S = positive_parameter(S)
        self.D = positive_parameter(D)
        self.variance = positive_parameter(variance)
        self.lengthscale = positive_parameter(lengthscale)
        self.delay = positive_parameter(delay)

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
        delay = self.delay
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
            t = t - delay
            t_prime = t_prime - delay
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
            t = t - delay
            t_prime = t_prime 
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


class TRCD_discontinuous(tf.Module):
    def __init__(self,
                 data: Data,
                 t_break: int = 7,
                 S: float = 1.0,
                 D0: float = 0.2,
                 D1: float = 0.2,
                 variance: float = 1.0,
                 variance_m: float =1.0,
                 variance_p: float = 1.0,
                 lengthscale: float = 5.0,
                 #variance_x_white: float = 0.25,
                 #variance_f_white: float = 0.25,
                 m0: float = 0.0,
                 b0: float = 0.0,
                 b: float = 0.0,
                 x0: float = 0.0,
                 mp: float = 0.0,
                 #delay: float = 0.01,
                 use_mrna_mean_function: bool = False):
        super().__init__()

        self.D0 = positive_parameter(D0)
        self.D1 = positive_parameter(D1)
        #self.delay = positive_parameter(delay)

        # Define mean and kernel
        #kernel_1 = Kernel_mRNA(S, self.D, variance, lengthscale)
        #kernel_2 = White_mRNA(variance_x_white, variance_f_white)
        #kernel = kernel_1 + kernel_2
        #k = kernel_1 + kernel_2

        k = Kernel_mRNA_discontinuous(S, t_break, self.D0, self.D1, variance, lengthscale)


        if use_mrna_mean_function:
            mean_function = MeanFunction_mRNA(m0, b0, b, x0, mp, self.D, delay)
        else:
            mean_function = None

        # here we would just use GPR2
        #self.model = gpflow.models.GPR(data, kernel, mean_function=mean_function)
        self.model = GPR_test(data, kernel = k, likelihood=Gaussian_test(variance_m, variance_p))

    def objective(self, data: Tuple[tf.Tensor, tf.Tensor]):
        return self.model.neg_log_marginal_likelihood(data)

    def call(self, x_input, data):
        return self.model.predict_y(x_input, data)

class TRCD_delay(tf.Module):
    def __init__(self,
                 data: Data,
                 S: float = 1.0,
                 D: float = 0.2,
                 variance: float = 1.0,
                 variance_m: float =1.0,
                 variance_p: float = 1.0,
                 lengthscale: float = 5.0,
                 #variance_x_white: float = 0.25,
                 #variance_f_white: float = 0.25,
                 m0: float = 0.0,
                 b0: float = 0.0,
                 b: float = 0.0,
                 x0: float = 0.0,
                 mp: float = 0.0,
                 delay: float = 0.01,
                 use_mrna_mean_function: bool = False):
        super().__init__()

        self.D = positive_parameter(D)
        self.delay = positive_parameter(delay)

        # Define mean and kernel
        #kernel_1 = Kernel_mRNA(S, self.D, variance, lengthscale)
        #kernel_2 = White_mRNA(variance_x_white, variance_f_white)
        #kernel = kernel_1 + kernel_2
        #k = kernel_1 + kernel_2

        k = Kernel_mRNA_delay(S, self.D, variance, lengthscale, self.delay)


        if use_mrna_mean_function:
            mean_function = MeanFunction_mRNA(m0, b0, b, x0, mp, self.D, delay)
        else:
            mean_function = None

        # here we would just use GPR2
        #self.model = gpflow.models.GPR(data, kernel, mean_function=mean_function)
        self.model = GPR_test(data, kernel = k, likelihood=Gaussian_test(variance_m, variance_p))

    def objective(self, data: Tuple[tf.Tensor, tf.Tensor]):
        return self.model.neg_log_marginal_likelihood(data)

    def call(self, x_input, data):
        return self.model.predict_y(x_input, data)

class TRCD(tf.Module):
    def __init__(self,
                 data: Data,
                 S: float = 1.0,
                 D: float = 0.2,
                 variance: float = 1.0,
                 variance_m: float =1.0,
                 variance_p: float = 1.0,
                 lengthscale: float = 5.0,
                 #variance_x_white: float = 0.25,
                 #variance_f_white: float = 0.25,
                 m0: float = 0.0,
                 b0: float = 0.0,
                 b: float = 0.0,
                 x0: float = 0.0,
                 mp: float = 0.0,
                 use_mrna_mean_function: bool = False,
                 use_delay: bool = True):
        super().__init__()

        self.D = positive_parameter(D)
        #self.delay = positive_parameter(delay)

        # Define mean and kernel
        #kernel_1 = Kernel_mRNA(S, self.D, variance, lengthscale)
        #kernel_2 = White_mRNA(variance_x_white, variance_f_white)
        #kernel = kernel_1 + kernel_2
        #k = kernel_1 + kernel_2

        k = Kernel_mRNA(S, self.D, variance, lengthscale)

        if use_mrna_mean_function:
            mean_function = MeanFunction_mRNA(m0, b0, b, x0, mp, self.D, delay)
        else:
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
