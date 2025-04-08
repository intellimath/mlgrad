#

import numpy as np

cdef class DistrFunc:
    #
    cdef double _evaluate_probability(double x):
        return 0
    #
    cdef void _evaluate_probability_array(double[::1] xs, double[::1] ys):
        cdef Py_ssize_t i, n = xs.shape[0]

        for i in range(n):
            ys[i] = self._evaluate_probability(xs[i])
    #
    def evaluate_probability_array(X, Y=None):
        if Y is None:
            Y = inventory.empty_array(len(X))
        self._evaluate_probability_array(X, Y)
        return Y
    #
    cdef double _evaluate_density(double x):
        return 0
    #
    cdef void _evaluate_density_array(double[::1] xs, double[::1] ys):
        cdef Py_ssize_t i, n = xs.shape[0]

        for i in range(n):
            ys[i] = self._evaluate_density(xs[i])
    #
    def evaluate_density_array(X, Y=None):
        if Y is None:
            Y = inventory.empty_array(len(X))
        self._evaluate_density_array(X, Y)
        return Y
    #
    cdef double _evaluate_loss(double x):
        return -log(self._evaluate_density(x))
    #
    cdef void _evaluate_loss_array(double[::1] xs, double[::1] ys):
        cdef Py_ssize_t i, n = xs.shape[0]

        for i in range(n):
            ys[i] = self._evaluate_loss(xs[i])
    #
    def evaluate_loss_array(X, Y=None):
        if Y is None:
            Y = inventory.empty_array(len(X))
        self._evaluate_loss_array(X, Y)
        return Y
    #
    cdef double _derivative_loss(double x):
        return 0
    #
    cdef void _derivative_loss_array(double[::1] xs, double[::1] ys):
        cdef Py_ssize_t i, n = xs.shape[0]

        for i in range(n):
            ys[i] = self._derivative_loss(xs[i])
    #
    def derivative_loss_array(X, Y=None):
        if Y is None:
            Y = inventory.empty_array(len(X))
        self._derivative_loss_array(X, Y)
        return Y
    #
    cdef double _derivative_loss_loc(double x):
        return 0
    #
    cdef void _derivative_loss_loc_array(double[::1] xs, double[::1] ys):
        cdef Py_ssize_t i, n = xs.shape[0]

        for i in range(n):
            ys[i] = self._derivative_loss_loc(xs[i])
    #
    def derivative_loss_loc_array(X, Y=None):
        if Y is None:
            Y = inventory.empty_array(len(X))
        self._derivative_loss_loc_array(X, Y)
        return Y
    #
    cdef double _derivative_loss_scale(double x):
        return 0
    #
    cdef void _derivative_loss_scale_array(double[::1] xs, double[::1] ys):
        cdef Py_ssize_t i, n = xs.shape[0]

        for i in range(n):
            ys[i] = self._derivative_loss_scale(xs[i])
    #
    def derivative_loss_scale_array(X, Y=None):
        if Y is None:
            Y = inventory.empty_array(len(X))
        self._derivative_loss_scale_array(X, Y)
        return Y
    #
        
cdef class SigmoidalDistrFunc(DistrFunc):
    #
    def __init__(self, loc=0, scale=1):
        self.loc = loc
        self.scale = scale
    #
    cdef double _evaluate_probability(double x):
        cdef double v = (x - self.loc) / self.scale

        if v >= 0:
            return 1 / (1 + exp(-v))
        else:
            v = exp(v)
            return v / (1 + v)
    #
    cdef double _evaluate_density(double x):
        cdef double v = (x - self.loc) / self.scale
        cdef double z

        if v >= 0:
            v = exp(-v)
        else:
            v = exp(v)
        z = v + 1
        return (1 / self.scale) * v / (z * z)
    #
    cdef double _evaluate_loss(double x):
        cdef double v = (x - self.loc) / self.scale
        cdef double s = log(self.scale) + v

        if v >= 0:
            return s + 2 * log(1 + exp(-v))
        else:
            v = exp(v)
            return s + 2 * log(v / (1 + v))
    #        
    cdef double _derivative_loss(double x):
        cdef double v = (x - self.loc) / self.scale

        if v >= 0:
            v = exp(-v)
            return (1/self.scale) * (1 - v) / (1 + v)
        else:
            v = exp(v)
            return (1/self.scale) * (v - 1) / (v + 1)
    #
    cdef double _derivative_loss_loc(double x):
        cdef double v = (x - self.loc) / self.scale

        if v >= 0:
            v = exp(-v)
            return (-1/self.scale) * (1 - v) / (1 + v)
        else:
            v = exp(v)
            return (-1/self.scale) * (v - 1) / (v + 1)
    #
    cdef double _scale_next(self, double[::1] X):
        cdef double scale = self.scale
        cdef double loc = self.loc
        cdef double xx_k, zz_k, v, S
        cdef Py_ssize_t k, N = X.shape[0]

        S = 0
        for k in range(N):
            xx_k = X[k] - loc
            zz_k = xx_k / scale
            if zz_k >= 0:
                v = exp(-zz_k)
                S += xx_k * ((1 - v) / (1 + v))
            else:
                v = exp(zz_k)
                S += xx_k * ((v - 1) / (v + 1))
        return S / N
    #
    cdef double _loc_next(self, double[::1] X):
        cdef double scale = self.scale
        cdef double loc = self.loc
        cdef double xx_k, zz_k, v, S, V
        cdef Py_ssize_t k, N = X.shape[0]

        S = 0
        V = 0
        for k in range(N):
            xx_k = X[k] - loc
            zz_k = xx_k / scale
            if zz_k >= 0:
                v = exp(-zz_k)
                v = (1 - v) / (1 + v)) / xx_k
            else:
                v = exp(zz_k)
                v = (v - 1) / (v + 1)) / xx_k
            S += v * X[k]
            V += v
        return S / V
    #
            
        
