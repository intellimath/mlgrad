# coding: utf-8

# The MIT License (MIT)
#
# Copyright (c) <2015-2024> <Shibzukhov Zaur, szport at gmail dot com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

cimport cython

import numpy as np
from mlgrad.models import as_array1d

cdef double double_max = PyFloat_GetMax()
cdef double double_min = PyFloat_GetMin()

cdef class Func2:

    cdef double _evaluate(self, double[::1] X):
        return 0
    
    cdef void _gradient(self, double[::1] X, double[::1] grad):
        pass
    #
    cdef double _gradient_j(self, double[::1] X, Py_ssize_t j):
        return 0
    #
    def __call__(self, X):
        cdef double[::1] x1d = as_array1d(X)
        return self._evaluate(x1d)

cdef class PowerNorm(Func2):
    
    def __init__(self, p=2.0):
        self.p = p
#         self.all = all

    cdef double _evaluate(self, double[::1] X):
        cdef Py_ssize_t i, m
        cdef double s, v
        cdef double* X_ptr = &X[0]

        s = 0
        for i in range(X.shape[0]):
            v = X_ptr[i]
            if v >= 0:
                s += pow(v, self.p)
            else:
                s += pow(-v, self.p)
        
        s /= self.p
        return s

    cdef void _gradient(self, double[::1] X, double[::1] grad):
        cdef Py_ssize_t i, m
        cdef double v
        cdef double* X_ptr = &X[0]
        cdef double* grad_ptr = &grad[0]
    
        for i in range(X.shape[0]):
            v = pow(fabs(X_ptr[i]), self.p-1.0)
            if v < 0:
                grad_ptr[i] = -v
            else:
                grad_ptr[i] = v
    #
    cdef double _gradient_j(self, double[::1] X, Py_ssize_t j):
        cdef double v = pow(fabs(X[j]), self.p-1.0)

        if v < 0:
             v = -v
        return v
    #
    def _repr_latex_(self):
        return r"$||\mathbf{w}||_{%s}^{%s}=\sum_{i=0}^n w_i^{%s}$" % (self.p, self.p, self.p)

cdef class SquareNorm(Func2):

    cdef double _evaluate(self, double[::1] X):
        cdef Py_ssize_t i, m
        cdef double s, v
        cdef double* X_ptr = &X[0]

        s = 0
        for i in range(X.shape[0]):
            v = X_ptr[i]
            s += v * v

        s /= 2.
        return s

    cdef void _gradient(self, double[::1] X, double[::1] grad):
        cdef Py_ssize_t i, m
        cdef double* X_ptr = &X[0]
        cdef double* grad_ptr = &grad[0]

        for i in range(X.shape[0]):
            grad_ptr[i] = X_ptr[i]    
    #
    cdef double _gradient_j(self, double[::1] X, Py_ssize_t j):
        return X[j]
    #
    def _repr_latex_(self):
        return r"$||\mathbf{w}||_2^2=\sum_{i=0}^n w_i^2$"
        

cdef class AbsoluteNorm(Func2):

    cdef double _evaluate(self, double[::1] X):
        cdef Py_ssize_t i
        cdef double s
        cdef double* X_ptr = &X[0]

        s = 0
        for i in range(X.shape[0]):
            s += fabs(X_ptr[i])
        return s

    cdef void _gradient(self, double[::1] X, double[::1] grad):
        cdef Py_ssize_t i, m
        cdef double* X_ptr = &X[0]
        cdef double* grad_ptr = &grad[0]

        for i in range(X.shape[0]):
            v = X_ptr[i]
            if v > 0:
                grad_ptr[i] = 1
            elif v < 0:
                grad_ptr[i] = -1
            else:
                grad_ptr[i] = 0
    #
    cdef double _gradient_j(self, double[::1] X, Py_ssize_t j):
        cdef double v = X[j]
        
        if v < 0:
            v = -v
        return v
    #    
    def _repr_latex_(self):
        return r"$||\mathbf{w}||_1=\sum_{i=0}^n |w_i|$"

cdef class SoftAbsoluteNorm(Func2):

    def __init__(self, eps=0.001):
        self.eps = eps
        self.eps2 = eps * eps

    cdef double _evaluate(self, double[::1] X):
        cdef Py_ssize_t i
        cdef double s, v
        cdef double* X_ptr = &X[0]

        s = 0
        for i in range(X.shape[0]):
            v = X_ptr[i]
            s += v * v
        return sqrt(self.eps2 + s) - self.eps

    cdef void _gradient(self, double[::1] X, double[::1] grad):
        cdef Py_ssize_t i
        cdef double s, v
        cdef double* X_ptr = &X[0]
        cdef double* grad_ptr = &grad[0]

        s = 0
        for i in range(X.shape[0]):
            grad_ptr[i] = v = X_ptr[i]
            s += v * v
        s = sqrt(self.eps2 + s)
        
        for i in range(X.shape[0]):
            grad_ptr[i] /= s
    #
    cdef double _gradient_j(self, double[::1] X, Py_ssize_t j):
        cdef Py_ssize_t i
        cdef double s, v
        cdef double* X_ptr = &X[0]

        s = 0
        for i in range(X.shape[0]):
            v = X_ptr[i]
            s += v * v
        s = sqrt(self.eps2 + s)
        
        return X_ptr[j] / s
    #    
    def _repr_latex_(self):
        return r"$||\mathbf{w}||_1=\sum_{i=0}^n |w_i|$"
    
cdef class SquareForm(Func2):
    
    def __init__(self, double[:,::1] matrix):
        if matrix.shape[0] != matrix.shape[1]-1:
            raise RuntimeError("Invalid shape: (%s,%s)" % (matrix.shape[0], matrix.shape[1]))
        self.matrix = matrix
    #
    cdef double _evaluate(self, double[::1] x):
        cdef double[:,::1] mat = self.matrix
        cdef Py_ssize_t n_row = mat.shape[0]
        cdef Py_ssize_t n_col = mat.shape[1]
        cdef double s, val
        cdef Py_ssize_t i, j
        
        val = 0
        for j in range(n_row):
            s = mat[j,0]
            for i in range(1, n_col):
                s += mat[j,i] * x[i-1]
            val += s*s
        return 0.5*val

    cdef void _gradient(self, double[::1] x, double[::1] y):
        cdef double[:,::1] mat = self.matrix
        cdef Py_ssize_t n_row = mat.shape[0]
        cdef Py_ssize_t n_col = mat.shape[1]
        cdef double s
        cdef Py_ssize_t i, j
        
        n_row = mat.shape[0]
        n_col = mat.shape[1]
        
        fill_memoryview(y, 0)
        for j in range(n_row):
            s = mat[j,0]
            for i in range(1, n_col):
                s += mat[j,i] * x[i-1]

            for i in range(1, n_col):
                y[i-1] += s*mat[j,i]

cdef class Rosenbrok(Func2):

    cdef double _evaluate(self, double[::1] X):
        return 10. * (X[1] - X[0]**2)**2 + 0.1*(1. - X[0])**2
    
    cdef void _gradient(self, double[::1] X, double[::1] grad):
        grad[0] = -40. * (X[1] - X[0]**2) * X[0] - 0.2 * (1. - X[0])
        grad[1] = 20. * (X[1] - X[0]**2)
        
        
cdef class Himmelblau(Func2):

    cdef double _evaluate(self, double[::1] X):
        return (X[0]**2 + X[1] - 11)**2 + (X[0] + X[1]**2 - 7)**2
    
    cdef void _gradient(self, double[::1] X, double[::1] grad):
        grad[0] = 4*(X[0]**2 + X[1] - 11) * X[0] + 2*(X[0] + X[1]**2 - 7)
        grad[1] = 2*(X[0]**2 + X[1] - 11) + 4*(X[0] + X[1]**2 - 7) * X[1]
        
cdef class SoftMin(Func2):
    
    def __init__(self, p=1.0):
        self.p = p
        self.evals = None

    cdef double _evaluate(self, double[::1] X):
        cdef Py_ssize_t i, m = X.shape[0]
        cdef double s, v, v_min
        cdef double p = self.p
        
        v_min = double_max
        for i in range(m):
            v = X[i]
            if v < v_min:
                v_min = v
        
        s = 0
        for i in range(m):
            s += exp(p*(v_min - X[i]))

        s = log(s)
        s -= p * v_min
        
        return -s / p

    cdef void _gradient(self, double[::1] X, double[::1] grad):
        cdef Py_ssize_t i, m = X.shape[0]
        cdef double s, v, v_min
        cdef double p = self.p
        # cdef double* grad_ptr = &grad[0]
        cdef double[::1] evals = self.evals

        if evals is None or evals.shape[0] != m:
            evals = self.evals = np.empty(m, 'd')
        
        v_min = double_max
        for i in range(m):
            v = X[i]
            if v < v_min:
                v_min = v

        s = 0
        for i in range(m):
            evals[i] = v = exp(p*(v_min - X[i]))
            s += v

        for i in range(m):
            grad[i] = evals[i] / s
    #
    cdef double _gradient_j(self, double[::1] X, Py_ssize_t j):
        cdef Py_ssize_t i, m = X.shape[0]
        cdef double s, v, v_min
        # cdef double* X_ptr = &X[0]
        cdef double p = self.p

        v_min = double_max
        for i in range(m):
            v = X[i]
            if v < v_min:
                v_min = v

        s = 0
        for i in range(m):
            s += exp(p*(v_min - X[i]))

        return exp(p*(v_min - X[j])) / s 
    #
    def _repr_latex_(self):
        return r"$||\mathbf{w}||_{%s}^{%s}=\sum_{i=0}^n w_i^{%s}$" % (self.p, self.p, self.p)

cdef class SoftMax(Func2):
    
    def __init__(self, p=1.0):
        self.p = p
        self.evals = None

    cdef double _evaluate(self, double[::1] X):
        cdef Py_ssize_t i, m = X.shape[0]
        cdef double s, v, v_max
        cdef double p = self.p
        
        v_max = double_min
        for i in range(m):
            v = X[i]
            if v > v_max:
                v_max = v
        
        s = 0
        for i in range(m):
            s += exp(p*(X[i] - v_max))

        s = log(s)
        s += p * v_max
        
        return s / p

    cdef void _gradient(self, double[::1] X, double[::1] grad):
        cdef Py_ssize_t i, m = X.shape[0]
        cdef double s, v, v_max
        cdef double p = self.p
        cdef double[::1] evals = self.evals

        if evals is None or evals.shape[0] != m:
            evals = self.evals = np.empty(m, 'd')
        
        v_max = double_min
        for i in range(m):
            v = X[i]
            if v > v_max:
                v_max = v

        s = 0
        for i in range(m):
            evals[i] = v = p*exp(p*(X[i] - v_max))
            s += v

        for i in range(m):
            grad[i] = evals[i] / s
    #
    cdef double _gradient_j(self, double[::1] X, Py_ssize_t j):
        cdef Py_ssize_t i, m = X.shape[0]
        cdef double s, v, v_max
        cdef double p = self.p

        v_max = double_min
        for i in range(m):
            v = X[i]
            if v > v_max:
                v_max = v

        s = 0
        for i in range(m):
            s += exp(p*(X[i] - v_max))

        return exp(p*(X[j] - v_max)) / s 
    #

cdef class PowerMax(Func2):
    
    def __init__(self, p=1.0):
        self.p = p
        self.evals = None

    cdef double _evaluate(self, double[::1] X):
        cdef Py_ssize_t i, m = X.shape[0]
        cdef double s, v, v_max
        cdef double p = self.p
        
        v_max = double_min
        for i in range(m):
            v = fabs(X[i])
            if v > v_max:
                v_max = v
        
        s = 0
        for i in range(m):
            v = X[i]
            if v >= 0:
                s += pow(v / v_max, p)
            else:
                s += pow(-v / v_max, p)

        s = pow(s, 1/p)
        s *= v_max
        
        return s

    cdef void _gradient(self, double[::1] X, double[::1] grad):
        cdef Py_ssize_t i, m = X.shape[0]
        cdef double s, v
        cdef double p = self.p
        cdef double[::1] evals = self.evals

        if evals is None or evals.shape[0] != m:
            evals = self.evals = np.empty(m, 'd')

        s = self._evaluate(X)
        
        for i in range(m):
            v = X[i] / s
            evals[i] = p * pow(v, p-1)
    #
    cdef double _gradient_j(self, double[::1] X, Py_ssize_t j):
        cdef Py_ssize_t i
        cdef double s
        cdef double p = self.p

        s = self._evaluate(X)
        return p * pow(X[j] / s, p-1)
    #
