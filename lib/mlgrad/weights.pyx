# coding: utf-8

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: embedsignature=True
# cython: initializedcheck=False
# cython: unraisable_tracebacks=True  

# The MIT License (MIT)
#
# Copyright (c) <2015-2021> <Shibzukhov Zaur, szport at gmail dot com>
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

import numpy as np
from libc.math cimport sqrt #, fabs, fmax, exp, log, atan
from mlgrad.func cimport CompSqrt

cdef class Weights(object):
    #
    @property
    def qval(self):
        return self.get_qvalue()
    #
    cpdef init(self):
        pass
    #
    cpdef eval_weights(self):
        pass
    #
    cpdef double[::1] get_weights(self):
        return None
    
    cpdef double get_qvalue(self):
        return 0
    #
    cpdef set_param(self, name, val):
        pass

cdef class ArrayWeights(Weights):
    #
    def __init__(self, weights):
        self.weights = np.asarray(weights)
    #
    cpdef double[::1] get_weights(self):
        return self.weights

cdef class ConstantWeights(Weights):
    #
    def __init__(self, N):
        self.weights = np.full((N,), 1.0/N, 'd')
    #
    cpdef eval_weights(self):
        pass
    #
    cpdef double[::1] get_weights(self):
        return self.weights
    
cdef class RWeights(Weights):
    #
    def __init__(self, Func func, Functional risk, normalize=1):
        self.func = func
        self.risk = risk
        self.weights = None
        self.lval_all = None
        self.normalize = normalize
        N = risk.batch.size
        self.lval_all = np.zeros(N, 'd')
        self.weights = np.zeros(N, 'd')
    #
    cpdef eval_weights(self):
        cdef Risk risk = self.risk
        cdef Py_ssize_t N = risk.batch.size
        cdef Py_ssize_t j, k
        cdef Py_ssize_t[::1] indices = risk.batch.indices
        cdef double[::1] weights = self.weights
        cdef Model mod = self.risk.model
        cdef double[:,::1] X = self.risk.X
        cdef double[::1] Y = self.risk.Y
        cdef Func func = self.func
        cdef double val

        for j in range(N):
            k = indices[j]
            val = mod.evaluate(X[k])
            weights[j] = func.derivative_div_x(val - Y[k])
        
        if self.normalize:
            normalize_memoryview(weights)
    #
    cpdef double get_qvalue(self):
        cdef double qval
        cdef Py_ssize_t j, k
        cdef Py_ssize_t N = self.risk.batch.size
        cdef Py_ssize_t[::1] indices = self.risk.batch.indices
        cdef Model mod = self.risk.model
        cdef double[:,::1] X = self.risk.X
        cdef double[::1] Y = self.risk.Y
        cdef Func func = self.func
        cdef double val
        
        qval = 0
        for j in range(N):
            k = indices[j]
            val = mod.evaluate(X[k])
            qval += func.evaluate(val - Y[k])
        qval /= N 
        return qval
    #
    cpdef double[::1] get_weights(self):
        return self.weights

cdef class MWeights(Weights):
    #
    def __init__(self, Average average, Functional risk, normalize=0, use_best_u=0):
        self.average = average
        self.risk = risk
        self.weights = None
        self.lval_all = None
        self.first_time = 1
        self.normalize = normalize
        self.best_u = PyFloat_GetMax()
        self.use_best_u = use_best_u
        N = len(risk.batch)
        self.weights = np.zeros(N, 'd')
        self.lval_all = np.zeros(N, 'd')
    #
    cpdef init(self):
        self.first_time = 1
    #
    cpdef eval_weights(self):
        cdef Risk risk = <Risk>self.risk
        cdef Py_ssize_t N = risk.batch.size

        risk.eval_losses(self.lval_all)

        if self.first_time:
            u0 = None
            self.first_time = 0 
        else:
            u0 = self.average.u
        
        self.average.fit(self.lval_all, u0)
        if self.use_best_u and self.average.u < self.best_u:
            self.best_u = self.average.u
        
        # if self.use_best_u:
        #     self.average.penalty.gradient(self.lval_all, self.best_u, self.weights)
        # else:
        self.average.gradient(self.lval_all, self.weights)

        if self.normalize:
            normalize_memoryview(self.weights)
    #
    cpdef double get_qvalue(self):
        return self.average.u
    #
    cpdef double[::1] get_weights(self):
        return self.weights

