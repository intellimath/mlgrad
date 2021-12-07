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
    cdef init(self):
        pass
    #
    cdef eval_weights(self):
        pass
    #
    cdef float[::1] get_weights(self):
        return None
    
    cdef float get_qvalue(self):
        return 0
    #
    cdef set_param(self, name, val):
        pass

cdef class ArrayWeights(Weights):
    #
    def __init__(self, weights):
        self.weights = np.asarray(weights)
    #
    cdef float[::1] get_weights(self):
        return self.weights

cdef class ConstantWeights(Weights):
    #
    def __init__(self, N):
        self.weights = np.full((N,), 1.0/N, 'f')
    #
    cdef eval_weights(self):
        pass
    #
    cdef float[::1] get_weights(self):
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
        self.lval_all = np.zeros(N, 'f')
        self.weights = np.zeros(N, 'f')
    #
    cdef eval_weights(self):
        cdef Risk risk = self.risk
        cdef Py_ssize_t N = risk.batch.size
        cdef Py_ssize_t j, k
        cdef Py_ssize_t[::1] indices = risk.batch.indices
        cdef float[::1] weights = self.weights
        cdef Model mod = self.risk.model
        cdef float[:,::1] X = self.risk.X
        cdef float[::1] Y = self.risk.Y
        cdef Func func = self.func
        cdef float val

        for j in range(N):
            k = indices[j]
            val = mod.evaluate(X[k])
            weights[j] = func.derivative_div_x(val - Y[k])
        
        if self.normalize:
            normalize_memoryview(weights)
    #
    cdef float get_qvalue(self):
        cdef float qval
        cdef Py_ssize_t j, k
        cdef Py_ssize_t N = self.risk.batch.size
        cdef Py_ssize_t[::1] indices = self.risk.batch.indices
        cdef Model mod = self.risk.model
        cdef float[:,::1] X = self.risk.X
        cdef float[::1] Y = self.risk.Y
        cdef Func func = self.func
        cdef float val
        
        qval = 0
        for j in range(N):
            k = indices[j]
            val = mod.evaluate(X[k])
            qval += func.evaluate(val - Y[k])
        qval /= N 
        return qval
    #
    cdef float[::1] get_weights(self):
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
        self.weights = np.zeros(N, 'f')
        self.lval_all = np.zeros(N, 'f')
    #
    cdef init(self):
        self.first_time = 1
    #
    cdef eval_weights(self):
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
        
        if self.use_best_u:
            self.average.penalty.gradient(self.lval_all, self.best_u, self.weights)
        else:
            self.average.gradient(self.lval_all, self.weights)

        if self.normalize:
            normalize_memoryview(self.weights)
    #
    cdef float get_qvalue(self):
        return self.average.u
    #
    cdef float[::1] get_weights(self):
        return self.weights

# cdef class SWMWeights(Weights):
#     #
#     def __init__(self, Average average, Functional risk, normalize=0, u_only=1):
#         self.average = average
#         self.risk = risk
#         self.weights = None
#         self.lval_all = None
#         self.first_time = 1
#         self.normalize = normalize
#         self.u_only = 1
#     #
#     cdef eval_weights(self):
#         cdef Risk risk = self.risk
#         cdef Py_ssize_t N = risk.batch.size
# #         cdef int[::1] indices = risk.batch.indices
#         cdef Py_ssize_t j, k
#         cdef float v, Nv, u
#         cdef Func func = self.average.penalty.func
#         cdef float[::1] weights
#         cdef float[::1] lval_all

#         if self.lval_all is None:
#             self.lval_all = np.empty((N,), 'f')

#         risk.eval_losses(self.lval_all)
        
#         if self.weights is None:
#             self.weights = np.empty((N,), 'f')

#         if self.first_time:
#             self.average.u = array_min(self.lval_all)
#             self.first_time = 0
        
#         self.average.fit(self.lval_all)
        
#         self.average.gradient(self.lval_all, self.weights)
        
#         u = self.average.u
        
#         weights = self.weights
#         lval_all = self.lval_all

#         Nv = 1. / N
#         for j in range(N):
#             v = Nv * (1 - func.derivative(lval_all[j] - u))
#             weights[j] = 0.5 * (v + weights[j])

#         if self.normalize:
#             normalize_memoryview(self.weights)
#     #
#     cdef float get_qvalue(self):
#         cdef int N = self.risk.batch.size
#         cdef int j, k
#         cdef float[::1] lval_all = self.lval_all

#         cdef Func func = self.average.penalty.func
#         cdef float u = self.average.u
#         cdef float qval = 0
        

#         for j in range(N):
#             lval = lval_all[j]
#             qval += 0.5*(lval + u - func.evaluate(lval - u))
#         qval /= N
#         return qval        
#     #
#     cdef float[::1] get_weights(self):
#         return self.weights


# cdef class WMWeights(Weights):
#     #
#     def __init__(self, Average average, Functional risk, normalize=0, u_only=1):
#         self.average = average
#         self.risk = risk
#         self.weights = None
#         self.lval_all = None
#         self.first_time = 1
#         self.u_only = 1
#     #
#     cdef eval_weights(self):
#         cdef Risk risk = self.risk
#         cdef Py_ssize_t N = risk.batch.size
#         cdef Py_ssize_t k, m
#         cdef float v, Nv, Nv2, u
#         cdef float[::1] weights
#         cdef float[::1] lval_all

#         if self.lval_all is None:
#             self.lval_all = np.empty((N,), 'f')

#         risk.eval_losses(self.lval_all)

#         if self.weights is None:
#             self.weights = np.empty((N,), 'f')

#         if self.first_time:
#             self.average.u = array_min(self.lval_all)
#             self.first_time = 0

#         self.average.fit(self.lval_all)

#         self.average.gradient(self.lval_all, self.weights)

#         u = self.average.u

#         weights = self.weights
#         lval_all = self.lval_all
        
#         m = 0
#         for k in range(N):
#             if lval_all[k] > u:
#                 m += 1
    
#         Nv = 1. / N
#         Nv2 = Nv * m
#         for k in range(N):
#             if lval_all[k] <= u:
#                 weights[k] = Nv + weights[k] * Nv2
#             else:
#                 weights[k] = weights[k] * Nv2
                
#         if self.normalize:
#             normalize_memoryview(self.weights)
#     #
#     cdef float get_qvalue(self):
#         cdef Py_ssize_t N = self.risk.batch.size
#         cdef Py_ssize_t k
#         cdef float u
#         cdef float qval
#         cdef float[::1] weights
#         cdef float[::1] lval_all = self.lval_all
        
#         u = self.average.u
#         qval = 0
#         for k in range(N):
#             lval = lval_all[k]
#             if lval <= u:
#                 qval += lval
#             else:
#                 qval += u
#         qval /= N
#         return qval        
#     #
#     cdef float[::1] get_weights(self):
#         return self.weights

