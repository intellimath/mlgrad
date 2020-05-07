# coding: utf-8

# cython: language_level=3
# cython: boundscheck=True
# cython: wraparound=True
# cython: nonecheck=True
# cython: embedsignature=True
# cython: initializedcheck=True

# The MIT License (MIT)
#
# Copyright (c) <2015-2020> <Shibzukhov Zaur, szport at gmail dot com>
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

cdef class Weights(object):
    #
    @property
    def qval(self):
        return self.get_qvalue()
    #
    cdef eval_weights(self):
        pass
    #
    cdef double[::1] get_weights(self):
        return None
    
    cdef double get_qvalue(self):
        return 0
        
    cdef set_param(self, name, val):
        pass

cdef class ArrayWeights(Weights):
    #
    def __init__(self, weights):
        self.weights = np.asarray(weights)
    #
    cdef double[::1] get_weights(self):
        return self.weights

cdef class ConstantWeights(Weights):
    #
    def __init__(self, N):
        self.weights = np.full((N,), 1.0/N, 'd')
    #
    cdef eval_weights(self):
        pass
    #
    cdef double[::1] get_weights(self):
        return self.weights
    
cdef class RWeights(Weights):
    #
    def __init__(self, Func func, Functional risk, normalize=1):
        self.func = func
        self.risk = risk
        self.weights = None
        self.lval_all = None
        self.normalize = normalize
    #
    cdef eval_weights(self):
        cdef Risk risk = self.risk
        cdef int N = risk.batch.size
        cdef int j, k
        cdef int[::1] indices = risk.batch.indices
        cdef double[::1] weights
        cdef double[::1] lval_all

        if self.lval_all is None:
            self.lval_all = np.empty((N,), 'd')
        if self.weights is None:
            self.weights = np.empty((N,), 'd')

        risk.eval_losses(self.lval_all)
        
        weights = self.weights
        lval_all = self.lval_all

        for j in range(N):
            k = indices[j]
            weights[j] = self.func.derivative_div_x(lval_all[k])
        
        if self.normalize:
            normalize_memoryview(weights)
    #
    cdef double get_qvalue(self):
        cdef double qval
        cdef int j, k
        cdef int N = self.risk.batch.size
        cdef int[::1] indices = self.risk.batch.indices
        cdef double[::1] lval_all = self.lval_all
        
        self.risk.eval_losses(lval_all)
        
        qval = 0
        for j in range(N):
            k = indices[j]
            qval += self.func.evaluate(lval_all[k])
        qval /= N 
        return qval
    #
    cdef double[::1] get_weights(self):
        return self.weights

cdef class MWeights(Weights):
    #
    def __init__(self, Average average, Functional risk, normalize=0, u_only=1):
        self.average = average
        self.risk = risk
        self.weights = None
        self.lval_all = None
        self.normalize = normalize
        self.first_time = 1
        self.normalize = normalize
        self.u_only = u_only
    #
    cdef eval_weights(self):
        cdef Risk risk = <Risk>self.risk
        cdef int N = risk.batch.size

        if self.lval_all is None or self.lval_all.shape[0] != N:
            self.lval_all = np.empty(N, 'd')

        risk.eval_losses(self.lval_all)
        
        if self.weights is None or self.weights.shape[0] != N:
            self.weights = np.zeros((N,), 'd')

        if self.first_time:
            self.average.u = array_min(self.lval_all)
            self.first_time = 0
        
        self.average.fit(self.lval_all, self.average.u)
        
        self.average.gradient(self.lval_all, self.weights)

        if self.normalize:
            normalize_memoryview(self.weights)
    #
    cdef double get_qvalue(self):
        return self.average.u
    #
    cdef double[::1] get_weights(self):
        return self.weights

cdef class SWMWeights(Weights):
    #
    def __init__(self, Average average, Functional risk, normalize=0, u_only=1):
        self.average = average
        self.risk = risk
        self.weights = None
        self.lval_all = None
        self.first_time = 1
        self.normalize = normalize
        self.u_only = 1
    #
    cdef eval_weights(self):
        cdef Risk risk = self.risk
        cdef int N = risk.batch.size
#         cdef int[::1] indices = risk.batch.indices
        cdef int j, k
        cdef double v, Nv, u
        cdef Func func = self.average.penalty.func
        cdef double[::1] weights
        cdef double[::1] lval_all

        if self.lval_all is None:
            self.lval_all = np.empty((N,), 'd')

        risk.eval_losses(self.lval_all)
        
        if self.weights is None:
            self.weights = np.empty((N,), 'd')

        if self.first_time:
            self.average.u = array_min(self.lval_all)
            self.first_time = 0
        
        self.average.fit(self.lval_all)
        
        self.average.gradient(self.lval_all, self.weights)
        
        u = self.average.u
        
        weights = self.weights
        lval_all = self.lval_all

        Nv = 1. / N
        for j in range(N):
            v = Nv * (1 - func.derivative(lval_all[j] - u))
            weights[j] = 0.5 * (v + weights[j])

        if self.normalize:
            normalize_memoryview(self.weights)
    #
    cdef double get_qvalue(self):
        cdef int N = self.risk.batch.size
        cdef int j, k
        cdef double[::1] lval_all = self.lval_all

        cdef Func func = self.average.penalty.func
        cdef double u = self.average.u
        cdef double qval = 0
        

        for j in range(N):
            lval = lval_all[j]
            qval += 0.5*(lval + u - func.evaluate(lval - u))
        qval /= N
        return qval        
    #
    cdef double[::1] get_weights(self):
        return self.weights


cdef class WMWeights(Weights):
    #
    def __init__(self, Average average, Functional risk, normalize=0, u_only=1):
        self.average = average
        self.risk = risk
        self.weights = None
        self.lval_all = None
        self.first_time = 1
        self.u_only = 1
    #
    cdef eval_weights(self):
        cdef Risk risk = self.risk
        cdef int N = risk.batch.size
        cdef int k, m
        cdef double v, Nv, Nv2, u
        cdef double[::1] weights
        cdef double[::1] lval_all

        if self.lval_all is None:
            self.lval_all = np.empty((N,), 'd')

        risk.eval_losses(self.lval_all)

        if self.weights is None:
            self.weights = np.empty((N,), 'd')

        if self.first_time:
            self.average.u = array_min(self.lval_all)
            self.first_time = 0

        self.average.fit(self.lval_all)

        self.average.gradient(self.lval_all, self.weights)

        u = self.average.u

        weights = self.weights
        lval_all = self.lval_all
        
        m = 0
        for k in range(N):
            if lval_all[k] > u:
                m += 1
    
        Nv = 1. / N
        Nv2 = Nv * m
        for k in range(N):
            if lval_all[k] <= u:
                weights[k] = Nv #+ weights[k] * Nv2
            else:
                weights[k] = weights[k] * Nv2
                
        if self.normalize:
            normalize_memoryview(self.weights)
    #
    cdef double get_qvalue(self):
        cdef int N = self.risk.batch.size
        cdef int k
        cdef double u
        cdef double qval
        cdef double[::1] weights
        cdef double[::1] lval_all = self.lval_all
        
        u = self.average.u
        qval = 0
        for k in range(N):
            lval = lval_all[k]
            if lval <= u:
                qval += lval
            else:
                qval += u
        qval /= N
        return qval        
    #
    cdef double[::1] get_weights(self):
        return self.weights

