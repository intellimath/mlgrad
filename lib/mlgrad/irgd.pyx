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

from sys import float_info

import numpy as np
from mlgrad.avragg import Average_FG
from mlgrad.gd import FG
from mlgrad.risk import Risk, Functional
# from mlgrad.averager import ScalarAdaM1, ArrayAdaM1

# cdef void eval_losses(Model model, Loss loss, double[:,::1], double[::1] Y, double[::1] lval_all):
#     cdef int k, N = X.shape[0]
#     cdef double y
#     cdef double* Y_ptr = &Y[0]
#     cdef double* lval_all_ptr = &lval_all[0]
#
#     for k in range(N):
#         y = model.evaluate(X[k])
#         lval_all_ptr[k] = loss.evaluate(y, Y_ptr[k])


cdef class IRGD(object):
    #
    def __init__(self, GD gd, Weights weights=None, tol=1.0e-4, n_iter=100, h_anneal=1.0, M=40, callback=None):
        """
        """
        self.gd = gd
        
        self.tol = tol
        self.n_iter = n_iter
        self.M = M
        self.m = 0

        self.h_anneal = h_anneal
        
        self.callback = callback
        
        self.weights = weights
        
        self.lval = self.lval1 = self.lval2 = 0
        
        self.is_warm_start = 0
    #
    @property
    def risk(self):
        return self.gd.risk
    #
    def fit(self):
        cdef Functional risk = self.gd.risk
        cdef int m = len(risk.param)
    
        self.param_best = np.zeros((m,), dtype='d')
#         self.param_prev = np.zeros((m,), dtype='d')
                       
        self.K = 0
        self.m = 0
        
        self.lvals = []
        #self.qvals = []
        self.n_iters = []
        
        self.K = 1
        self.weights.eval_weights()
        risk.use_weights(self.weights.weights)

        self.lval_best = self.weights.get_qvalue()
        copy_memoryview(self.param_best, risk.param)
        
        if self.callback is not None:
            self.callback(self)
        
        while self.K <= self.n_iter:
            
            self.gd.fit()
            
            self.n_iters.append(self.gd.K)

            if self.callback is not None:
                self.callback(self)

            self.weights.eval_weights()
            risk.use_weights(self.weights.weights)
            
            self.lval = self.weights.get_qvalue()
            self.lvals.append(self.lval)

            if self.lval < self.lval_best:
                copy_memoryview(self.param_best, risk.param)
                self.lval_best = self.lval            
                
            if self.stop_condition():
                self.finalize()
                break

            self.K += 1
            
            self.gd.h_rate.h *= self.h_anneal
    #
    cdef finalize(self):
        cdef Functional risk = self.gd.risk

        copy_memoryview(risk.param, self.param_best)
    #
    cdef bint stop_condition(self):
        cdef double lval_min
        
        if self.K < 3:
            self.lval1, self.lval2 = self.lval, self.lval1
            return 0        

        lval_min = min3(self.lval, self.lval1, self.lval2)
        if 0.5 * fabs(self.lval - 2*self.lval1 + self.lval2) / (1.0e-8 + fabs(lval_min)) < self.tol:
            return 1

        self.lval1, self.lval2 = self.lval, self.lval1

        return 0

