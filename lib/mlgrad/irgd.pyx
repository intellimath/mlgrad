# coding: utf-8

# cython: language_level=3
# cython: boundscheck=True
# cython: wraparound=True
# cython: nonecheck=True
# cython: embedsignature=True
# cython: initializedcheck=True
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

from sys import float_info

import numpy as np
from mlgrad.avragg import Average_FG
from mlgrad.gd import FG
from mlgrad.risk import Risk, Functional

cdef class IRGD(object):
    #
    def __init__(self, GD gd, Weights weights=None, tol=1.0e-5, n_iter=100, h_anneal=0.95, M=40, callback=None):
        """
        """
        self.gd = gd
        
        self.tol = tol
        self.n_iter = n_iter
        self.M = M
        self.m = 0

        self.param_best = np.zeros(len(self.gd.risk.param), dtype='d')
#         self.param_prev = np.zeros((m,), dtype='d')
        
        self.h_anneal = h_anneal
        
        self.callback = callback
        
        self.weights = weights
        
        self.lval = self.lval1 = self.lval2 = 0
        
        # self.is_warm_start = False
        self.completed = False
        
        self.K = 0
        self.m = 0
        
        self.lvals = []
        #self.qvals = []
        self.n_iters = []
        
        self.K = 0
        
    #
    @property
    def risk(self):
        return self.gd.risk
    #
    def fit(self):
        cdef Functional risk = self.gd.risk
                           
        self.weights.init()
        self.weights.eval_weights()
        risk.use_weights(self.weights.weights)

#         if not self.is_warm_start:
        self.lval_best = self.weights.get_qvalue()
        self.param_best[:] = risk.param
        
        if self.callback is not None:
            self.callback(self)
        
        K = 0
        while K < self.n_iter:
            
            self.gd.fit()
#             if not self.is_warm_start:
#                 self.is_warm_start = True
            
            self.n_iters.append(self.gd.K)

            if self.callback is not None:
                self.callback(self)

            self.weights.eval_weights()
            risk.use_weights(self.weights.weights)
            
            self.lval = self.weights.get_qvalue()
                
            if self.stop_condition():
                self.completed = 1

            if self.lval < self.lval_best:
                self.param_best[:] = risk.param
                self.lval_best = self.lval

            if K > 11 and self.lval > self.lvals[-1]:
                self.gd.h_rate.h *= self.h_anneal
                self.gd.h_rate.init()

            self.lvals.append(self.lval)

            if self.completed:
                break
            

            K += 1            
        #
        self.K += K
        self.finalize()

    #
    cdef finalize(self):
        cdef Functional risk = self.gd.risk

        risk.param[:] = self.param_best
    #
    cdef bint stop_condition(self):
#         cdef double lval_min
        
        if fabs(self.lval - self.lval_best) / (1 + fabs(self.lval_best)) < self.tol:
            return 1
        
#         if self.K < 3:
#             self.lval1, self.lval2 = self.lval, self.lval1
#             return 0        

#         lval_min = min3(self.lval, self.lval1, self.lval2)
#         if 0.5 * fabs(self.lval - 2*self.lval1 + self.lval2) / (1.0e-8 + fabs(lval_min)) < self.tol:
#             return 1

#         self.lval1, self.lval2 = self.lval, self.lval1

        return 0

