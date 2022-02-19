# coding: utf-8

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
class IRERM(object):
    #
    def __init__(self, erm, weights=None, tol=1.0e-5, n_iter=100, h_anneal=0.95, M=40, callback=None):
        """
        """
        self.erm = erm
        
        self.tol = tol
        self.n_iter = n_iter
        self.M = M
        self.m = 0

        self.param_best = np.zeros(len(erm.risk.param), dtype='d')
        
        self.h_anneal = h_anneal
        
        self.callback = callback
        
        self.weights = weights
        
        self.lval = self.lval1 = self.lval2 = 0
        
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
        return self.erm.risk
    #
    def fit(self):
        risk = self.erm.risk
        weights = self.weights
        erm = self.erm
                           
        weights.init()
        weights.eval_weights()
        risk.use_weights(weights.weights)

        self.lval_best = weights.get_qvalue()
        self.param_best[:] = risk.param
        
        if self.callback is not None:
            self.callback(self)
        
        for K in range(self.n_iter):
            erm.fit()
            
            self.n_iters.append(self.erm.K)

            if self.callback is not None:
                self.callback(self)

            weights.eval_weights()
            risk.use_weights(weights.weights)
            
            self.lval = self.weights.get_qvalue()
                
            if self.stop_condition():
                self.completed = 1

            if self.lval < self.lval_best:
                self.param_best[:] = risk.param
                self.lval_best = self.lval

            if K > 11 and self.lval > self.lvals[-1]:
                erm.h_rate.h *= self.h_anneal
                erm.h_rate.init()

            self.lvals.append(self.lval)

            if self.completed:
                break
        #
        self.K += K
        self.finalize()

    #
    def finalize(self):
        risk.param[:] = self.param_best
        self.lval = self.lval_best
    #
    def stop_condition(self):
        if abs(self.lval - self.lval_best) / (1 + abs(self.lval_best)) < self.tol:
            return 1
        
        return 0

