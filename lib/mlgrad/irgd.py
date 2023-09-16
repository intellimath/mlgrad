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

from sys import float_info

import numpy as np
# from mlgrad.avragg import Average_FG
# from mlgrad.gd import FG
# from mlgrad.risk import Risk, Functional

class IRGD:
    #
    def __init__(self, gd, weights, tol=1.0e-5, n_iter=100, h_anneal=0.99, M=11, callback=None):
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
        risk = self.gd.risk
                           
        self.weights.init()
        self.weights.eval_weights()
        risk.use_weights(self.weights.weights)

        self.lval_best = self.weights.get_qvalue()
        # print(self.lval_best)
        self.param_best[:] = risk.param
        
        if self.callback is not None:
            self.callback(self)

        m = 0
        M = self.M
        for K in range(self.n_iter):     
            # print(K)
            self.gd.fit()
            
            self.n_iters.append(self.gd.K)

            if self.callback is not None:
                self.callback(self)

            self.weights.eval_weights()
            risk.use_weights(self.weights.weights)
            
            self.lval = self.weights.get_qvalue()
            # print(self.lval)
                
            if self.stop_condition():
                self.completed = 1

            if self.lval < self.lval_best:
                self.param_best[:] = risk.param
                self.lval_best = self.lval
                m = 0
            else:
                m += 1

            if K > 5 and self.lval > self.lvals[-1]:
                self.gd.h_rate.h *= self.h_anneal
                self.gd.h_rate.init()

            if m > M:
                self.completed = 1

            self.lvals.append(self.lval)

            if self.completed:
                break
        #
        self.K += K
        self.finalize()

    #
    def finalize(self):
        risk = self.gd.risk

        print(self.lval_best)
        risk.param[:] = self.param_best
    #
    def stop_condition(self):
        
        if abs(self.lval - self.lval_best) / (1 + abs(self.lval_best)) < self.tol:
            return True
        
        return False

