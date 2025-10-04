# coding: utf-8

# The MIT License (MIT)
#
# Copyright (c) <2015-2022> <Shibzukhov Zaur, szport at gmail dot com>
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

from mlgrad.funcs cimport CompSqrt

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
    def __init__(self, Risk risk, _normalize=1):
        # self.func = func
        self.risk = risk
        self._normalize = _normalize
        self.weights = np.zeros(len(risk.X), 'd')
    #
    cpdef eval_weights(self):
        self.risk._evaluate_losses_derivative_div_all(self.weights)
        if self._normalize:
            inventory._normalize(self.weights)
    #
    cpdef double get_qvalue(self):
        return self.risk._evaluate()
    #
    cpdef double[::1] get_weights(self):
        return self.weights

cdef class MWeights(Weights):
    #
    def __init__(self, Average average, Risk risk, _normalize=0):
        self.average = average
        self.risk = risk
        self.first_time = 1
        self._normalize = _normalize
        self.weights = np.empty(len(risk.X), 'd')
        self.lvals = np.empty(len(risk.X), 'd')
    #
    cpdef init(self):
        self.first_time = 1
    #
    cpdef eval_weights(self):
        cdef double u
        self.risk._evaluate_losses_all(self.lvals)
        u = self.average._evaluate(self.lvals)
        self.average._gradient(self.lvals, self.weights)
        if self._normalize:
            inventory._normalize(self.weights)
    #
    cpdef double get_qvalue(self):
        return self.average.u
    #
    cpdef double[::1] get_weights(self):
        return self.weights

cdef class MWeights2(Weights):
    #
    def __init__(self, Average average, ERisk2 risk, _normalize=0):
        self.average = average
        self.risk = risk
        self.first_time = 1
        self._normalize = _normalize
        self.weights = np.empty(len(risk.X), 'd')
        self.lvals = np.empty(len(risk.X), 'd')
    #
    cpdef init(self):
        self.first_time = 1
    #
    cpdef eval_weights(self):
        cdef double u
        self.risk._evaluate_losses_all(self.lvals)
        u = self.average._evaluate(self.lvals)
        self.average._gradient(self.lvals, self.weights)
        if self._normalize:
            inventory._normalize(self.weights)
    #
    cpdef double get_qvalue(self):
        return self.average.u
    #
    cpdef double[::1] get_weights(self):
        return self.weights

cdef class MWeights22(Weights):
    #
    def __init__(self, Average average, ERisk22 risk, _normalize=0):
        self.average = average
        self.risk = risk
        self.first_time = 1
        self._normalize = _normalize
        self.weights = np.empty(len(risk.X), 'd')
        self.lvals = np.empty(len(risk.X), 'd')
    #
    cpdef init(self):
        self.first_time = 1
    #
    cpdef eval_weights(self):
        cdef double u
        self.risk._evaluate_losses_all(self.lvals)
        u = self.average._evaluate(self.lvals)
        self.average._gradient(self.lvals, self.weights)
        if self._normalize:
            inventory._normalize(self.weights)
    #
    cpdef double get_qvalue(self):
        return self.average.u
    #
    cpdef double[::1] get_weights(self):
        return self.weights
    
cdef class WeightsCompose(Weights):
    
    def __init__(self, Weights weights1, Weights weights2, _normalize=1):
        self._weights1 = weights1
        self._weights2 = weights2
        self.weights = np.ones(len(weights1.risk.X), 'd')
        self._normalize = _normalize
        self.risk = self._weights1.risk
    #
    cpdef init(self):
        self._weights1.init()
        self._weights2.init()
    #   
    cpdef eval_weights(self):
        self._weights1.eval_weights()
        self._weights2.eval_weights()
        
        inventory.mul(self.weights, self._weights1.weights, self._weights2.weights)
        
        if self._normalize:
            inventory._normalize(self.weights)        
    #
    cpdef double[::1] get_weights(self):
        return self.weights
    #
    cpdef double get_qvalue(self):
        return self._weights1.get_qvalue()
    #

cdef class WRWeights(Weights):
    
    def __init__(self, Average average, Risk risk, _normalize=1):
        self.risk = risk
        self._weights1 = MWeights(average, risk, _normalize=0)
        self._weights2 = RWeights(risk, _normalize=0)
        self.weights = np.ones(len(risk.X), 'd')
        self._normalize = _normalize
    #
    cpdef init(self):
        self._weights1.init()
        self._weights2.init()
    #   
    cpdef eval_weights(self):
        self._weights1.eval_weights()
        self._weights2.eval_weights()
        
        inventory.mul(self.weights, self._weights1.weights, self._weights2.weights)
        
        if self._normalize:
            inventory._normalize(self.weights)        
    #
    cpdef double[::1] get_weights(self):
        return self.weights
    #
    cpdef double get_qvalue(self):
        return self._weights1.get_qvalue()
    #
