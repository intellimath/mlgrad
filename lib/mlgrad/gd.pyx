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
# Copyright (c) <2015-2019> <Shibzukhov Zaur, szport at gmail dot com>
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

#from cython.parallel cimport parallel, prange

from mlgrad.model cimport Model
from mlgrad.func cimport Func
from mlgrad.regular cimport FuncMulti
from mlgrad.avragg cimport Average, ArithMean
from mlgrad.averager cimport ArrayAdaM1
from mlgrad.weights cimport Weights, ConstantWeights, ArrayWeights
from mlgrad.risk cimport Risk, Functional

import numpy as np

# cdef class Fittable(object):

#     cpdef init(self):
#         pass

#     cpdef fit(self):
#         pass

from mlgrad.abc import Fittable

cdef class GD: 

    cpdef init(self):
        init_rand()
        self.risk.init()
#         if self.normalizer is not None:
#             self.normalizer.normalize(self.risk.param)
        
        n_param = len(self.risk.param)
        
#         if self.param_prev is None:
#             self.param_prev = np.zeros((n_param,), dtype='d')
        self.param_min = np.array(self.risk.param, dtype='d', copy=True)
#         print(self.param_min.base)
        
        if self.stop_condition is None:
            self.stop_condition = DiffL1StopCondition(self)
        self.stop_condition.init()    

        if self.grad_averager is None:
            self.grad_averager = ArraySave()
        self.grad_averager.init(n_param)
        
#         if self.param_averager is not None:
#             self.param_averager.init(n_param)
            
        self.h_rate.init()
            
        self.m = 0
        self.lval = self.lval_min = self.risk.evaluate()
        self.lvals = [self.lval]
    #
    def fit(self):
        cdef Risk risk = self.risk
#         cdef double[::1] param = risk.param
#         cdef double[::1] param_min = self.param_min
#         cdef int i, j, n = param.shape[0]

        self.init()

        self.K = 1
        self.completed = 0
        while self.K < self.n_iter:

            self.lval_prev = self.lval

            risk.batch.generate()
            self.fit_epoch()
#             if self.normalizer is not None:
#                 self.normalizer.normalize(param)

            self.lval = risk.lval = risk.evaluate()
            if self.lval < self.lval_min:
                self.lval_min = self.lval
                copy_memoryview(self.param_min, risk.param)
#                 print(self.param_min.base)

#             j = 1
#             while self.lval > self.lval_prev:
#                 if j == 10:
#                     break
#                 for i in range(n):
#                     risk.param[i] = 0.5 * (risk.param[i] + self.param_prev[i])
#                 self.lval = risk.lval = risk.evaluate()
#                 j += 1

            self.lvals.append(self.lval)

            if self.callback is not None:
                self.callback(self)

            if self.stop_condition.verify():
                self.completed = 1
                break

            self.K += 1

        self.finalize()
    #
    cpdef gradient(self):
        self.risk.gradient()
    #
    cpdef fit_epoch(self):
        cdef Functional risk = self.risk
        cdef Py_ssize_t i, n_param = len(risk.param)
        cdef double[::1] grad_average
        cdef double[::1] param = risk.param
        cdef double h

        h = self.h = self.h_rate.get_rate()
        
        self.gradient()
        
        self.grad_averager.update(risk.grad_average, h)
        grad_average = self.grad_averager.array_average
        for i in range(n_param):
            param[i] -= grad_average[i]
            
#         if self.param_averager is not None:
#             self.param_averager.update(risk.param)
#             copy_memoryview(risk.param, self.param_averager.array_average)
    #
    def use_gradient_averager(self, averager):
        self.grad_averager = averager
#
#     def use_param_averager(self, averager):
#         self.param_averager = averager
#
    cpdef finalize(self):
        cdef Functional risk = self.risk
        
        copy_memoryview(risk.param, self.param_min)
        
            
include "gd_fg.pyx"
include "gd_fg_rud.pyx"
#include "gd_rk4.pyx"
include "gd_sgd.pyx"
#include "gd_sag.pyx"

Fittable.register(GD)
Fittable.register(FG)
Fittable.register(FG_RUD)
Fittable.register(SGD)

include "stopcond.pyx"
include "paramrate.pyx"
