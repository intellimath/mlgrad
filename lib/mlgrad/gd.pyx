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

#from cython.parallel cimport parallel, prange

cimport mlgrad.inventory as inventory

# from mlgrad.models cimport Model
from mlgrad.funcs cimport Func
from mlgrad.funcs2 cimport Func2
from mlgrad.avragg cimport Average, ArithMean
from mlgrad.weights cimport Weights, ConstantWeights, ArrayWeights
from mlgrad.risks cimport Risk, Functional

from mlgrad.averager import ArrayAverager, ArraySave, _get_averager

import numpy as np

cdef double double_max = PyFloat_GetMax()
cdef double double_min = PyFloat_GetMin()

from math import isnan

cdef class GD:

    @property
    def sample_weights(self):
        return np.asarray(self.risk.weights)
    #
    def use_gradient_averager(self, averager):
        if type(averager) is str:
            averager = _get_averager(averager)
        if issubclass(averager, ArrayAverager):
            self.grad_averager = averager()
        else:
            raise TypeError(f"invalid averager: {averager}")
    #
    def add_projector(self, projector):
        if self.projector is None:
            self.projector = []
        self.projector.append(projector)
    #
    def init(self):
        self.risk.init()

        if self.projector is not None:
            for prj in self.projector:
                prj.project(self.risk.model)

        n_param = self.risk.model.n_param

#         if self.param_prev is None:
#             self.param_prev = np.zeros((n_param,), dtype='d')
        if self.param_min is None:
            self.param_min = self.risk.param.copy()
        else:
            self.param_min[:] = self.risk.param

        if self.param_copy is None:
            self.param_copy = self.risk.param.copy()
        else:
            self.param_copy[:] = self.risk.param

        # if self.stop_condition is None:
        #     self.stop_condition = DiffL1StopCondition(self)
        # self.stop_condition.init()

        if self.grad_averager is None:
            self.grad_averager = ArraySave()
        self.grad_averager._init(n_param)
        # print(self.grad_averager)

        # if self.param_transformer is not None:
        #     self.param_transformer.init(n_param)

    #
    def fit_step(self):
        cdef double lval
        cdef Projector prj

        self.fit_epoch()

        if self.projector is not None:
            for prj in self.projector:
                prj._project(self.risk.model)

        lval = self.risk._evaluate()
        self.lvals.append(lval)

        if self.callback is not None:
            self.callback(self)

        if self.stop_cond._is_minval(lval):
            inventory.move(self.param_min, self.risk.param)
    #
    def fit(self):
        cdef Risk risk = self.risk
        cdef Py_ssize_t i, k = 0, m=0, M=self.M
        cdef double lval #, lval_prev, lval_min, lval_min_prev
        cdef double tol = self.tol
        cdef double Q
        cdef inventory.StopCondition stop_cond
        cdef Projector prj

        self.risk.batch.init()
        self.init()

        lval = self.risk._evaluate()
        self.lvals = [lval]

        self.stop_cond = StopCondition(lval, self.tol)
        self.K = 0

        self.h_rate.init()

        if self.projector is not None:
            for prj in self.projector:
                prj._project(self.risk.model)

        self.completed = 0
        for k in range(self.n_iter):
            self.K = k
            if inventory._hasnan(&self.risk.param[0], self.risk.n_param):
                raise ValueError(f"param has NaN value at step {k+1}")

            self.fit_step()

            if self.stop_cond._stop_condition():
                self.completed = 1
                break

        self.K += 1
        self.finalize()
    #
    def gradient(self):
        cdef Risk risk = self.risk
        risk._gradient()
    #
    def fit_epoch(self):
        cdef Risk risk = self.risk
        cdef Py_ssize_t i, j, n_repeat = 1, m

        if risk.n_sample > 0 and risk.batch is not None and risk.batch.size > 0:
            n_repeat, m = divmod(risk.n_sample, risk.batch.size)
            if m > 0:
                n_repeat += 1

        for j in range(n_repeat):
            risk.batch.generate()

            self.h = self.h_rate.get_rate()

            self.gradient()

            if inventory.hasnan(risk.grad_average):
                raise ValueError(f"grad_average has NaN value at step {self.K+1} (j={j})")

            self.grad_averager._update(risk.grad_average, self.h)
            risk._update_param(self.grad_averager.array_average)

            # if self.param_transformer is not None:
            #     self.param_transformer.transform(risk.model.param)
    #
    # def use_transformer(self, transformer):
    #     self.param_transformer = transformer
#
    def finalize(self):
        cdef Risk risk = self.risk

        inventory.move(risk.param, self.param_min)

include "gd_fg.pyx"
include "gd_fg_rud.pyx"
#include "gd_rk4.pyx"
# include "gd_sgd.pyx"
#include "gd_sag.pyx"

# Fittable.register(GD)
# Fittable.register(FG)
# Fittable.register(FG_RUD)
# Fittable.register(SGD)

# include "stopcond.pyx"
include "paramrate.pyx"
include "projector.pyx"


