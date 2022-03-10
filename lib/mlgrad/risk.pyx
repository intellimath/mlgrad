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

from mlgrad.model cimport Model, MLModel
from mlgrad.func cimport Func, Square
from mlgrad.loss cimport Loss, ErrorLoss
from mlgrad.distance cimport Distance
from mlgrad.regnorm cimport FuncMulti, SquareNorm
from mlgrad.avragg cimport Average, ArithMean
from mlgrad.batch import make_batch, WholeBatch

import numpy as np

cdef object np_double = np_double
cdef object np_empty = np.empty
cdef object np_zeros = np.zeros
cdef object np_ones = np.ones

# from cython.parallel cimport parallel, prange

# from openmp cimport omp_get_num_procs, omp_get_thread_num

# cdef int num_procs = omp_get_num_procs()
# if num_procs > 4:
#     num_procs /= 2
# else:
#     num_procs = 2

cdef class Functional:
    #
    cpdef init(self):
        pass
    #
    def evaluate(self):
        return self._evaluate()
    #
    cdef double _evaluate(self):
        return 0
    #
    cdef void _gradient(self):
        pass
    #
    def gradient(self):
        self._gradient()
    #

cdef class SimpleFunctional(Functional):
    #
    def __init__(self, FuncMulti func, double[::1] param=None):
        self.regnorm = func
        if self.param is None:
            raise RuntimeError("Param is not specified")
        self.param = param
        self.n_param = len(self.param)
        self.grad_average = np.zeros(self.n_param, np_double)
        self.batch = None
        self.n_sample = 0
    #
    cdef double _evaluate(self):
        self.lval = self.regnorm._evaluate(self.param)
        return self.lval
    #
    cdef void _gradient(self):
        self.regnorm._gradient(self.param, self.grad_average)
        
cdef class Risk(Functional):
    #
    cdef void _evaluate_models(self):
        cdef Model _model = self.model

        cdef double[:, ::1] X = self.X
        cdef Py_ssize_t[::1] indices = self.batch.indices
        
        cdef Py_ssize_t j, k
        cdef double[::1] Yp = self.Yp

        for j in range(self.batch.size):
            k = indices[j]
            Yp[j] = _model._evaluate(X[k])
    #
    cdef void _evaluate_losses(self):
        cdef Loss _loss = self.loss

        cdef double[:, ::1] X = self.X
        cdef double[::1] Y = self.Y
        cdef double[::1] L = self.L
        cdef double[::1] Yp = self.Yp
        cdef Py_ssize_t[::1] indices = self.batch.indices
        
        cdef Py_ssize_t j, k

        for j in range(self.batch.size):
            k = indices[j]
            L[j] = _loss._evaluate(Yp[j], Y[k])
    #
    cdef void _evaluate_losses_derivative_div(self):
        cdef Py_ssize_t j, k
        cdef Loss _loss = self.loss

        cdef double[:, ::1] X = self.X
        cdef double[::1] Yp = self.Yp
        cdef double[::1] Y = self.Y
        cdef double[::1] LD = self.LD
        cdef Py_ssize_t[::1] indices = self.batch.indices

        for j in range(self.batch.size):
            k = indices[j]
            LD[j] = _loss._derivative_div(Yp[j], Y[k])
    #
    cdef void _evaluate_weights(self):
            pass
    #
    def evaluate_weights(self):
        # W = np_zeros(self.batch.size, np_double)
        # self._evaluate_weights()
        # return W
        pass
    #
    def evaluate_losses(self):
        # L = np_zeros(self.batch.size, np_double)
        self._evaluate_losses()
        return self.L.base
    #
    def evaluate_models(self):
        # Y = np_zeros(self.batch.size, np_double)
        self._evaluate_models()
        return self.Yp.base
    #
    def evaluate_losses_derivative_div(self):
        # DL = np_zeros(self.batch.size, np_double)
        self._evaluate_losses_derivative_div()
        return self.LD.base
    #
    def use_weights(self, weights):
        if weights is None:
            n = self.batch.size
            self.weights = np.full(n, 1./n, np.double)
        else:
            self.weights = weights
    #
    def use_batch(self, batch not None):
        self.batch = batch
    #
    
cdef class MRisk(Risk): 
    #
    def __init__(self, double[:,::1] X not None, double[::1] Y not None, Model model not None, 
                       Loss loss=None, Average avg=None,
                       FuncMulti regnorm=None, Batch batch=None, tau=1.0e-3):
        self.model = model
        self.param = model.param
        self.n_param = model.n_param
        self.n_input = model.n_input

        if self.model.grad is None:
            self.model.grad = np.zeros(self.n_param, np_double)

        if self.model.grad_x is None:
            self.model.grad_x = np.zeros(model.n_input, np_double)

        if loss is None:
            self.loss = ErrorLoss(Square())
        else:
            self.loss = loss

        if avg is None:
            self.avg = ArithMean()
        else:
            self.avg = avg

        self.regnorm = regnorm
        if regnorm is not None:
            self.grad_r = np.zeros(self.n_param, np_double)

        self.grad = np.zeros(self.n_param, np_double)
        self.grad_average = np.zeros(self.n_param, np_double)
        
        if X.shape[1] != self.n_input:
            raise ValueError('X.shape[1] != model.n_input')

        self.X = X
        self.Y = Y
        self.n_sample = len(Y)
        self.tau = tau

        if batch is None:
            self.batch = WholeBatch(self.n_sample)
        else:
            self.batch = batch

        size = self.batch.size 
        self.weights = np.full(size, 1./size, np_double)
        self.Yp = np.zeros(size, np_double)
        self.L = np.zeros(size, np_double)
        self.LD = np.zeros(size, np_double)
        self.lval = 0
        self.first = 1
    #
    def use_batch(self, batch not None):
        self.batch = batch
    #
    cpdef init(self):
        self.batch.init()
    #
    cdef double _evaluate(self):        
        self._evaluate_models()
        self._evaluate_losses()
        if self.first:
            self.avg.fit(self.L, None)
            self.first = 0
        else:
            self.avg.fit(self.L, self.avg.u)
        
        if self.regnorm is not None:
            v = self.tau * self.regnorm._evaluate(self.model.param)
            self.lval += v

        self.lval = self.avg.u

        return self.lval
    #
    cdef void _gradient(self):
        cdef Model _model = self.model
        cdef Loss _loss = self.loss

        cdef Py_ssize_t i, j, k
        cdef double y, lval_dy, lval, vv
        
        cdef double[::1] Xk
        cdef double[:, ::1] X = self.X
        cdef double[::1] Y = self.Y
        cdef double[::1] weights = self.weights
        cdef double[::1] grad = self.grad
        cdef double[::1] grad_average = self.grad_average

        cdef Py_ssize_t[::1] indices = self.batch.indices
        cdef double[::1] Yp = self.Yp

        # self._evaluate_models()
        # self._evaluate_losses()
        self.avg._gradient(self.L, weights)

        clear_memoryview(grad_average)

        for j in range(self.batch.size):
            k = indices[j]
            Xk = X[k]
            
            _model._gradient(Xk, grad)

            vv = _loss._derivative(Yp[j], Y[k]) * weights[j]
            for i in range(self.n_param):
                grad_average[i] += vv * grad[i]

        if self.regnorm is not None:
            self.regnorm._gradient(_model.param, self.grad_r)
            for i in range(self.n_param):
                grad_average[i] += self.tau * self.grad_r[i]

cdef class ED(Risk):
    #
    def __init__(self, double[:,::1] X, Distance distfunc):
        self.X = X
        self.distfunc = distfunc
        self.param = None
        self.weights = None
        self.regnorm = None
        self.grad = None
        self.grad_average = None
        self.weights = None
        self.n_sample = X.shape[0]
        self.n_param = X.shape[1]
        self.batch = WholeBatch(self.n_sample)
    #
    cpdef init(self):
        n_sample = self.n_sample    
        n_param = self.n_param

        if self.param is None:
            self.param = np.zeros(n_param, dtype=np_double)
        
        if self.grad is None:
            self.grad = np.zeros(n_param, dtype=np_double)

        if self.grad_average is None:
            self.grad_average = np.zeros(n_param, dtype=np_double)

        if self.weights is None:
            self.weights = np.full(n_sample, 1./n_sample, np_double)
            
        self.L = np.zeros(n_sample, 'd')
        
        self.lval = 0
    #    
    cdef double _evaluate(self):
        cdef int k, n_sample = self.n_sample, n_param = self.n_param
        cdef double S
        
        cdef double[:,::1] X = self.X
        cdef double[::1] param = self.param
        cdef double[::1] weights = self.weights

        S = 0
        for k in range(n_sample):
            S += weights[k] * self.distfunc.evaluate(X[k], param)

        self.lval = S
        return S
    #
    cdef void _gradient(self):
        cdef Py_ssize_t i, k
        cdef double S, wk

        cdef double[:,::1] X = self.X
        cdef double[::1] param = self.param
        cdef double[::1] weights = self.weights
        cdef double[::1] grad = self.grad
        cdef double[::1] grad_average = self.grad_average
        cdef double[::1] Xk

        fill_memoryview(self.grad_average, 0.0)
        for k in range(self.n_sample):
            Xk = X[k]
            wk = weights[k]

            self.distfunc._gradient(Xk, param, grad)
            for i in range(self.n_param):
                grad_average[i] -= wk * grad[i]                    
    #
    cdef void _evaluate_models(self):
        pass
    #
    cdef void _evaluate_losses(self):
        cdef int n_sample = self.n_sample
        cdef int k

        cdef double[:,::1] X = self.X
        cdef double[::1] param = self.param
        cdef double[::1] L = self.L

        for k in range(n_sample):
            L[k] = self.distfunc.evaluate(X[k], param)

cdef class ERisk(Risk):
    #
    def __init__(self, double[:,::1] X not None, double[::1] Y not None, Model model not None, 
                 Loss loss=None, FuncMulti regnorm=None, Batch batch=None, tau=0.001):

        self.model = model
        self.param = model.param
        
        self.n_param = model.n_param
        self.n_input = model.n_input
        if self.model.grad is None:
            self.model.grad = np.zeros(self.n_param, np_double)

        if self.model.grad_x is None:
            self.model.grad_x = np.zeros(self.n_input, np_double)

        if loss is None:
            self.loss = ErrorLoss(Square())
        else:
            self.loss = loss

        self.regnorm = regnorm
        if self.regnorm is not None:
            self.grad_r = np.zeros(self.n_param, np_double)

        self.grad = np.zeros(self.n_param, np_double)
        self.grad_average = np.zeros(self.n_param, np_double)

        self.X = X
        self.Y = Y
        self.n_sample = len(Y)
        self.tau = tau

        if batch is None:
            self.batch = WholeBatch(self.n_sample)
        else:
            self.batch = batch

        size = self.batch.size 
        self.weights = np.full(size, 1./size, np_double)
        self.Yp = np.zeros(size, np_double)
        self.L = np.zeros(size, np_double)
        self.LD = np.zeros(size, np_double)
        self.lval = 0
    #
    cpdef init(self):
        self.batch.init()
    #
    cdef double _evaluate(self):
        cdef Py_ssize_t j, k, N = self.n_sample
        cdef double S

        cdef Model _model = self.model
        cdef Loss _loss = self.loss

        cdef double[:, ::1] X = self.X
        cdef double[::1] Y = self.Y
        cdef double[::1] L = self.L
        cdef double[::1] weights = self.weights
        cdef Py_ssize_t[::1] indices = self.batch.indices
        
        self._evaluate_models()
        self._evaluate_losses()
        
        S = 0
        for j in range(self.batch.size):
            S += weights[j] * L[j]
                    
        if self.regnorm is not None:
            S += self.tau * self.regnorm._evaluate(_model.param)                

        self.lval = S
        return S
    #
    cdef void _gradient(self):
        cdef Model _model = self.model
        cdef Loss _loss = self.loss

        cdef Py_ssize_t i, j, k
        cdef double vv
        
        cdef double[:, ::1] X = self.X
        cdef double[::1] Y = self.Y
        cdef double[::1] weights = self.weights
        cdef double[::1] grad = self.grad
        cdef double[::1] grad_average = self.grad_average

        cdef Py_ssize_t[::1] indices = self.batch.indices
        cdef double[::1] Yp = self.Yp
                
#         if weights.shape[0] != size:
#             self.weights = np.full(size, 1./size, np_double)
#             weights = self.weights
        
        clear_memoryview(self.grad_average)

        for j in range(self.batch.size):
            k = indices[j]

            _model._gradient(X[k], grad)
            vv = weights[j] * _loss._derivative(Yp[j], Y[k])
            for i in range(self.n_param):
                self.grad_average[i] += vv * grad[i]

        if self.regnorm is not None:
            self.regnorm._gradient(_model.param, self.grad_r)
            for i in range(self.n_param):
                self.grad_average[i] += self.tau * self.grad_r[i]

cdef class ERiskGB(Risk):
    #
    def __init__(self, double[:,::1] X not None, double[::1] Y not None, Model model not None, 
                 Loss loss=None, FuncMulti regnorm=None, Batch batch=None, 
                 alpha=1.0, tau=0.001):

        self.model = model
        self.param = model.param
        
        self.n_param = model.n_param
        self.n_input = model.n_input
        if self.model.grad is None:
            self.model.grad = np_zeros(self.n_param, np_double)

        if self.model.grad_x is None:
            self.model.grad_x = np_zeros(self.n_input, np_double)

        if loss is None:
            self.loss = ErrorLoss(Square())
        else:
            self.loss = loss

        self.regnorm = regnorm
        if self.regnorm is not None:
            self.grad_r = np_zeros(self.n_param, np_double)

        self.grad = np.zeros(self.n_param, np_double)
        self.grad_average = np_zeros(self.n_param, np_double)

        self.X = X
        self.Y = Y
        self.n_sample = len(Y)
        self.tau = tau
        

        if batch is None:
            self.batch = WholeBatch(self.n_sample)
        else:
            self.batch = batch

        size = self.batch.size 
        self.weights = np.full(size, 1./size, np_double)
        self.Yp = np.zeros(size, np_double)
        self.L = np.zeros(size, np_double)
        self.LD = np.zeros(size, np_double)
        self.lval = 0
        
        self.H = np_zeros(self.n_sample, np_double)
        self.alpha = alpha
    #
    def use_weights(self, weights not None):
        self.weights = weights
    #
    def use_batch(self, batch not None):
        self.batch = batch
    #
    cpdef init(self):
        self.batch.init()
    #
    cdef void _evaluate_models(self):
        cdef Py_ssize_t j, k
        cdef double y
        cdef Model _model = self.model
        cdef Loss _loss = self.loss

        cdef double[:, ::1] X = self.X
        cdef double[::1] Y = self.Y
        cdef Py_ssize_t[::1] indices = self.batch.indices
        cdef double alpha = self.alpha
        cdef double[::1] H = self.H
        cdef double[::1] Yp = self.Yp
        
        for j in range(self.batch.size):
            k = indices[j]
            Yp[j] = H[k] + alpha * _model._evaluate(X[k])
    #
    cdef void _evaluate_losses(self):
        cdef Py_ssize_t j, k
        cdef double y
        cdef Loss _loss = self.loss

        cdef double[:, ::1] X = self.X
        cdef double[::1] Y = self.Y
        cdef Py_ssize_t[::1] indices = self.batch.indices
        cdef double[::1] Yp = self.Yp
        cdef double[::1] L = self.L
        
        for j in range(self.batch.size):
            k = indices[j]
            L[j] = _loss._evaluate(Yp[j], Y[k])
    #
    cdef void _evaluate_losses_derivative_div(self):
        cdef Py_ssize_t j, k
        cdef double y
        cdef Loss _loss = self.loss

        cdef double[:, ::1] X = self.X
        cdef double[::1] Y = self.Y
        cdef Py_ssize_t[::1] indices = self.batch.indices

        cdef double[::1] Yp = self.Yp
        cdef double[::1] LD = self.LD
        
        for j in range(self.batch.size):
            k = indices[j]
            LD[j] = _loss._derivative_div(Yp[j], Y[k])
    #
    cdef double _evaluate(self):
        cdef Py_ssize_t j, k
        cdef double S, y

        cdef Model _model = self.model

        cdef double[:, ::1] X = self.X
        cdef double[::1] Y = self.Y
        cdef double[::1] weights = self.weights

        cdef double[::1] L = self.L
        
        self._evaluate_models()
        self._evaluate_losses()
        
        S = 0
        for j in range(self.batch.size):
            S += weights[j] * L[j] 
                    
        if self.regnorm is not None:
            S += self.tau * self.regnorm._evaluate(_model.param)                

        self.lval = S
        return S
    #
    cdef void _gradient(self):
        cdef Model _model = self.model
        cdef Loss _loss = self.loss

        cdef Py_ssize_t i, j, k
        cdef double y, vv
        
        cdef double[:, ::1] X = self.X
        cdef double[::1] Y = self.Y
        cdef double[::1] weights = self.weights
        cdef double[::1] grad = self.grad
        cdef double[::1] grad_average = self.grad_average

        cdef Py_ssize_t[::1] indices = self.batch.indices

        cdef double alpha = self.alpha
        cdef double[::1] Yp = self.Yp
        cdef double[::1] L = self.L
        
        clear_memoryview(self.grad_average)

        for j in range(self.batch.size):
            k = indices[j]

            # y = H[j] + alpha * _model._evaluate(X[k])
            _model._gradient(X[k], grad)

            vv = alpha * _loss._derivative(Yp[j], Y[k]) * weights[j]
            for i in range(self.n_param):
                self.grad_average[i] += vv * grad[i]

        if self.regnorm is not None:
            self.regnorm._gradient(_model.param, self.grad_r)
            for i in range(self.n_param):
                self.grad_average[i] += self.tau * self.grad_r[i]
    #
    cdef double derivative_alpha(self):
        # cdef Model _model = self.model
        cdef Loss _loss = self.loss

        cdef Py_ssize_t j, k, N = self.n_sample
        cdef double y, v
        
        cdef double[:, ::1] X = self.X
        cdef double[::1] Y = self.Y
        cdef double[::1] weights = self.weights

        cdef Py_ssize_t size = self.batch.size 
        cdef Py_ssize_t[::1] indices = self.batch.indices
        cdef double alpha = self.alpha
        cdef double[::1] H = self.H
        cdef double ret = 0

        cdef double[::1] Yp = self.Yp
        cdef double[::1] L = self.L
        
        for j in range(size):
            k = indices[j]

            # v = _model._evaluate(X[k])
            # y = H[k] + alpha * v
            ret += _loss._derivative(Yp[j], Y[k]) * weights[j] * Yp[j]
            
        return ret
                
# cdef class AER(ERisk):
#     #
#     def __init__(self, double[:,::1] X, double[::1] Y, 
#                  Model model, Loss loss, Average loss_averager=None,
#                  FuncMulti regnorm=None, Batch batch=None, tau=1.0e-3):
#         self.X = X
#         self.Y = Y
#         self.n_sample = len(Y)
#         self.model = model
#         self.param = model.param
#         self.loss = loss
#         self.loss_averager = loss_averager
#         self.regnorm = regnorm
#         self.weights = None
#         self.grad = None
#         self.grad_r = None
#         self.grad_average = None
#         self.lval_all = None
#         self.mval_all = None
#         self.tau = tau
#         if batch is None:
#             self.batch = WholeBatch(self.n_sample)
#         else:
#             self.batch = batch
#         if loss_averager is None:
#             self.loss_averager = ArithMean()
#         else:
#             self.loss_averager = loss_averager
#     #
#     cpdef init(self):
#         ERisk.init(self)
#         size = self.batch.size
#         if self.lval_all is None:
#             self.lval_all = np.zeros(size, np_double)
#         if self.mval_all is None:
#             self.mval_all = np.zeros(size, np_double)
#     #
#     cdef eval_all(self):
#         cdef int j, k
#         cdef double y
#         cdef Model _model = self.model
#         cdef Loss _loss = self.loss

#         cdef double[:, ::1] X = self.X
#         cdef double[::1] Y = self.Y

#         cdef Py_ssize_t size = self.batch.size 
#         cdef Py_ssize_t[::1] indices = self.batch.indices
        
#         cdef double[::1] mval_all = self.mval_all
#         cdef double[::1] lval_all = self.lval_all

#         for j in range(size):
#             k = indices[j]
#             mval_all[j] = y = _model._evaluate(X[k])
#             lval_all[j] = _loss._evaluate(y, Y[k])
#     #
#     cdef double _evaluate(self):
#         cdef Model _model = self.model
        
#         self.eval_all()
#         self.loss_averager.fit(self.lval_all)
#         self.lval = self.loss_averager.u
                    
#         if self.regnorm is not None:
#             self.lval += self.tau * self.regnorm._evaluate(_model.param)                

#         return self.lval
    
#     cdef void _gradient(self):
#         cdef Py_ssize_t i, j, k, n_param = self.model.n_param
#         cdef double lval_dy, wk

#         cdef Model _model = self.model
#         cdef Loss _loss = self.loss
        
#         cdef double[:, ::1] X = self.X
#         cdef double[::1] Y = self.Y

#         cdef double[::1] weights = self.weights
#         cdef double[::1] grad = self.grad
#         cdef double[::1] grad_average = self.grad_average

#         cdef Py_ssize_t size = self.batch.size 
#         cdef Py_ssize_t[::1] indices = self.batch.indices

#         cdef double[::1] mval_all = self.mval_all
#         cdef double[::1] lval_all = self.lval_all

#         self.eval_all()
#         self.loss_averager.fit(lval_all)
#         self.loss_averager._gradient(lval_all, weights)
        
#         fill_memoryview(grad_average, 0.)
#         for j in range(size):
#             k = indices[j]
#             lval_dy = weights[j] * _loss._derivative(mval_all[j], Y[k])
            
#             _model._gradient(X[k], grad)
            
#             for i in range(n_param):
#                 grad_average[i] += lval_dy * grad[i]
                
#         if self.regnorm is not None:
#             self.regnorm._gradient(self.model.param, self.grad_r)
#             for i in range(n_param):
#                 self.grad_average[i] += self.tau * self.grad_r[i]    
    
cdef class ER2(Risk):
    #
    def __init__(self, double[:,::1] X, double[:,::1] Y, MLModel model, MultLoss loss,
                       FuncMulti regnorm=None, Batch batch=None, tau=1.0e-3):
        self.model = model
        self.param = model.param
        self.loss = loss
        self.regnorm = regnorm
        self.weights = None
        self.grad = None
        self.grad_u = None
        self.grad_r = None
        self.grad_average = None
        self.X = X
        self.Y = Y
        self.n_sample = len(Y)
        if batch is None:
            self.batch = WholeBatch(self.n_sample)
        else:
            self.batch = batch
            
        self.L = np.zeros(self.batch.size, 'd')
    #
    def use_weights(self, weights):
        self.weights = weights
    #
    #cdef object get_loss(self):
    #    return self.loss
    #
    cpdef init(self):
        N = self.n_sample    
        self.n_param = self.model.n_param
        self.n_input = self.model.n_input
        self.n_output = self.model.n_output

        # if self.model.grad is None:
        #     self.model.grad = np.zeros((n_param,), np_double)
            
        if self.grad is None:
            self.grad = np.zeros(self.n_param, dtype=np_double)

        if self.grad_u is None:
            self.grad_u = np.zeros(self.n_output, dtype=np_double)

        if self.grad_average is None:
            self.grad_average = np.zeros(self.n_param, dtype=np_double)

        if self.regnorm:
            if self.grad_r is None:
                self.grad_r = np.zeros(self.n_param, dtype=np_double)
                
        if self.weights is None:
            self.weights = np.full((N,), 1./N, np_double)
        
        self.lval = 0
    #
    cdef void _evaluate_losses(self):
        cdef Py_ssize_t j, k, N = self.n_sample
        cdef MLModel _model = self.model
        cdef MultLoss _loss = self.loss
        #cdef double v
        cdef double[:, ::1] X = self.X
        cdef double[:, ::1] Y = self.Y
        cdef double[::1] output = _model.output

        cdef Py_ssize_t size = self.batch.size 
        cdef Py_ssize_t[::1] indices = self.batch.indices
        cdef double[::1] L = self.L

        for j in range(size):
            k = indices[j]
            _model.forward(X[k])
            L[k] = _loss._evaluate(output, Y[k])

        #if self.regnorm is not None:
        #    v = self.tau * self.regnorm._evaluate(self.model.param) / N
        #    for k in range(N):
        #        lval_all[k] += v
    #
    cdef double _evaluate(self):
        cdef Py_ssize_t j, k, N = self.n_sample
        cdef double y, lval, S

        cdef MLModel _model = self.model
        cdef MultLoss _loss = self.loss

        cdef double[:, ::1] X = self.X
        cdef double[:, ::1] Y = self.Y
        cdef double[::1] output = _model.output
        cdef double[::1] weights = self.weights

        cdef Py_ssize_t size = self.batch.size 
        cdef Py_ssize_t[::1] indices = self.batch.indices

        S = 0
        for j in range(size):
            k = indices[j]
            _model.forward(X[k])
            lval = _loss._evaluate(output, Y[k])
            S += weights[k] * lval
                    
        if self.regnorm is not None:
            S += self.tau * self.regnorm._evaluate(self.model.param)

        self.lval = S
        return S
    #
    cdef void _gradient(self):
        cdef Py_ssize_t j, k, n_param = self.model.n_param, N = self.n_sample
        cdef double y, yk, wk, S

        cdef MLModel _model = self.model
        cdef MultLoss _loss = self.loss
        cdef double[:, ::1] X = self.X
        cdef double[:, ::1] Y = self.Y
        cdef double[::1] output = _model.output
        cdef double[::1] weights = self.weights      
        cdef double[::1] Xk, Yk
        cdef double[::1] grad = self.grad
        cdef double[::1] grad_u = self.grad_u
        cdef double[::1] grad_average = self.grad_average

        cdef Py_ssize_t size = self.batch.size 
        cdef Py_ssize_t[::1] indices = self.batch.indices

        fill_memoryview(grad_average, 0)
                
        #S = 0
        for j in range(size):
            k = indices[j]
            Xk = X[k]
            Yk = Y[k]

            _model.forward(Xk)
            _loss._gradient(output, Yk, grad_u)
            _model.backward(Xk, grad_u, grad)
            
            wk = weights[k]
            
            for i in range(n_param):
                grad_average[i] += wk * grad[i]
                
        if self.regnorm is not None:
            self.regnorm._gradient(self.model.param, self.grad_r)
            for i in range(n_param):
                grad_average[i] += self.tau * self.grad_r[i]

# cdef class ER21(Risk):
#     #
#     def __init__(self, double[:,::1] X, double[::1] Y, MLModel model, MultLoss2 loss,
#                        FuncMulti regnorm=None, Batch batch=None, tau=1.0e-3):
#         self.model = model
#         self.param = model.param
#         self.loss = loss
#         self.regnorm = regnorm
#         self.weights = None
#         self.grad = None
#         self.grad_u = None
#         self.grad_r = None
#         self.grad_average = None
#         self.X = X
#         self.Y = Y
#         self.n_sample = len(Y)
#         if batch is None:
#             self.batch = WholeBatch(self.n_sample)
#         else:
#             self.batch = batch
            
#         self.init()
#     #
#     def use_weights(self, weights):
#         self.weights = weights
#     #
#     #cdef object get_loss(self):
#     #    return self.loss
#     # 
#     cpdef init(self):
#         N = self.n_sample    
#         self.n_param = self.model.n_param
#         self.n_input = self.model.n_input
#         self.n_output = self.model.n_output 

#         # if self.model.grad is None:
#         #     self.model.grad = np.zeros((n_param,), np_double)
            
#         if self.grad is None:
#             self.grad = np.zeros(self.n_param, dtype=np_double)

#         if self.grad_u is None:
#             self.grad_u = np.zeros(self.n_output, dtype=np_double)

#         if self.grad_average is None:
#             self.grad_average = np.zeros(self.n_param, dtype=np_double)

#         if self.regnorm:
#             if self.grad_r is None:
#                 self.grad_r = np.zeros(self.n_param, dtype=np_double)
                
#         if self.weights is None:
#             self.weights = np.full((N,), 1./N, np_double)
        
#         self.lval = 0
#     #
#     cdef void _evaluate_losses(self, double[::1] lval_all):
#         cdef Py_ssize_t j, k, N = self.n_sample
#         cdef MLModel _model = self.model
#         cdef MultLoss2 _loss = self.loss
#         #cdef double v
#         cdef double[:, ::1] X = self.X
#         cdef double[::1] Y = self.Y
#         cdef double[::1] output = _model.output

#         cdef Py_ssize_t size = self.batch.size 
#         cdef Py_ssize_t[::1] indices = self.batch.indices

#         for j in range(size):
#             k = indices[j]
#             _model.forward(X[k])
#             lval_all[k] = _loss._evaluate(output, Y[k])

#         #if self.regnorm is not None:
#         #    v = self.tau * self.regnorm._evaluate(self.model.param) / N
#         #    for k in range(N):
#         #        lval_all[k] += v
#     #
#     cdef double _evaluate(self):
#         cdef Py_ssize_t j, k, N = self.n_sample
#         cdef double y, lval, S

#         cdef MLModel _model = self.model
#         cdef MultLoss2 _loss = self.loss

#         cdef double[:, ::1] X = self.X
#         cdef double[::1] Y = self.Y
#         cdef double[::1] output = _model.output
#         cdef double[::1] weights = self.weights

#         cdef Py_ssize_t size = self.batch.size 
#         cdef Py_ssize_t[::1] indices = self.batch.indices

#         S = 0
#         for j in range(size):
#             k = indices[j]
#             _model.forward(X[k])
#             lval = _loss._evaluate(output, Y[k])
#             S += weights[k] * lval
                    
#         if self.regnorm is not None:
#             S += self.tau * self.regnorm._evaluate(self.model.param)

#         self.lval = S
#         return S
#     #
#     cdef void _gradient(self):
#         cdef Py_ssize_t j, k, n_param = self.model.n_param, N = self.n_sample
#         cdef double y, yk, wk, S

#         cdef MLModel _model = self.model
#         cdef MultLoss2 _loss = self.loss
#         cdef double[:, ::1] X = self.X
#         cdef double[::1] Y = self.Y
#         cdef double[::1] output = _model.output
#         cdef double[::1] weights = self.weights      
#         cdef double[::1] Xk 
#         cdef double[::1] grad = self.grad
#         cdef double[::1] grad_u = self.grad_u
#         cdef double[::1] grad_average = self.grad_average

#         cdef Py_ssize_t size = self.batch.size 
#         cdef Py_ssize_t[::1] indices = self.batch.indices
        
#         fill_memoryview(grad_average, 0)
                
#         #S = 0
#         for j in range(size):
#             k = indices[j]
#             Xk = X[k]
#             yk = Y[k]

#             _model.forward(Xk)
#             _loss._gradient(output, yk, grad_u)
#             _model.backward(Xk, grad_u, grad)
            
#             wk = weights[k]
            
#             for i in range(n_param):
#                 grad_average[i] += wk * grad[i]
                
#         if self.regnorm is not None:
#             self.regnorm._gradient(self.model.param, self.grad_r)
#             for i in range(n_param):
#                 grad_average[i] += self.tau * self.grad_r[i]
    