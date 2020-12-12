# coding: utf-8

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: embedsignature=True
# cython: initializedcheck=False

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

from mlgrad.model cimport Model, MLModel
from mlgrad.func cimport Func
from mlgrad.distance cimport Distance
from mlgrad.regular cimport FuncMulti
from mlgrad.avragg cimport Average, ArithMean
from mlgrad.averager cimport ArrayAdaM1
from mlgrad.weights cimport Weights, ConstantWeights, ArrayWeights
from mlgrad.batch import make_batch

import numpy as np

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
    def __call__(self):
        return self.evaluate()
    #
    cdef double evaluate(self):
        return 0
    #
    cdef void gradient(self):
        pass
    #

#     def __call__(self, param):
#         return self.func.evaluate(as_array_1d(param))
    
cdef class SimpleFunctional(Functional):
    #
    def __init__(self, FuncMulti func, double[::1] param=None):
        self.regular = func
        self.param = param
        self.grad_average = None
        self.batch = None
        self.n_sample = 0
    #
    cpdef init(self):
        if self.param is None:
            raise RuntimeError("Param is not specified")
        self.grad_average = np.zeros(self.param.shape[0], 'd')
    #
    cdef double evaluate(self):
        self.lval = self.regular.evaluate(self.param)
        return self.lval
    #
    cdef void gradient(self):
        self.regular.gradient(self.param, self.grad_average)
        
cdef class Risk(Functional):
    #
    cdef double eval_loss(self, int k):
        return 0
    #
    cdef void gradient_loss(self, int k):
        pass
    #
    cdef void eval_losses(self, double[::1] lval_all):
        pass
    #
#     cdef double evaluate(self):
#         return 0
#     #
#     cdef void gradient(self):
#         pass

cdef class ED(Risk):
    #
    def __init__(self, double[:,::1] X, Distance distfunc):
        self.X = X
        self.distfunc = distfunc
        self.param = None
        self.weights = None
        self.regular = None
        self.grad = None
        self.grad_average = None
        self.weights = None
        self.n_sample = X.shape[0]
        self.n_param = X.shape[1]
        self.batch = WholeBatch(self.n_sample)        
    #
    def use_weights(self, weights):
        self.weights = weights
    #
    cpdef init(self):
        n_sample = self.n_sample    
        n_param = self.n_param

        if self.param is None:
            self.param = np.zeros(n_param, dtype='d')
        
        if self.grad is None:
            self.grad = np.zeros(n_param, dtype='d')

        if self.grad_average is None:
            self.grad_average = np.zeros(n_param, dtype='d')

        if self.weights is None:
            self.weights = np.full(n_sample, 1./n_sample, 'd')
        
        self.lval = 0
    #    
    cdef double evaluate(self):
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
    cdef void gradient(self):
        cdef int n_sample = self.n_sample, n_param = self.n_param
        cdef int i, k
        cdef double S, wk

        cdef double[:,::1] X = self.X
        cdef double[::1] param = self.param
        cdef double[::1] weights = self.weights
        cdef double[::1] grad = self.grad
        cdef double[::1] grad_average = self.grad_average
        cdef double[::1] Xk

        fill_memoryview(self.grad_average, 0.0)
        for k in range(n_sample):
            Xk = X[k]
            wk = weights[k]

            self.distfunc.gradient(Xk, param, grad)
            for i in range(n_param):
                grad_average[i] -= wk * grad[i]                    
    #
    cdef void eval_losses(self, double[::1] lval_all):
        cdef int n_sample = self.n_sample
        cdef int k

        cdef double[:,::1] X = self.X
        cdef double[::1] param = self.param

        for k in range(n_sample):
            lval_all[k] = self.distfunc.evaluate(X[k], param)

cdef class ER(Risk):
    #
    def __init__(self, double[:,::1] X, double[::1] Y, Model model, Loss loss,
                       FuncMulti regular=None, Batch batch=None, Average avg=None, tau=1.0e-3):
        self.model = model
        self.param = model.param
        self.loss = loss
        self.regular = regular
        self.weights = None
        self.grad = None
        self.grad_r = None
        self.grad_average = None
        self.X = X
        self.Y = Y
        self.Yp = None
        self.n_sample = len(Y)
        self.tau = tau
        if batch is None:
            self.batch = WholeBatch(self.n_sample)
        else:
            self.batch = batch
        self.avg = avg
    #
    def use_weights(self, weights not None):
        self.weights = weights
    #
    def use_batch(self, batch not None):
        self.batch = batch
    #
    cpdef init(self):
        N = self.n_sample    
        n_param = self.model.n_param
        n_input = self.model.n_input

        if self.model.grad is None:
            self.model.grad = np.zeros(n_param, 'd')

        if self.model.grad_x is None:
            self.model.grad_x = np.zeros(n_input, 'd')

        if self.grad is None:
            self.grad = np.zeros(n_param, dtype='d')

        if self.grad_average is None:
            self.grad_average = np.zeros(n_param, dtype='d')

        if self.regular is not None:
            if self.grad_r is None:
                self.grad_r = np.zeros(n_param, dtype='d')

        size = self.batch.size 
        if self.weights is None:
            self.weights = np.full(size, 1./size, 'd')
            
        if self.Yp is None:
            self.Yp = np.zeros(size, 'd')

        self.lval = 0
        
#         if self.avg is not None:
#             self.avg_grad = np.zeros(zise, 'd')
    #
    cdef void eval_losses(self, double[::1] lval_all):
        cdef Py_ssize_t j, k, N = self.n_sample
        cdef double y
        cdef Model _model = self.model
        cdef Loss _loss = self.loss

        cdef double[:, ::1] X = self.X
        cdef double[::1] Y = self.Y
        cdef Py_ssize_t size = self.batch.size 
        cdef Py_ssize_t[::1] indices = self.batch.indices
        
        cdef Py_ssize_t k1, k2, k3, k4
        cdef double y1, y2, y3, y4
        
        j = 0
        while size >= 4:
            k1 = indices[j]
            k2 = indices[j+1]
            k3 = indices[j+2]
            k4 = indices[j+3]

            y1 = _model.evaluate(X[k1])
            y2 = _model.evaluate(X[k2])
            y3 = _model.evaluate(X[k3])
            y4 = _model.evaluate(X[k4])

            lval_all[j] = _loss.evaluate(y1, Y[k1])
            lval_all[j+1] = _loss.evaluate(y2, Y[k2])
            lval_all[j+2] = _loss.evaluate(y3, Y[k3])
            lval_all[j+3] = _loss.evaluate(y4, Y[k4])
            
            j += 4
            size -= 4
            
        while size > 0:
            k = indices[j]
            y = _model.evaluate(X[k])
            lval_all[j] = _loss.evaluate(y, Y[k])
            j += 1
            size -= 1
    #
    cdef double eval_loss(self, int k):
        cdef double y
        
        y = self.model.evaluate(self.X[k])
        return self.loss.evaluate(y, self.Y[k])
    #
    cdef double evaluate(self):
        cdef Py_ssize_t j, k, N = self.n_sample
        cdef double S, yk

        cdef Model _model = self.model
        cdef Loss _loss = self.loss

        cdef double[:, ::1] X = self.X
        cdef double[::1] Y = self.Y
        cdef double[::1] weights = self.weights
        cdef Py_ssize_t size = self.batch.size 
        cdef Py_ssize_t[::1] indices = self.batch.indices
        cdef double[::1] Yp = self.Yp

        cdef Py_ssize_t k1, k2, k3, k4
        cdef double y1, y2, y3, y4
        cdef double s1, s2, s3, s4
        
        S = 0
        j = 0
        while size >= 4:
            k1 = indices[j]
            k2 = indices[j+1]
            k3 = indices[j+2]
            k4 = indices[j+3]

            y1 = _model.evaluate(X[k1])
            y2 = _model.evaluate(X[k2])
            y3 = _model.evaluate(X[k3])
            y4 = _model.evaluate(X[k4])

            s1 = weights[j] * _loss.evaluate(y1, Y[k1])
            s2 = weights[j+1] * _loss.evaluate(y2, Y[k2])
            s3 = weights[j+2] * _loss.evaluate(y3, Y[k3])
            s4 = weights[j+3] * _loss.evaluate(y4, Y[k4])
            
            S += s1 + s2 + s3 + s4

            j += 4
            size -= 4
        
        
        while size > 0:
            k = indices[j]
            yk = _model.evaluate(X[k])

            S += weights[j] * _loss.evaluate(yk, Y[k])
            
            j += 1
            size -= 1
                    
        if self.regular is not None:
            S += self.tau * self.regular.evaluate(_model.param)                

        self.lval = S
        return S
    #
    cdef void gradient_loss(self, int k):
        cdef Model _model = self.model
        cdef Loss _loss = self.loss
        cdef double[::1] Xk, Yk
        cdef double y, yk, lval, lval_dy
        cdef Py_ssize_t n_param = self.model.n_param
        cdef double wk = self.weights[k]
        
        Xk = self.X[k]
        yk = self.Y[k]

        y = _model.evaluate(Xk)
        lval_dy = _loss.derivative(y, yk)
        
        lval_dy *= wk

        _model.gradient(Xk, self.grad)
        
        for i in range(n_param):
            self.grad[i] *= lval_dy            
    #
    cdef void gradient(self):
        cdef Model _model = self.model
        cdef Loss _loss = self.loss

        cdef Py_ssize_t i, j, k, n_param = self.model.n_param, N = self.n_sample
        cdef double y, yk, lval_dy, wk, lval, vv
        
        cdef double[:, ::1] X = self.X
        cdef double[::1] Y = self.Y
        cdef double[::1] weights = self.weights
        cdef double[::1] grad = self.grad
        cdef double[::1] grad_average = self.grad_average

        cdef Py_ssize_t size = self.batch.size 
        cdef Py_ssize_t[::1] indices = self.batch.indices
        
        fill_memoryview(grad_average, 0.)

        for j in range(size):
            k = indices[j]
            yk = Y[k]

            y = _model.evaluate(X[k])
            _model.gradient(X[k], grad)

            lval_dy = _loss.derivative(y, yk)
            wk = weights[j]

            vv = lval_dy * wk
            for i in range(n_param):
                grad_average[i] += vv * grad[i]

        if self.regular is not None:
            self.regular.gradient(self.model.param, self.grad_r)
            for i in range(n_param):
                grad_average[i] += self.tau * self.grad_r[i]

cdef class AER(ER):
    #
    def __init__(self, double[:,::1] X, double[::1] Y, 
                 Model model, Loss loss, Average loss_averager=None,
                 FuncMulti regular=None, Batch batch=None, tau=1.0e-3):
        self.X = X
        self.Y = Y
        self.n_sample = len(Y)
        self.model = model
        self.param = model.param
        self.loss = loss
        self.loss_averager = loss_averager
        self.regular = regular
        self.weights = None
        self.grad = None
        self.grad_r = None
        self.grad_average = None
        self.lval_all = None
        self.mval_all = None
        self.tau = tau
        if batch is None:
            self.batch = WholeBatch(self.n_sample)
        else:
            self.batch = batch
        if loss_averager is None:
            self.loss_averager = ArithMean()
        else:
            self.loss_averager = loss_averager
    #
    cpdef init(self):
        ER.init(self)
        size = self.batch.size
        if self.lval_all is None:
            self.lval_all = np.zeros(size, 'd')
        if self.mval_all is None:
            self.mval_all = np.zeros(size, 'd')
    #
    cdef eval_all(self):
        cdef int j, k
        cdef double y
        cdef Model _model = self.model
        cdef Loss _loss = self.loss

        cdef double[:, ::1] X = self.X
        cdef double[::1] Y = self.Y

        cdef Py_ssize_t size = self.batch.size 
        cdef Py_ssize_t[::1] indices = self.batch.indices
        
        cdef double[::1] mval_all = self.mval_all
        cdef double[::1] lval_all = self.lval_all

        for j in range(size):
            k = indices[j]
            mval_all[j] = y = _model.evaluate(X[k])
            lval_all[j] = _loss.evaluate(y, Y[k])
    #
    cdef double evaluate(self):
        cdef Model _model = self.model
        
        self.eval_all()
        self.loss_averager.fit(self.lval_all)
        self.lval = self.loss_averager.u
                    
        if self.regular is not None:
            self.lval += self.tau * self.regular.evaluate(_model.param)                

        return self.lval
    
    cdef void gradient(self):
        cdef Py_ssize_t i, j, k, n_param = self.model.n_param
        cdef double lval_dy, wk

        cdef Model _model = self.model
        cdef Loss _loss = self.loss
        
        cdef double[:, ::1] X = self.X
        cdef double[::1] Y = self.Y

        cdef double[::1] weights = self.weights
        cdef double[::1] grad = self.grad
        cdef double[::1] grad_average = self.grad_average

        cdef Py_ssize_t size = self.batch.size 
        cdef Py_ssize_t[::1] indices = self.batch.indices

        cdef double[::1] mval_all = self.mval_all
        cdef double[::1] lval_all = self.lval_all

        self.eval_all()
        self.loss_averager.fit(lval_all)
        self.loss_averager.gradient(lval_all, weights)
        
        fill_memoryview(grad_average, 0.)
        for j in range(size):
            k = indices[j]
            lval_dy = weights[j] * _loss.derivative(mval_all[j], Y[k])
            
            _model.gradient(X[k], grad)
            
            for i in range(n_param):
                grad_average[i] += lval_dy * grad[i]
                
        if self.regular is not None:
            self.regular.gradient(self.model.param, self.grad_r)
            for i in range(n_param):
                self.grad_average[i] += self.tau * self.grad_r[i]    
    
cdef class ER2(Risk):
    #
    def __init__(self, double[:,::1] X, double[:,::1] Y, MLModel model, MultLoss loss,
                       FuncMulti regular=None, Batch batch=None, tau=1.0e-3):
        self.model = model
        self.param = model.param
        self.loss = loss
        self.regular = regular
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
    #
    def use_weights(self, weights):
        self.weights = weights
    #
    #cdef object get_loss(self):
    #    return self.loss
    #
    cpdef init(self):
        N = self.n_sample    
        n_param = self.model.n_param
        n_output = self.model.n_output

        # if self.model.grad is None:
        #     self.model.grad = np.zeros((n_param,), 'd')
            
        if self.grad is None:
            self.grad = np.zeros(n_param, dtype='d')

        if self.grad_u is None:
            self.grad_u = np.zeros(n_output, dtype='d')

        if self.grad_average is None:
            self.grad_average = np.zeros(n_param, dtype='d')

        if self.regular:
            if self.grad_r is None:
                self.grad_r = np.zeros(n_param, dtype='d')
                
        if self.weights is None:
            self.weights = np.full((N,), 1./N, 'd')
        
        self.lval = 0
    #
    cdef double eval_loss(self, int k):
        self.model.forward(self.X[k])
        return self.loss.evaluate(self.model.output, self.Y[k])
    #
    cdef void gradient_loss(self, int k):
        cdef MLModel _model = self.model
        cdef MultLoss _loss = self.loss
        cdef double[::1] Xk, Yk
        cdef Py_ssize_t j
        cdef double wk = self.weights[k]
        cdef double[::1] grad_u = self.grad_u
        
        Xk = self.X[k]
        Yk = self.Y[k]

        _model.forward(Xk)
        #lval = _loss.evaluate(_model.output, Yk)
        _loss.gradient(_model.output, Yk, self.grad_u)
        
        for j in range(_model.n_output):
            grad_u[j] *= wk

        _model.backward(Xk, grad_u, self.grad)

        #self.lval += lval
    #
    cdef void eval_losses(self, double[::1] lval_all):
        cdef Py_ssize_t j, k, N = self.n_sample
        cdef MLModel _model = self.model
        cdef MultLoss _loss = self.loss
        #cdef double v
        cdef double[:, ::1] X = self.X
        cdef double[:, ::1] Y = self.Y
        cdef double[::1] output = _model.output

        cdef Py_ssize_t size = self.batch.size 
        cdef Py_ssize_t[::1] indices = self.batch.indices

#         for j in prange(size, nogil=True, num_threads=num_procs):
        for j in range(size):
            k = indices[j]
            _model.forward(X[k])
            lval_all[k] = _loss.evaluate(output, Y[k])

        #if self.regular is not None:
        #    v = self.tau * self.regular.evaluate(self.model.param) / N
        #    for k in range(N):
        #        lval_all[k] += v
    #
    cdef double evaluate(self):
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
            lval = _loss.evaluate(output, Y[k])
            S += weights[k] * lval
                    
        if self.regular is not None:
            S += self.tau * self.regular.evaluate(self.model.param)
            
        self.lval = S
        return S
    #
    cdef void gradient(self):
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
            _loss.gradient(output, Yk, grad_u)
            _model.backward(Xk, grad_u, grad)
            
            wk = weights[k]
            
            for i in range(n_param):
                grad_average[i] += wk * grad[i]
                
        if self.regular is not None:
            self.regular.gradient(self.model.param, self.grad_r)
            for i in range(n_param):
                grad_average[i] += self.tau * self.grad_r[i]

    