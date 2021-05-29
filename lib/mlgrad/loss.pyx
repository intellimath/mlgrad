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

from libc.math cimport fabs, pow, sqrt, fmax, exp, log
import numpy as np

cdef double float_max = PyFloat_GetMax()
cdef double float_min = PyFloat_GetMin()

cdef class Loss(object):
    #
    cdef double evaluate(self, const double y, const double yk) nogil:
        return 0
    #
    cdef double derivative(self, const double y, const double yk) nogil:
        return 0
    #
    cdef double difference(self, const double x, const double y) nogil:
        return 0

cdef class SquareErrorLoss(Loss):
    #
    cdef double evaluate(self, const double y, const double yk) nogil:
        cdef double r = y - yk
        return r * r
    #
    cdef double derivative(self, const double y, const double yk) nogil:
#         cdef double r = y - yk
        return 2 * (y - yk)
    #
    cdef double difference(self, const double y, const double yk) nogil:
        return fabs(y-yk)
    #
    def _repr_latex_(self):
        return r"$(y - \tilde y)^2$" 

cdef class ErrorLoss(Loss):
    #
    def __init__(self, Func func):
        self.func = func
    #
    cdef double evaluate(self, const double y, const double yk) nogil:
        return self.func.evaluate(y - yk)
    #
    cdef double derivative(self, const double y, const double yk) nogil:
        return self.func.derivative(y-yk)
    #
    cdef double difference(self, const double y, const double yk) nogil:
        return fabs(y-yk)
    #
    def _repr_latex_(self):
        return r"$\ell(y - \tilde y)$" 

cdef class IdErrorLoss(Loss):
    #
    cdef double evaluate(self, const double y, const double yk) nogil:
        return (y - yk)
    #
    cdef double derivative(self, const double y, const double yk) nogil:
        return 1
    #
    cdef double difference(self, const double y, const double yk) nogil:
        return fabs(y-yk)
    #
    def _repr_latex_(self):
        return r"$(y - \tilde y)$" 
    
cdef class RelativeErrorLoss(Loss):
    #
    def __init__(self, Func func):
        self.func = func
    #
    cdef double evaluate(self, const double y, const double yk) nogil:
        cdef double v = fabs(yk) + 1
        cdef double b = v / (v + yk*yk)

        return self.func.evaluate(b * (y - yk))
    #
    cdef double derivative(self, const double y, const double yk) nogil:
        cdef double v = fabs(yk) + 1
        cdef double b = v / (v + yk*yk)

        return b * self.func.derivative(b * (y - yk))
    #
    cdef double difference(self, const double y, const double yk) nogil:
        cdef double v = fabs(yk) + 1
        cdef double b = v / (v + yk*yk)

        return b * fabs(y - yk)
    #
    def _repr_latex_(self):
        return r"$\ell(y - \tilde y)$" 
    
cdef class MarginLoss(Loss):
    #
    def __init__(self, Func func):
        self.func = func
    #
    cdef double evaluate(self, const double u, const double yk) nogil:
        return self.func.evaluate(u*yk)
    #
    cdef double derivative(self, const double u, const double yk) nogil:
        return yk*self.func.derivative(u*yk)
    #
    cdef double difference(self, const double u, const double yk) nogil:
        return -u*yk
    #
    def _repr_latex_(self):
        return r"$\ell(u\tilde y)$" 


cdef class MLoss(Loss):

    def __init__(self, Func rho, Loss loss):
        self.rho = rho
        self.loss = loss

    cdef double evaluate(self, const double y, const double yk) nogil:
        return self.rho.evaluate(self.loss.evaluate(y, yk))
        
    cdef double derivative(self, const double y, const double yk) nogil:
        return self.rho.derivative(self.loss.evaluate(y, yk)) * self.loss.derivative(y, yk)

cdef class MultLoss2:

    cdef double evaluate(self, double[::1] y, double yk) nogil:
        return 0
        
    cdef void gradient(self, double[::1] y, double yk, double[::1] grad) nogil:
        pass
    
cdef class SoftMinLoss2(MultLoss2):

    def __init__(self, Loss lossfunc, q):
        self.lossfunc = lossfunc
        self.q = q
        self.vals = np.zeros(q, 'd')
    
    cdef double evaluate(self, double[::1] y, double yk) nogil:
        cdef Py_ssize_t i, n = self.q
        cdef double val, val_min = float_max
        cdef double S
        cdef double[::1] vals = self.vals
        
        for i in range(n):
            val = vals[i] = self.lossfunc.evaluate(y[i], yk)
            if val < val_min:
                val_min = val
                
        S = 0
        for i in range(n):
            S += exp(val_min - vals[i])
        S = log(S)
        S = val_min - S

        return S
        
    cdef void gradient(self, double[::1] y, double yk, double[::1] grad) nogil:        
        cdef Py_ssize_t i, n = self.q
        cdef double val, val_min = float_max
        cdef double S
        cdef double[::1] vals = self.vals

        for i in range(n):
            val = vals[i] = self.lossfunc.evaluate(y[i], yk)
            if val < val_min:
                val_min = val

        S = 0
        for i in range(n):
            vals[i] = val = exp(val_min - vals[i])
            S += val
                
        for i in range(n):
            grad[i] = vals[i] * self.lossfunc.derivative(y[i], yk) / S
    
cdef class MultLoss:

    cdef double evaluate(self, double[::1] y, double[::1] yk) nogil:
        return 0
        
    cdef void gradient(self, double[::1] y, double[::1] yk, double[::1] grad) nogil:
        pass

cdef class ErrorMultLoss(MultLoss):
    def __init__(self, Func func):
        self.func = func
    
    cdef double evaluate(self, double[::1] y, double[::1] yk) nogil:
        cdef Py_ssize_t i, n = y.shape[0]
        cdef double s = 0
        
        for i in range(n):
            s += self.func.evaluate(y[i] - yk[i])
        return s
        
    cdef void gradient(self, double[::1] y, double[::1] yk, double[::1] grad) nogil:
        cdef Py_ssize_t i, n = y.shape[0]
        for i in range(n):
            grad[i] = self.func.derivative(y[i]-yk[i])
        
    def _repr_latex_(self):
        return r"$\ell(y - \tilde y)$" 

cdef class MarginMultLoss(MultLoss):

    def __init__(self, Func func):
        self.func = func

    cdef double evaluate(self, double[::1] u, double[::1] yk) nogil:
        cdef Py_ssize_t i, n = u.shape[0]
        cdef double s = 0

        for i in range(n):
            s += self.func.evaluate(u[i]*yk[i])
        return s
        
    cdef void gradient(self, double[::1] u, double[::1] yk, double[::1] grad) nogil:
        cdef Py_ssize_t i, n = u.shape[0]
        for i in range(n):
            grad[i] = yk[i] * self.func.derivative(u[i]*yk[i])

    def _repr_latex_(self):
        return r"$\ell(u\tilde y)$" 


