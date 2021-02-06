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

from libc.math cimport fabs, pow, sqrt, fmax, exp, log

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

cdef class RelativeErrorLoss(Loss):
    #
    def __init__(self, Func func, eps=1.0e-9):
        self.func = func
        self.eps = eps
    #
    cdef double evaluate(self, const double y, const double yk) nogil:
        cdef double eps = self.eps
        cdef double re = (fabs(y) + eps) / (fabs(yk) + eps)
        return self.func.evaluate(re - 1)
    #
    cdef double derivative(self, const double y, const double yk) nogil:
        cdef double eps = self.eps
        cdef double re0 = (fabs(yk) + eps)
        cdef double re = (fabs(y) + eps) / re0
        return self.func.derivative(re - 1) / re0
    #
    cdef double difference(self, const double y, const double yk) nogil:
        cdef double eps = self.eps
        cdef double re = (fabs(y) + eps) / (fabs(yk) + eps)
        return fabs(re - 1)
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

cdef class MinLoss:

    def __init__(self, Loss loss):
        self.loss = loss
    
    cdef double evaluate(self, double[::1] y, double yk) nogil:
        cdef Py_ssize_t i, n = y.shape[0]
        cdef double val, val_min = float_min
        
        for i in range(n):
            val = self.loss.evaluate(y[i], yk)
            if val < val_min:
                val_min = val

        self.val_min = val_min
        return val_min
        
    cdef void gradient(self, double[::1] y, double yk, double[::1] grad) nogil:        
        cdef Py_ssize_t i, n = y.shape[0]
        cdef double val, val_min = self.val_min
        
        for i in range(n):
            val = self.loss.evaluate(y[i], yk)
            if val == val_min:
                grad[i] = 1
            else:
                grad[i] = 0
    
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


