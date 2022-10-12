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

from libc.math cimport fabs, pow, sqrt, fmax, exp, log
import numpy as np

cdef double double_max = PyFloat_GetMax()
cdef double double_min = PyFloat_GetMin()

cdef class Loss(object):
    #
    cdef double _evaluate(self, const double y, const double yk) nogil:
        return 0
    #
    cdef double _derivative(self, const double y, const double yk) nogil:
        return 0
    #
    cdef double _derivative_div(self, const double y, const double yk) nogil:
        return 0
    #
    cdef double _residual(self, const double y, const double yk) nogil:
        return 0
    #
    def evaluate_all(self, double[::1] Y, double[::1] Yp):
        cdef Py_ssize_t k, N = Y.shape[0]
        
        L = np.empty(N, 'd')
        for k in range(N):
            L[k] = self._evaluate(Y[k], Yp[k])
        return L
    #
    def evaluate_all2(self, Model model, double[:, ::1] X, double[::1] Yp):
        cdef Py_ssize_t k, N = Yp.shape[0]
        cdef double y
        
        L = np.empty(N, 'd')
        for k in range(N):
            y = model._evaluate(X[k])
            L[k] = self._evaluate(y, Yp[k])
        return L
    #
    def derivative_div_all(self, double[::1] Y, double[::1] Yp):
        cdef Py_ssize_t k, N = Y.shape[0]
        
        L = np.empty(N, 'd')
        for k in range(N):
            L[k] = self._derivative_div(Y[k], Yp[k])
        return L
    #
    def derivative_all(self, double[::1] Y, double[::1] Yp):
        cdef Py_ssize_t k, N = Y.shape[0]
        
        L = np.empty(N, 'd')
        for k in range(N):
            L[k] = self._derivative(Y[k], Yp[k])
        return L
    #
    def __call__(self, y, yk):
        return self._evaluate(y, yk)
    #
    def deriv(self, y, yk):
        return self._derivative(y, yk)
    #
    def get_attr(self, name):
        return getattr(self, name)
    
cdef class LossFunc(Loss):
    #
    pass

cdef class SquareErrorLoss(LossFunc):
    #
    def __init__(self):
        self.func = Square()
    #
    cdef double _evaluate(self, const double y, const double yk) nogil:
        cdef double r = y - yk
        return 0.5 * r * r 
    #
    cdef double _derivative(self, const double y, const double yk) nogil:
        return (y - yk)
    #
    cdef double _derivative_div(self, const double y, const double yk) nogil:
        return 1
    #
    cdef double _residual(self, const double y, const double yk) nogil:
        return fabs(y-yk)
    #
    def _repr_latex_(self):
        return r"$(y - \tilde y)^2$" 

cdef class ErrorLoss(LossFunc):
    #
    def __init__(self, Func func):
        self.func = func
    #
    cdef double _evaluate(self, const double y, const double yk) nogil:
        return self.func._evaluate(y - yk)
    #
    cdef double _derivative(self, const double y, const double yk) nogil:
        return self.func._derivative(y-yk)
    #
    cdef double _derivative_div(self, const double y, const double yk) nogil:
        return self.func._derivative_div_x(y-yk)
    #
    cdef double _residual(self, const double y, const double yk) nogil:
        return fabs(y-yk)
    #
    def _repr_latex_(self):
        return r"$\ell(y - \tilde y)$" 

cdef class IdErrorLoss(Loss):
    #
    cdef double _evaluate(self, const double y, const double yk) nogil:
        return (y - yk)
    #
    cdef double _derivative(self, const double y, const double yk) nogil:
        return 1
    #
    cdef double _residual(self, const double y, const double yk) nogil:
        return fabs(y-yk)
    #
    def _repr_latex_(self):
        return r"$(y - \tilde y)$" 
    
cdef class RelativeErrorLoss(LossFunc):
    #
    def __init__(self, Func func):
        self.func = func
    #
    cdef double _evaluate(self, const double y, const double yk) nogil:
        cdef double v = fabs(yk) + 1
        cdef double b = v / (v + yk*yk)

        return self.func._evaluate(b * (y - yk))
    #
    cdef double _derivative(self, const double y, const double yk) nogil:
        cdef double v = fabs(yk) + 1
        cdef double b = v / (v + yk*yk)

        return b * self.func._derivative(b * (y - yk))
    #
    cdef double _derivative_div(self, const double y, const double yk) nogil:
        cdef double v = fabs(yk) + 1
        cdef double b = v / (v + yk*yk)

        return b * self.func._derivative_div_x(b * (y - yk))
    #
    cdef double _residual(self, const double y, const double yk) nogil:
        cdef double v = fabs(yk) + 1
        cdef double b = v / (v + yk*yk)

        return b * fabs(y - yk)
    #
    def _repr_latex_(self):
        return r"$\ell(y - \tilde y)$" 
    
cdef class MarginLoss(LossFunc):
    #
    def __init__(self, Func func):
        self.func = func
    #
    cdef double _evaluate(self, const double u, const double yk) nogil:
        return self.func._evaluate(u*yk)
    #
    cdef double _derivative(self, const double u, const double yk) nogil:
        return yk*self.func._derivative(u*yk)
    #
    cdef double _derivative_div(self, const double u, const double yk) nogil:
        return yk*self.func._derivative_div_x(u*yk)
    #
    cdef double _residual(self, const double u, const double yk) nogil:
        return self.func._value(u*yk)
    #
    def _repr_latex_(self):
        return r"$\ell(u\tilde y)$" 


cdef class MLoss(Loss):

    def __init__(self, Func rho, Loss loss):
        self.rho = rho
        self.loss = loss

    cdef double _evaluate(self, const double y, const double yk) nogil:
        return self.rho._evaluate(self.loss._evaluate(y, yk))
        
    cdef double _derivative(self, const double y, const double yk) nogil:
        return self.rho._derivative(self.loss._evaluate(y, yk)) * self.loss._derivative(y, yk)

cdef class MultLoss:

    cdef double _evaluate(self, double[::1] y, double yk) nogil:
        return 0
        
    cdef void _gradient(self, double[::1] y, double yk, double[::1] grad) nogil:
        pass
    
cdef class SoftMinLoss(MultLoss):

    def __init__(self, Loss lossfunc, q, a=1):
        self.lossfunc = lossfunc
        self.q = q
        self.a = a
        self.vals = np.zeros(q, 'd')
    
    cdef double _evaluate(self, double[::1] y, double yk) nogil:
        cdef Py_ssize_t i, n = self.q
        cdef double val, val_min = double_max
        cdef double S, a = self.a
        cdef double[::1] vals = self.vals
       
        for i in range(n):
            val = vals[i] = self.lossfunc._evaluate(y[i], yk)
            if val < val_min:
                val_min = val

        S = 0
        for i in range(n):
            S += exp(a*(val_min - vals[i]))
        S = log(S) /a
        S -= val_min

        return -S

    cdef void _gradient(self, double[::1] y, double yk, double[::1] grad) nogil:        
        cdef Py_ssize_t i, n = self.q
        cdef double val, val_min = double_max
        cdef double S, a = self.a
        cdef double[::1] vals = self.vals

        for i in range(n):
            val = vals[i] = self.lossfunc._evaluate(y[i], yk)
            if val < val_min:
                val_min = val

        S = 0
        for i in range(n):
            vals[i] = val = exp(a*(val_min - vals[i]))
            S += val

        for i in range(n):
            vals[i] /= S

        for i in range(n):
            grad[i] = vals[i] * self.lossfunc._derivative(y[i], yk)
            
cdef class SquareErrorMultiLoss(MultLoss):
    
    cdef double _evaluate(self, double[::1] y, double yk) nogil:
        cdef Py_ssize_t i, n = y.shape[0]
        cdef double val, S

        S = 0
        for i in range(n):
            val = y[i] - yk
            S += val * val

        return S/2

    cdef void _gradient(self, double[::1] y, double yk, double[::1] grad) nogil:        
        cdef Py_ssize_t i, n = y.shape[0]

        for i in range(n):
            grad[i] = y[i] - yk            

cdef class ErrorMultiLoss(MultLoss):
    
    def __init__(self, Func, func):
            self.func = func
    
    cdef double _evaluate(self, double[::1] y, double yk) nogil:
        cdef Py_ssize_t i, n = y.shape[0]
        cdef double S

        S = 0
        for i in range(n):
            S += self.func._evaluate(y[i] - yk)

        return S

    cdef void _gradient(self, double[::1] y, double yk, double[::1] grad) nogil:        
        cdef Py_ssize_t i, n = y.shape[0]

        for i in range(n):
            grad[i] = self.func._derivative(y[i] - yk)

cdef class MarginMultiLoss(MultLoss):
    
    def __init__(self, Func, func):
            self.func = func
    
    cdef double _evaluate(self, double[::1] y, double yk) nogil:
        cdef Py_ssize_t i, n = y.shape[0]
        cdef double S

        S = 0
        for i in range(n):
            S += self.func._evaluate(y[i] * yk)

        return S

    cdef void _gradient(self, double[::1] y, double yk, double[::1] grad) nogil:        
        cdef Py_ssize_t i, n = y.shape[0]

        for i in range(n):
            grad[i] = yk * self.func._derivative(y[i] * yk)
            
# cdef class MultLoss2:

#     cdef double _evaluate(self, double[::1] y, double[::1] yk) nogil:
#         return 0

#     cdef void _gradient(self, double[::1] y, double[::1] yk, double[::1] grad) nogil:
#         pass

# cdef class ErrorMultLoss2(MultLoss2):
#     def __init__(self, Func func):
#         self.func = func
    
#     cdef double _evaluate(self, double[::1] y, double[::1] yk) nogil:
#         cdef Py_ssize_t i, n = y.shape[0]
#         cdef double s = 0
        
#         for i in range(n):
#             s += self.func._evaluate(y[i] - yk[i])
#         return s
        
#     cdef void _gradient(self, double[::1] y, double[::1] yk, double[::1] grad) nogil:
#         cdef Py_ssize_t i, n = y.shape[0]
#         for i in range(n):
#             grad[i] = self.func._derivative(y[i]-yk[i])
        
#     def _repr_latex_(self):
#         return r"$\ell(y - \tilde y)$" 

# cdef class MarginMultLoss2(MultLoss2):

#     def __init__(self, Func func):
#         self.func = func

#     cdef double _evaluate(self, double[::1] u, double[::1] yk) nogil:
#         cdef Py_ssize_t i, n = u.shape[0]
#         cdef double s = 0

#         for i in range(n):
#             s += self.func._evaluate(u[i]*yk[i])
#         return s
        
#     cdef void _gradient(self, double[::1] u, double[::1] yk, double[::1] grad) nogil:
#         cdef Py_ssize_t i, n = u.shape[0]
#         for i in range(n):
#             grad[i] = yk[i] * self.func._derivative(u[i]*yk[i])

#     def _repr_latex_(self):
#         return r"$\ell(u\tilde y)$" 

