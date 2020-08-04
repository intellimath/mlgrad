# coding: utf-8

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: embedsignature=True
# cython: initializedcheck=False

# The MIT License (MIT)
#
# Copyright (c) <2015-2020> <Shibzukhov Zaur, szport at gmail dot com>
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

from mlgrad.averager cimport ScalarAdaM1

#
# IF USE_OPENMP:
from cython.parallel cimport parallel, prange
#

from openmp cimport omp_get_num_procs

cdef int num_procs = omp_get_num_procs()
if num_procs >= 4:
    num_procs /= 2
else:
    num_procs = 2

cdef double max_float = PyFloat_GetMax() 

import numpy as np

cdef class Penalty(object):
    #
    cdef double evaluate(self, double[::1] Y, double u):
        return 0
    #
    cdef double derivative(self, double[::1] Y, double u):
        return 0
    #
    cdef void gradient(self, double[::1] Y, double u, double[::1] grad):
        pass
    #
    cdef double iterative_next(self, double[::1] Y, double u):
        return 0
    
cdef class PenaltyAverage(Penalty):
    #
    def __init__(self, Func func):
        self.func = func
    #
    cdef double evaluate(self, double[::1] Y, double u):
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double psum
    
        psum = 0
        for k in prange(N, nogil=True, schedule='static', num_threads=num_procs):
#         for k in range(N):
            psum += self.func.evaluate(Y[k] - u)    
        
        return psum / N
    #
    cdef double derivative(self, double[::1] Y, double u):
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double gsum
        
        gsum = 0
        for k in prange(N, nogil=True, schedule='static', num_threads=num_procs):
#         for k in range(N):
            gsum += self.func.derivative(Y[k] - u)                        
            
        return -gsum / N
    #
    cdef double iterative_next(self, double[::1] Y, double u):
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double S, G, v, y
        
        S = 0
        G = 0
        for k in prange(N, nogil=True, schedule='static', num_threads=num_procs):
#         for k in range(N):
            y = Y[k]
            v = self.func.derivative_div_x(y - u)
            G += v
            S += v * y
        
        return S / G
    #
    cdef void gradient(self, double[::1] Y, double u, double[::1] grad):
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double v, S
        
        S = 0
        for k in prange(N, nogil=True, schedule='static', num_threads=num_procs):
#         for k in range(N):
            v = self.func.derivative2(Y[k] - u)
            S += v
            grad[k] = v
                
        for k in prange(N, nogil=True, schedule='static', num_threads=num_procs):
#         for k in range(N):
            grad[k] /= S
    

cdef class PenaltyScale(Penalty):
    #
    def __init__(self, Func func):
        self.func = func
    #
    cdef double evaluate(self, double[::1] Y, double s):
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef Func func = self.func
        cdef double S
    
        S = 0
        for k in prange(N, nogil=True, schedule='static', num_threads=num_procs):
#         for k in range(N):
            S += func.evaluate(Y[k] / s)
    
        return S / N + log(s)
    #
    cdef double derivative(self, double[::1] Y, double s):
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double S, y_k, v
        
        S = 0
        for k in prange(N, nogil=True, schedule='static', num_threads=num_procs):
#         for k in range(N):
            v = Y[k] / s
            S += self.func.derivative(v) * v
            
        return (1 - (S / N)) / s

    cdef double iterative_next(self, double[::1] Y, double s):
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double S, v, y_k
        
        S = 0
        for k in prange(N, nogil=True, schedule='static', num_threads=num_procs):
#         for k in range(N):
            y_k = Y[k]
            S += self.func.derivative_div_x(y_k / s) * y_k * y_k
        
        return sqrt(S / N)
    #
    cdef void gradient(self, double[::1] Y, double s, double[::1] grad):
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double S, v
        
        S = 0
        for k in prange(N, nogil=True, schedule='static', num_threads=num_procs):
#         for k in range(N):
            v = Y[k] / s
            S += self.func.derivative2(v) * v * v
        S += N
            
        for k in prange(N, nogil=True, schedule='static', num_threads=num_procs):
            v = Y[k] / s
            grad[k] = self.func.derivative(v) / S
        
        
cdef class Average(object):
    #
    cdef init(self, double[::1] Y, u0=None):
        
        self.pmin = max_float

        if u0 is not None:
            self.u = u0
        elif self.first:
            self.u = 0
            
        self.first = 0

        self.u_best = self.u        
        
        self.m = 0
    #
    def __call__(self, double[::1] Y): 
        self.fit(Y)
        return self.u
    #
    cpdef fit(self, double[::1] Y, u0=None):
        cdef int j, K, n_iter = self.n_iter
        cdef Penalty penalty = self.penalty
        
        self.init(Y, u0)
        self.pval = penalty.evaluate(Y, self.u)

        K = 1
        #
        while K < n_iter:                
            #
            self.u_prev = self.u
            self.pval_prev = self.pval
            self.fit_epoch(Y)

            self.pval = penalty.evaluate(Y, self.u)
            if self.pval > self.pval_prev:
                for j in range(10):
                    if self.pval <= self.pval_prev:
                        break
                    self.u = 0.5 * (self.u + self.u_prev)
                    self.pval = penalty.evaluate(Y, self.u)
            #
            if self.stop_condition():
                break
            #
            K += 1

        self.K = K
        self.u = self.u_best
    ##
    cdef fit_epoch(self, double[::1] Y):
        return None
    #
    cdef gradient(self, double[::1] Y, double[::1] grad):
        self.penalty.gradient(Y, self.u, grad)
    #
    cdef bint stop_condition(self):
        cdef double p, pmin
        #
        pval = self.pval
        pmin = self.pmin
        
        if pval < pmin:
            self.pmin = pval
            self.u_best = self.u
            self.m = 0
        elif pval == pmin:
            if self.u < self.u_best:
                self.u_best = self.u
        #
        if fabs(pval - pmin) / (1 + fabs(pmin)) < self.tol:
            return 1
            
        if self.m > self.m_iter:
            return 1

        self.m += 1
                
        return 0

    
cdef class ParametrizedAverage(Average):
    #
    def __init__(self, ParametrizedFunc func, Average avr):
        self.func = func
        self.avr = avr
    #
    cpdef fit(self, double[::1] Y, u0=None):
        cdef double u
        cdef Py_ssize_t k, N = Y.shape[0]
        
        self.avr.fit(Y)
        self.func.u = self.avr.u
        u = 0
#         for k in prange(N, nogil=True, schedule='static'):
        for k in range(N):
            u += self.func.evaluate(Y[k])
        u /= N
        self.u = u
    #
    cdef gradient(self, double[::1] Y, double[::1] grad):
        cdef int k, M, N = Y.shape[0]
        cdef double N1, H
        
        self.avr.gradient(Y, grad)
 
        H = 0
#         for k in prange(N, nogil=True, schedule='static'):
        for k in range(N):
            H += self.func.derivative_u(Y[k]) * grad[k]
        
        N1 = 1./N
#         for k in prange(N, nogil=True, schedule='static'):
        for k in range(N):
            grad[k] = (self.func.derivative(Y[k]) + H) * N1
    #

include "avragg_it.pyx"
include "avragg_fg.pyx"

cdef class WMAverage(Average):
    #
    def __init__(self, Average avr, double beta=1):
        self.avr = avr
        self.beta = beta
        self.u = 0
    #
    cpdef fit(self, double[::1] Y, u0=None):
        cdef double u=0, v, yk, avr_u
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double beta = self.beta
        
        self.avr.fit(Y)
        avr_u = self.avr.u

        u = 0
        for k in prange(N, nogil=True, schedule='static', num_threads=num_procs):
#         for k in range(N):
            yk = Y[k]
            if yk <= avr_u:
                v = yk
            else:
                v = beta*avr_u
            u += v
        self.u = u / N
        self.u_best = self.u
    #
    cdef gradient(self, double[::1] Y, double[::1] grad):
        cdef Py_ssize_t k, m, N = Y.shape[0]
        cdef double u, v, N1, yk
        cdef double beta = self.beta

        self.avr.gradient(Y, grad)
        u = self.avr.u

        m = 0
        for k in range(N):
            yk = Y[k]
            if yk > u:
                m += 1

        N1 = 1./N
        for k in prange(N, nogil=True, schedule='static', num_threads=num_procs):
#         for k in range(N):
            yk = Y[k]
            if yk <= u:
                v = 1 + beta * m * grad[k]
            else:
                v = beta * m * grad[k]
            grad[k] = v * N1
    #

cdef class WMAverageMixed(Average):
    #
    def __init__(self, Average avr, double beta=1):
        self.avr = avr
        self.beta = beta
        self.u = 0
    #
    cpdef fit(self, double[::1] Y, u0=None):
        cdef double u, v, yk, avr_u
        cdef Py_ssize_t k, N = Y.shape[0]
        
        self.avr.fit(Y)
        avr_u = self.avr.u

        m = 0
        for k in range(N):
            if Y[k] > avr_u:
                m += 1

        u = 0
        v = 0
        for k in prange(N, nogil=True, schedule='static', num_threads=num_procs):
#         for k in range(N):
            yk = Y[k]
            if yk <= avr_u:
                u += yk
            else:
                v += yk

        self.u = (1-self.beta) * u / (N-m) + self.beta * v / m

        self.u_best = self.u
    #
    cdef gradient(self, double[::1] Y, double[::1] grad):
        cdef Py_ssize_t k, m, N = Y.shape[0]
        cdef double v, N1, N2, yk, avr_u

        self.avr.gradient(Y, grad)
        avr_u = self.avr.u

        m = 0
        for k in range(N):
            if Y[k] > avr_u:
                m += 1

        N1 = (1-self.beta) / (N-m)
        N2 = self.beta / m
        for k in prange(N, nogil=True, schedule='static', num_threads=num_procs):
#         for k in range(N):
            yk = Y[k]
            if yk <= avr_u:
                v = N1
            else:
                v = N2
            grad[k] = v
    #

cdef class WMAverage2(Average):
    #
    def __init__(self, Average avr):
        self.avr = avr
        self.u = 0
    #
    cpdef fit(self, double[::1] Y, u0=None):
        cdef double u, v, yk, avr_u
        cdef Py_ssize_t k, m, N = Y.shape[0]
        
        self.avr.fit(Y)
        u = 0
        m = 0
        avr_u = self.avr.u
#         for k in prange(N, nogil=True, schedule='static', num_threads=num_procs):
        for k in range(N):
            yk = Y[k]
            if yk < avr_u:
                v = yk
            else:
                m += 1
                v = avr_u
            u += v
        u /= N-m
        self.u = u
        self.u_best = self.u
    #
    cdef gradient(self, double[::1] Y, double[::1] grad):
        cdef Py_ssize_t k, m, N = Y.shape[0]
        cdef double u, N1, yk

        self.avr.gradient(Y, grad)
        u = self.avr.u

        m = 0
        for k in range(N):
            yk = Y[k]
            if yk >= u:
                m += 1

        if N == m:
            print("N == m")
        N1 = 1./(N-m)
#         for k in prange(N, nogil=True, schedule='static', num_threads=num_procs):
        for k in range(N):
            yk = Y[k]
            if yk < u:
                grad[k] = N1 * (1 + m * grad[k])
            else:
                grad[k] = N1 * m * grad[k]
    #

cdef class HMAverage(Average):
    #
    def __init__(self, Average avr, n_iter=1000, tol=1.0e-8):
        self.avr = avr
        self.Z = None
        self.u = 0
        self.n_iter = n_iter
        self.tol = tol
    #
    cpdef fit(self, double[::1] Y, u0=None):
        cdef double v, w, yk, avr_z
        cdef double u, u_prev
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double q, S
        cdef int m
        cdef double[::1] Z
        cdef double[::1] grad = np.zeros(N, 'd')
        cdef Average wm = self.avr

        if self.Z is None:
            self.Z = np.zeros(N, 'd')
        Z = self.Z

        if u0 is None:
            wm.fit(Y)
            u = wm.u
#             for k in range(N):
#                 u += Y[k]
#             u /= N
        else:
            u = u0
            
        self.K = 1
        while self.K < self.n_iter:
            u_prev = u
            for k in range(N):
                w = Y[k] - u
                Z[k] = w * w

            self.avr.fit(Z)
            avr_z = sqrt(self.avr.u)
            self.avr.gradient(Z, grad)
            
            m = 0
            for k in range(N):
                if fabs(Y[k] - u) > avr_z:
                    m += 1
            
            v = 0
            for k in prange(N, nogil=True, schedule='static', num_threads=num_procs):
#             for k in range(N):
                yk = Y[k]
                if fabs(yk - u) <= avr_z:
                    w = (1 + m*grad[k]) * yk
                else:
                    w = m*grad[k] * yk
                v += w

            u = v / N
            
            if fabs(u_prev - u) / fabs(1+fabs(u)) < self.tol:
                break

            self.K += 1
        self.u = u
        self.u_best = self.u
    #
    cdef gradient(self, double[::1] Y, double[::1] grad):
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double u, v, w, N1, yk
        cdef double q, avr_z, S
        cdef int m
        cdef double[::1] Z = self.Z

        u = self.u
        for k in range(N):
            w = Y[k] - u
            Z[k] = w * w

        self.avr.fit(Z)
        avr_z = sqrt(self.avr.u)
        self.avr.gradient(Z, grad)

        m = 0
        for k in range(N):
            if fabs(Y[k] - u) > avr_z:
                m += 1

        N1 = 1./ N
        for k in prange(N, nogil=True, schedule='static', num_threads=num_procs):
#         for k in range(N):
            if fabs(Y[k] - u) <= avr_z:
                v = 1 + m*grad[k]
            else:
                v = m*grad[k]
            grad[k] = v * N1
    #
    
cdef class ArithMean(Average):
   # 
    cpdef fit(self, double[::1] Y, u0=None):
        cdef double u
        cdef Py_ssize_t k, N = Y.shape[0]
        
        u = 0
#         for k in prange(N, nogil=True, schedule='static', num_threads=num_procs):
        for k in range(N):
            u += Y[k]
        self.u = u / N
        self.u_best = self.u
    #
    cdef gradient(self, double[::1] Y, double[::1] grad):
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double N1
                 
        N1 = 1./N
#         for k in prange(N, nogil=True, schedule='static', num_threads=num_procs):
        for k in range(N):
            grad[k] = N1


cdef class Minimal(Average):
    #
    cpdef fit(self, double[::1] Y, u0=None):
        cdef double yk, y_min = Y[0]
        cdef int k, N = Y.shape[0]

        for k in range(N):
            yk = Y[k]
            if yk < y_min:
                y_min = yk
        self.u = y_min
        self.u_best = self.u
    #
    cdef gradient(self, double[::1] Y, double[::1] grad):
        cdef int k, N = Y.shape[0]
        cdef int m = 0

        for k in range(N):
            if Y[k] == self.u:
                grad[k] = 1
                m += 1
            else:
                grad[k] = 0

        if m > 1:
            for k in range(N):
                if grad[k] > 0:
                    grad[k] /= m

cdef class Maximal(Average):
    #
    cpdef fit(self, double[::1] Y, u0=None):
        cdef double yk, y_max = Y[0]
        cdef int k, N = Y.shape[0]
        for k in range(N):
            yk = Y[k]
            if yk > y_max:
                y_max = yk
        self.u = y_max
        self.u_best = self.u
    #
    cdef gradient(self, double[::1] Y, double[::1] grad):
        cdef int k, N = Y.shape[0]
        cdef int m = 0
                 
        for k in range(N):
            if Y[k] == self.u:
                grad[k] = 1
                m += 1
            else:
                grad[k] = 0

        if m > 1:
            for k in range(N):
                if grad[k] > 0:
                    grad[k] /= m

cdef class KolmogorovMean(Average):
    #
    def __init__(self, Func func, Func invfunc):
        self.func = func
        self.invfunc = invfunc
    #
    cpdef fit(self, double[::1] Y, u0=None):
        cdef double u, yk
        cdef int k, N = Y.shape[0]
        
        u = 0
#         for k in prange(N, nogil=True, schedule='static'):
        for k in range(N):
            yk = Y[k]
            u += self.func.evaluate(yk)
        u /= N
        self.uu = u
        self.u = self.invfunc.evaluate(u)
        self.u_best = self.u
    #
    cdef gradient(self, double[::1] Y, double[::1] grad):
        cdef int k, N = Y.shape[0]
        cdef double V
        
        V = self.invfunc.derivative(self.uu)
#         for k in prange(N, nogil=True, schedule='static'):
        for k in range(N):
            grad[k] = self.func.derivative(Y[k]) * V

cdef class SoftMinimal(Average):
    #
    def __init__(self, a):
        self.a = a
    #
    cpdef fit(self, double[::1] Y, u0=None):
        cdef double u, yk
        cdef int k, N = Y.shape[0]
        
        u = 0
#         for k in prange(N, nogil=True, schedule='static'):
        for k in range(N):
            yk = Y[k]
            u += exp(-yk*self.a)
        u /= N
        self.u = - log(u) / self.a
        self.u_best = self.u
    #
    cdef gradient(self, double[::1] Y, double[::1] grad):
        cdef int k, l, N = Y.shape[0]
        cdef double u, yk, yl
        
        for l in range(N):
            u = 0
            yl = Y[l]
#             for k in prange(N, nogil=True, schedule='static'):
            for k in range(N):
                yk = Y[k] - yl
                u += exp(-yk*self.a)
        
            grad[l] = 1. / u

cdef inline double nearest_value(double[::1] u, double y):
    cdef Py_ssize_t j, K = u.shape[0]
    cdef double u_j, u_min=0, d_min = max_float

    for j in range(K):
        u_j = u[j]
        d = fabs(y - u_j)
        if d < d_min:
            d_min = d
            u_min = u_j
    return u_min

cdef inline Py_ssize_t nearest_index(double[::1] u, double y):
    cdef Py_ssize_t j, j_min, K = u.shape[0]
    cdef double u_j, d_min = max_float

    for j in range(K):
        u_j = u[j]
        d = fabs(y - u_j)
        if d < d_min:
            d_min = d
            j_min = j
    return j_min

# cdef class KPenaltyAverage(Penalty):
#     #
#     def __init__(self, Func func):
#         self.func = func
#     #
#     cdef double evaluate(self, double[::1] Y, double[::1] u):
#         cdef Py_ssize_t j, jmin, k, N = Y.shape[0], K = u.shape[0]
#         cdef double psum, y
    
#         psum = 0
# #         for k in prange(N, nogil=True, schedule='static', num_threads=num_procs):
#         for k in range(N):
#             y = nearest_value(u, Y[k])
#             psum += self.func.evaluate(y)
        
#         return psum / N
#     #
#     cdef void gradient_u(self, double[::1] Y, double[::1] u, double[::1] grad):
#         cdef Py_ssize_t k, j, N = Y.shape[0]
#         cdef double y
        
#         fill_memoryview(grad, 0)
# #         for k in prange(N, nogil=True, schedule='static', num_threads=num_procs):
#         for k in range(N):
#             y = Y[k]
#             j = nearest_index(u, y)
#             grad[j] += self.func.derivative(y - u[j])                        
        
#         for j in range(u.shape[0]):
#             grad[j] /= N
#     #
#     cdef double iterative_next(self, double[::1] Y, double u):
#         cdef Py_ssize_t k, N = Y.shape[0]
#         cdef Func func = self.func
#         cdef double gsum, G, v, y
        
#         gsum = 0
#         G = 0
# #         for k in prange(N, nogil=True, schedule='static', num_threads=num_procs):
#         for k in range(N):
#             y = nearest_value(u, Y[k])
#             v = self.func.derivative_div_x(y)
#             G += v
#             gsum += v * y
        
#         return gsum / G
#     #
#     cdef void gradient(self, double[::1] Y, double u, double[::1] grad):
#         cdef Py_ssize_t k, N = Y.shape[0]
#         cdef Func func = self.func
#         cdef double v, S
        
#         S = 0
#         for k in prange(N, nogil=True, schedule='static', num_threads=num_procs):
# #         for k in range(N):
#             v = func.derivative2(Y[k] - u)
#             S += v
#             grad[k] = v
                
#         for k in prange(N, nogil=True, schedule='static', num_threads=num_procs):
# #         for k in range(N):
#             grad[k] /= S

            
# cdef class KAverages(object):
#     #
#     def use_deriv_averager(self, averager):
#         self.deriv_averager = averager
#     #
#     cdef init(self, double[::1] Y, u0=None):
#         if self.deriv_averager is not None:
#             self.deriv_averager.init()
        
#         self.pmin = PyFloat_GetMax()

#         if u0 is not None:
#             self.u = u0
#         elif self.first:
#             self.u = 0
            
#         self.first = 0

#         self.u_best = self.u        
        
#         self.m = 0
#     #
#     def __call__(self, double[::1] Y): 
#         self.fit(Y)
#         return self.u
#     #
#     cpdef fit(self, double[::1] Y, u0=None):
#         cdef int j, K, n_iter = self.n_iter
#         cdef Penalty penalty = self.penalty
        
#         self.init(Y, u0)
#         self.pval = penalty.evaluate(Y, self.u)

#         K = 1
#         #
#         while K < n_iter:                
#             #
#             self.u_prev = self.u
#             self.pval_prev = self.pval
#             self.fit_epoch(Y)

#             self.pval = penalty.evaluate(Y, self.u)
#             if self.pval > self.pval_prev:
#                 for j in range(10):
#                     if self.pval <= self.pval_prev:
#                         break
#                     self.u = 0.5 * (self.u + self.u_prev)
#                     self.pval = penalty.evaluate(Y, self.u)
#             #
#             if self.stop_condition():
#                 break
#             #
#             K += 1

#         self.K = K
#         self.u = self.u_best
#     ##
#     cdef fit_epoch(self, double[::1] Y):
#         return None
#     #
#     cdef gradient(self, double[::1] Y, double[::1] grad):
#         self.penalty.gradient(Y, self.u, grad)
#     #
#     cdef bint stop_condition(self):
#         cdef double p, pmin

#         pval = self.pval
#         pmin = self.pmin
        
#         if pval < pmin:
#             self.pmin = pval
#             self.u_best = self.u
#             self.m = 0

#         if fabs(pval - pmin) / (1 + fabs(pval)) < self.tol:
#             return 1
            
#         if self.m > self.m_iter:
#             return 1

#         self.m += 1
                
#         return 0
