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

cimport cython
from mlgrad.func cimport Func, ParameterizedFunc
from libc.math cimport fabs, pow, sqrtf, fmax, log, exp

from cython.parallel cimport parallel, prange
 
from openmp cimport omp_get_num_procs
 
cdef int num_procs = 2 #omp_get_num_procs()
# if num_procs >= 4:
#     num_procs /= 2
# else:
#     num_procs = 2

cdef double max_double = PyFloat_GetMax() 

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

@cython.final
cdef class PenaltyAverage(Penalty):
    #
    def __init__(self, Func func):
        self.func = func
    #
    @cython.cdivision(True)
    @cython.final
    cdef double evaluate(self, double[::1] Y, double u):
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double *YY = &Y[0]
        cdef double S
        cdef Func func = self.func

        S = 0
        for k in prange(N, nogil=True, num_threads=num_procs):
        # for k in range(N):
            S += func.evaluate(YY[k] - u)

        return S / N
    # 
    @cython.cdivision(True)
    @cython.final
    cdef double derivative(self, double[::1] Y, double u):
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double *YY = &Y[0]
        cdef double S
        cdef Func func = self.func
        
        S = 0
        for k in prange(N, nogil=True, num_threads=num_procs):
        # for k in range(N):
            S += func.derivative(YY[k] - u)                        
    
        return -S / N
    #
    @cython.cdivision(True)
    @cython.final
    cdef double iterative_next(self, double[::1] Y, double u):
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double *YY = &Y[0]
        cdef double S, V, v, yk
        cdef Func func = self.func

        S = 0
        V = 0
        for k in prange(N, nogil=True, num_threads=num_procs):
        # for k in range(N):
            yk = YY[k]
            v = func.derivative_div_x(yk - u)
            V += v
            S += v * yk
        
        return S / V
    #
    @cython.cdivision(True)
    @cython.final
    cdef void gradient(self, double[::1] Y, double u, double[::1] grad):
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double *YY = &Y[0]
        cdef double v, S
        cdef double *GG = &grad[0]
        cdef Func func = self.func
        
        S = 0
        for k in prange(N, nogil=True, num_threads=num_procs):
        # for k in range(N):
            GG[k] = v = func.derivative2(YY[k] - u)
            S += v
        
        for k in prange(N, nogil=True, num_threads=num_procs):
        # for k in range(N):
            GG[k] /= S

@cython.final
cdef class PenaltyScale(Penalty):
    #
    def __init__(self, Func func):
        self.func = func
    #
    @cython.cdivision(True)
    @cython.final
    cdef double evaluate(self, double[::1] Y, double s):
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef Func func = self.func
        cdef double S
        cdef double v

        S = 0
        for k in prange(N, nogil=True, num_threads=num_procs):
        # for k in range(N):
            v = Y[k]    
            S += func.evaluate(v / s)
    
        return S / N + log(s)
    #
    @cython.cdivision(True)
    @cython.final
    cdef double derivative(self, double[::1] Y, double s):
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double S, v
        cdef Func func = self.func
        
        S = 0
        for k in prange(N, nogil=True, num_threads=num_procs):
        # for k in range(N):
            v = Y[k] / s
            S += func.derivative(v) * v
            
        return (1 - (S / N)) / s
    #
    @cython.cdivision(True)
    @cython.final
    cdef double iterative_next(self, double[::1] Y, double s):
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double S, v, y_k
        cdef Func func = self.func
        
        S = 0
        for k in prange(N, nogil=True, num_threads=num_procs):
        # for k in range(N):
            y_k = Y[k]
            S += func.derivative_div_x(y_k / s) * y_k * y_k
        
        return sqrtf(S / N)
    #
    @cython.cdivision(True)
    @cython.final
    cdef void gradient(self, double[::1] Y, double s, double[::1] grad):
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double S, v
        cdef Func func = self.func
        
        S = 0
        for k in prange(N, nogil=True, num_threads=num_procs):
        # for k in range(N):
            v = Y[k] / s
            S += func.derivative2(v) * v * v
        S += N
            
        for k in prange(N, nogil=True, num_threads=num_procs):
            v = Y[k] / s
            grad[k] = func.derivative(v) / S
        
        
cdef class Average(object):
    #
    cdef init(self, double[::1] Y, u0=None):
        
        self.pmin = max_double/2
        
        if u0 is not None:
            self.u = u0
        else:
            self.u = array_mean(Y)

        self.u_best = self.u        
        
        self.m = 0
        
        if self.h < 0:
            self.h = 0.1
    #
    def __call__(self, double[::1] Y): 
        self.fit(Y)
        return self.u
    #
    def nabla(self, double[::1] Y):
        grad = np.zeros(len(Y), 'd')
        self.gradient(Y, grad)
        return grad
    #
    cpdef fit(self, double[::1] Y, u0=None):
        cdef int j, K, n_iter = self.n_iter
        cdef Penalty penalty = self.penalty
        cdef double h = self.h, h1 = 1-h
        
        self.init(Y, u0)
        self.pval = penalty.evaluate(Y, self.u)
        if self.pval < self.pmin:
            self.pmin = self.pval
            self.u_best = self.u

        K = 1
        #
        while K < n_iter:                
            #
            self.u_prev = self.u
            self.pval_prev = self.pval
            self.fit_epoch(Y)

            self.pval = penalty.evaluate(Y, self.u)
            #
            if self.pval < self.pmin:
                self.pmin = self.pval
                self.u_best = self.u
                self.m = 0
            # elif self.pval > self.pval_prev:
            #     self.u = h1 * self.u_prev + h * self.u
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
    @cython.cdivision(True)
    cdef bint stop_condition(self):
#         cdef double p, pmin, prev
#         #
#         pval = self.pval
#         pmin = self.pmin
#         prev = self.pval_prev
        
#         if pval < pmin:
#             pmin = self.pmin = pval
#             self.u_best = self.u
#             self.m = 0
#         elif pval == pmin:
#             if self.u < self.u_best:
#                 self.u_best = self.u
        #
        if fabs(self.pval - self.pmin) / (1. + fabs(self.pmin)) < self.tol:
            return 1
            
        if self.m > self.m_iter:
            return 1

        self.m += 1
                
        return 0

include "avragg_it.pyx"
include "avragg_fg.pyx"


@cython.final
cdef class MAverage(Average):
    #
    def __init__(self, Func func, tol=1.0e-9, n_iter=1000): #, gamma=0.1):
        self.func = func
        self.n_iter = n_iter
        self.tol = tol
        # self.gamma = gamma
    #
    @cython.cdivision(True)
    @cython.final
    cpdef fit(self, double[::1] Y, u0=None):
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double *YY = &Y[0]
        cdef double yk, v, S, V, Z
        cdef double u, u_min, z, z_min
        cdef double tol = self.tol
        cdef Func func = self.func
        cdef int K
        cdef bint finish = 0
        
        if u0 is None:
            u = (YY[0] + YY[N//2] + YY[N-1]) / 3
            # u = array_mean(Y)
        else:
            u = u0
        
        u_min = u
        z_min = max_double

        for K in range(self.n_iter):
            S = 0
            V = 0
            Z = 0
            for k in prange(N, nogil=True, num_threads=num_procs):
            # for k in range(N):
                yk = YY[k]
                v = func.derivative_div_x(yk - u)
                S += v * yk
                V += v
                Z += func.evaluate(yk - u)

            u = S / V
            z = Z / N

            if K > 0 and fabs(z - z_min) / (1 + fabs(z_min)) < tol:
                finish = 1
            
            if z < z_min:
                z_min = z
                u_min = u
                
            if finish:
                break

        self.u = u_min
        self.u_best = u_min
    #
    @cython.cdivision(True)
    @cython.final
    cdef gradient(self, double[::1] Y, double[::1] grad):
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double *YY = &Y[0]
        cdef double *GG = &grad[0]
        cdef double v, V, u=self.u
        cdef Func func = self.func
        
        V = 0
        for k in prange(N, nogil=True, num_threads=num_procs):
        # for k in range(N):
            GG[k] = v = func.derivative2(YY[k] - u)
            V += v
            
        # for k in prange(N, nogil=True, num_threads=num_procs):
        for k in range(N):
            GG[k] /= V
     
@cython.final
cdef class ParameterizedAverage(Average):
    #
    def __init__(self, ParameterizedFunc func, Average avr):
        self.func = func
        self.avr = avr
    #
    @cython.final
    @cython.cdivision(True)
    cpdef fit(self, double[::1] Y, u0=None):
        cdef Py_ssize_t k
        cdef Py_ssize_t N = Y.shape[0], M
        cdef double c
        cdef double S = 0
#         cdef double u1,u2,u3,u4
#         cdef double v1,v2,v3,v4
        cdef double *YY = &Y[0]
        cdef ParameterizedFunc func = self.func
        
        self.avr.fit(Y, u0)
        c = self.avr.u 

        for k in prange(N, nogil=True, num_threads=num_procs):
    #         for k in range(N):
            S += func.evaluate(YY[k], c)

        self.u = S / N
    #
    @cython.cdivision(True)
    @cython.final
    cdef gradient(self, double[::1] Y, double[::1] grad):
        cdef Py_ssize_t k
        cdef Py_ssize_t N = Y.shape[0], M
        cdef double c, v
        cdef double N1 = 1.0/N
        cdef double H, S
        cdef double *YY = &Y[0]
        cdef double *GG
        cdef ParameterizedFunc func = self.func
        
        self.avr.gradient(Y, grad)
        c = self.avr.u
        
        H = 0
        for k in prange(N, nogil=True, num_threads=num_procs):
            H += func.derivative_u(YY[k], c)
        H *= N1

        S = 0
        GG = &grad[0]
        for k in prange(N, nogil=True, num_threads=num_procs):
            v = N1 * func.derivative(YY[k], c) +  H * GG[k]
            GG[k] = v
            S += v
        
        v = S
        for k in prange(N, nogil=True, num_threads=num_procs):
            GG[k] /= v
    #

@cython.final
cdef class WMAverage(Average):
    #
    def __init__(self, Average avr):
        self.avr = avr
        self.u = 0
    #
    @cython.cdivision(True)
    @cython.final
    cpdef fit(self, double[::1] Y, u0=None):
        cdef double v, yk, avr_u
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double *YY = &Y[0]
        cdef double S
        
        self.avr.fit(Y, u0)
        avr_u = self.avr.u

        S = 0
        for k in prange(N, nogil=True, num_threads=num_procs):
        # for k in range(N):
            yk = YY[k]
            v = yk if yk <= avr_u else avr_u
            S += v

        self.u = S / N
        self.u_best = self.u
        self.K = self.avr.K
    #
    @cython.cdivision(True)
    @cython.final
    cdef gradient(self, double[::1] Y, double[::1] grad):
        cdef Py_ssize_t k, m, N = Y.shape[0]
        cdef double u, v, gk
        cdef double N1 = 1.0/N
        cdef double *YY = &Y[0]
        cdef double *GG = &grad[0]

        # self.avr.fit(Y, self.avr.u)
        self.avr.gradient(Y, grad)
        u = self.avr.u

        m = 0
        for k in range(N):
            if YY[k] > u:
                m += 1

        for k in prange(N, nogil=True, num_threads=num_procs):
        # for k in range(N):
            gk = GG[k]
            v = (1 + m * gk) if YY[k] <= u else (m * gk)
            GG[k] = v * N1
    #

cdef class WMAverageMixed(Average):
    #
    def __init__(self, Average avr, double gamma=1):
        self.avr = avr
        self.gamma = gamma
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
#         for k in prange(N, nogil=True, num_threads=num_procs):
        for k in range(N):
            yk = Y[k]
            if yk <= avr_u:
                u += yk
            else:
                v += yk

        self.u = (1-self.gamma) * u / (N-m) + self.gamma * v / m

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

        N1 = (1-self.gamma) / (N-m)
        N2 = self.gamma / m
#         for k in prange(N, nogil=True, num_threads=num_procs):
        for k in range(N):
            yk = Y[k]
            if yk <= avr_u:
                v = N1
            else:
                v = N2
            grad[k] = v
    #

cdef class TMAverage(Average):
    #
    def __init__(self, Average avr):
        self.avr = avr
        self.u = 0
    #
    cpdef fit(self, double[::1] Y, u0=None):
        cdef double u, v, yk, avr_u
        cdef Py_ssize_t k, M, N = Y.shape[0]
        
        self.avr.fit(Y)
        u = 0
        M = 0
        avr_u = self.avr.u
#         for k in prange(N, nogil=True, num_threads=num_procs):
        for k in range(N):
            yk = Y[k]
            if yk <= avr_u:
                u += yk
                M += 1

        self.u = u / M
        self.u_best = self.u
        self.K = self.avr.K
    #
    cdef gradient(self, double[::1] Y, double[::1] grad):
        cdef Py_ssize_t k, M, N = Y.shape[0]
        cdef double u, N1, yk

        self.avr.gradient(Y, grad)
        u = self.avr.u

        M = 0
        for k in range(N):
            yk = Y[k]
            if yk <= u:
                M += 1

        N1 = 1./M
#         for k in prange(N, nogil=True, num_threads=num_procs):
        for k in range(N):
            yk = Y[k]
            if yk <= u:
                grad[k] = N1
            else:
                grad[k] = 0
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
    @cython.cdivision(True)
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
            avr_z = sqrtf(self.avr.u)
            self.avr.gradient(Z, grad)
            
            m = 0
            for k in range(N):
                if fabs(Y[k] - u) > avr_z:
                    m += 1
            
            v = 0
#             for k in prange(N, nogil=True, num_threads=num_procs):
            for k in range(N):
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
    @cython.cdivision(True)
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
        avr_z = sqrtf(self.avr.u)
        self.avr.gradient(Z, grad)

        m = 0
        for k in range(N):
            if fabs(Y[k] - u) > avr_z:
                m += 1

        N1 = 1./ N
#         for k in prange(N, nogil=True, num_threads=num_procs):
        for k in range(N):
            if fabs(Y[k] - u) <= avr_z:
                v = 1 + m*grad[k]
            else:
                v = m*grad[k]
            grad[k] = v * N1
    #
    
cdef class ArithMean(Average):
    #
    @cython.cdivision(True)
    cpdef fit(self, double[::1] Y, u0=None):
        cdef double u
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double *YY =&Y[0]
        
        u = 0
        for k in prange(N, nogil=True, num_threads=num_procs):
#         for k in range(N):
            u += YY[k]
        self.u = u / N
        self.u_best = self.u
    #
    @cython.cdivision(True)
    cdef gradient(self, double[::1] Y, double[::1] grad):
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double *GG = &grad[0]
        cdef double v
                 
        v = 1./N
        for k in prange(N, nogil=True, num_threads=num_procs):
#         for k in range(N):
            GG[k] = v

cdef class Minimal(Average):
    #
    cpdef fit(self, double[::1] Y, u0=None):
        cdef double yk, y_min = Y[0]
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double *y = &Y[0]

        for k in range(N):
            yk = y[k]
            if yk < y_min:
                y_min = yk
        self.u = y_min
        self.u_best = self.u
    #
    cdef gradient(self, double[::1] Y, double[::1] grad):
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef int m = 0
        cdef double u = self.u
        cdef double *g = &grad[0]
        cdef double *y = &Y[0]

        for k in range(N):
            if y[k] == u:
                g[k] = 1
                m += 1
            else:
                g[k] = 0

        if m > 1:
            for k in range(N):
                if g[k] > 0:
                    g[k] /= m

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
#         for k in prange(N, nogil=True):
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
#         for k in prange(N, nogil=True):
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
        cdef double a = self.a
        
        u = 0
#         for k in prange(N, nogil=True):
        for k in range(N):
            yk = Y[k]
            u += exp(-yk*a)
        u /= N
        self.u = - log(u) / a
        self.u_best = self.u
    #
    cdef gradient(self, double[::1] Y, double[::1] grad):
        cdef int k, l, N = Y.shape[0]
        cdef double u, yk, yl
        cdef double a = self.a
        
        for l in range(N):
            u = 0
            yl = Y[l]
            for k in range(N):
                yk = Y[k] - yl
                u += exp(-yk*a)
        
            grad[l] = 1. / u

cdef inline double nearest_value(double[::1] u, double y):
    cdef Py_ssize_t j, K = u.shape[0]
    cdef double u_j, u_min=0, d_min = max_double

    for j in range(K):
        u_j = u[j]
        d = fabs(y - u_j)
        if d < d_min:
            d_min = d
            u_min = u_j
    return u_min

cdef inline Py_ssize_t nearest_index(double[::1] u, double y):
    cdef Py_ssize_t j, j_min, K = u.shape[0]
    cdef double u_j, d_min = max_double

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
# #         for k in prange(N, nogil=True, num_threads=num_procs):
#         for k in range(N):
#             y = nearest_value(u, Y[k])
#             psum += self.func.evaluate(y)
        
#         return psum / N
#     #
#     cdef void gradient_u(self, double[::1] Y, double[::1] u, double[::1] grad):
#         cdef Py_ssize_t k, j, N = Y.shape[0]
#         cdef double y
        
#         fill_memoryview(grad, 0)
# #         for k in prange(N, nogil=True, num_threads=num_procs):
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
# #         for k in prange(N, nogil=True, num_threads=num_procs):
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
#         for k in prange(N, nogil=True, num_threads=num_procs):
# #         for k in range(N):
#             v = func.derivative2(Y[k] - u)
#             S += v
#             grad[k] = v
                
#         for k in prange(N, nogil=True, num_threads=num_procs):
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
