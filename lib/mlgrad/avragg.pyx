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

from cython.parallel cimport parallel, prange
 
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
        cdef Py_ssize_t k=0, M, N = Y.shape[0]
        cdef double S
        cdef double y1, y2, y3, y4
        cdef double v1, v2, v3, v4
    
        S = 0

        M = 4 * (N // 4)
        if M > 0:
            for k in prange(0, M, 4, nogil=True, num_threads=num_procs):
    #         for k in range(0, M, 4):
                y1 = Y[k]
                y2 = Y[k+1]
                y3 = Y[k+2]
                y4 = Y[k+3]

                v1 = self.func.evaluate(y1-u)
                v2 = self.func.evaluate(y2-u)
                v3 = self.func.evaluate(y3-u)
                v4 = self.func.evaluate(y4-u)

                S += v1 + v2 + v3 + v4
            
        k += 4
        while k < N:
            S += self.func.evaluate(Y[k] - u)
            k += 1
       
        return S / N
    #
    cdef double derivative(self, double[::1] Y, double u):
        cdef Py_ssize_t k, M, N = Y.shape[0]
        cdef double S
        cdef double y1, y2, y3, y4
        cdef double v1, v2, v3, v4
        
        S = 0

        M = 4 * (N // 4)
        if M > 0:
            for k in prange(0, M, 4, nogil=True, num_threads=num_procs):
#             for k in range(0, M, 4):
                y1 = Y[k]
                y2 = Y[k+1]
                y3 = Y[k+2]
                y4 = Y[k+3]

                v1 = self.func.derivative(y1 - u)
                v2 = self.func.derivative(y2 - u)                       
                v3 = self.func.derivative(y3 - u)
                v4 = self.func.derivative(y4 - u)

                S += v1 + v2 + v3 + v4

        k += 4
        while k < N:
            S += self.func.derivative(Y[k] - u)                        
            k += 1
    
        return -S / N
    #
    cdef double iterative_next(self, double[::1] Y, double u):
        cdef Py_ssize_t k, M, N = Y.shape[0]
        cdef double S, V, v, yk
        cdef double y1, y2, y3, y4
        cdef double v1, v2, v3, v4
    
        S = 0
        V = 0
        
        M = 4 * (N // 4)
        if M > 0:
            for k in prange(0, M, 4, nogil=True, num_threads=num_procs):
#             for k in range(0, M, 4):
                y1 = Y[k]
                y2 = Y[k+1]
                y3 = Y[k+2]
                y4 = Y[k+3]

                v1 = self.func.derivative_div_x(y1 - u)
                v2 = self.func.derivative_div_x(y2 - u)
                v3 = self.func.derivative_div_x(y3 - u)
                v4 = self.func.derivative_div_x(y4 - u)

                V += v1 + v2 + v3 + v4
                S += v1*y1 + v2*y2 + v3*y3 + v4*y4
            
        k += 4
        while k < N:
            yk = Y[k]
            v = self.func.derivative_div_x(yk - u)
            V += v
            S += v * yk
            k += 1
        
        return S / V
    #
    cdef void gradient(self, double[::1] Y, double u, double[::1] grad):
        cdef Py_ssize_t k, M, N = Y.shape[0]
        cdef double v, S
        cdef double y1, y2, y3, y4
        cdef double v1, v2, v3, v4
        
        S = 0

        M = 4 * (N // 4)
        if M > 0:
            for k in prange(0, M, 4, nogil=True, num_threads=num_procs):
#             for k in range(0, M, 4):
                y1 = Y[k]
                y2 = Y[k+1]
                y3 = Y[k+2]
                y4 = Y[k+3]

                v1 = self.func.derivative2(y1 - u)
                v2 = self.func.derivative2(y2 - u)
                v3 = self.func.derivative2(y3 - u)
                v4 = self.func.derivative2(y4 - u)

                grad[k]   = v1
                grad[k+1] = v2
                grad[k+2] = v3
                grad[k+3] = v4

                S += v1 + v2 + v3 + v4
            
        k += 4
        if k < N:
            while k < N: 
                v = self.func.derivative2(Y[k] - u)
                S += v
                grad[k] = v
                k += 1            
                

        if M > 0:
            for k in prange(0, M, 4, nogil=True, num_threads=num_procs):
#             for k in range(0, M, 4):
                grad[k] /= S
                grad[k+1] /= S
                grad[k+2] /= S
                grad[k+3] /= S
    
        k += 4
        while k < N:
            grad[k] /= S
            k += 1

cdef class PenaltyScale(Penalty):
    #
    def __init__(self, Func func):
        self.func = func
    #
    cdef double evaluate(self, double[::1] Y, double s):
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef Func func = self.func
        cdef double S
        cdef double v1, v2, v3, v4

        S = 0
        for k in prange(0, N, 4, nogil=True, num_threads=num_procs):
#         for k in range(N):
            v1 = Y[k]
            v2 = Y[k+1]
            v3 = Y[k+2]
            v4 = Y[k+3]
    
            S += func.evaluate(v1 / s)
            S += func.evaluate(v2 / s)
            S += func.evaluate(v3 / s)
            S += func.evaluate(v4 / s)
        
        while k < N:
            S += func.evaluate(Y[k] / s)
            k += 1
    
        return S / N + log(s)
    #
    cdef double derivative(self, double[::1] Y, double s):
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double S, y_k, v
        
        S = 0
        for k in prange(N, nogil=True, num_threads=num_procs):
#         for k in range(N):
            v = Y[k] / s
            S += self.func.derivative(v) * v
            
        return (1 - (S / N)) / s

    cdef double iterative_next(self, double[::1] Y, double s):
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double S, v, y_k
        
        S = 0
        for k in prange(N, nogil=True, num_threads=num_procs):
#         for k in range(N):
            y_k = Y[k]
            S += self.func.derivative_div_x(y_k / s) * y_k * y_k
        
        return sqrt(S / N)
    #
    cdef void gradient(self, double[::1] Y, double s, double[::1] grad):
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double S, v
        
        S = 0
        for k in prange(N, nogil=True, num_threads=num_procs):
#         for k in range(N):
            v = Y[k] / s
            S += self.func.derivative2(v) * v * v
        S += N
            
        for k in prange(N, nogil=True, num_threads=num_procs):
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
    def nabla(self, double[::1] Y):
        grad = np.zeros(len(Y), 'd')
        self.gradient(Y, grad)
        return grad
    #
    cpdef fit(self, double[::1] Y, u0=None):
        cdef int j, K, n_iter = self.n_iter
        cdef Penalty penalty = self.penalty
        
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
        cdef double p, pmin, prev
        #
        pval = self.pval
        pmin = self.pmin
        prev = self.pval_prev
        
        if pval < pmin:
            self.pmin = pval
            self.u_best = self.u
            self.m = 0
#         elif pval == pmin:
#             if self.u < self.u_best:
#                 self.u_best = self.u
        #
        if fabs(pval - prev) / (1 + fabs(pmin)) < self.tol:
            return 1
            
        if self.m > self.m_iter:
            return 1

        self.m += 1
                
        return 0

    
cdef class ParameterizedAverage(Average):
    #
    def __init__(self, ParameterizedFunc func, Average avr):
        self.func = func
        self.avr = avr
    #
    cpdef fit(self, double[::1] Y, u0=None):
        cdef Py_ssize_t k
        cdef Py_ssize_t N = Y.shape[0], M
        cdef double c
        cdef double u = 0
        cdef double u1,u2,u3,u4
        cdef double v1,v2,v3,v4
        
        self.avr.fit(Y)
        c = self.avr.u 

        M = 4 * (N // 4)
        if M > 0:
            for k in prange(0, M, 4, nogil=True, num_threads=num_procs):
    #         for k in range(N):
                u1 = Y[k]
                u2 = Y[k+1]
                u3 = Y[k+2]
                u4 = Y[k+3]

                v1 = self.func.evaluate(u1, c)
                v2 = self.func.evaluate(u2, c)
                v3 = self.func.evaluate(u3, c)
                v4 = self.func.evaluate(u4, c)

                u += v1 + v2 + v3 + v4
    
        k += 4
        while k < N:
            u += self.func.evaluate(Y[k], c)
            k += 1

        self.u = u / N
    #
    cdef gradient(self, double[::1] Y, double[::1] grad):
        cdef Py_ssize_t k
        cdef Py_ssize_t N = Y.shape[0], M
        cdef double c
        cdef double N1 = 1/N
        cdef double H
        cdef double u1,u2,u3,u4
        cdef double v1,v2,v3,v4
        cdef double w1,w2,w3,w4
        
        self.avr.gradient(Y, grad)
        c = self.avr.u
        
        H = 0
        M = 4 * (N // 4)
        if M > 0:
            for k in prange(0, M, 4, nogil=True, num_threads=num_procs):
#             for k in range(N):
                u1 = Y[k]
                u2 = Y[k+1]
                u3 = Y[k+2]
                u4 = Y[k+3]

                v1 = self.func.derivative_u(u1, c)
                v2 = self.func.derivative_u(u2, c)
                v3 = self.func.derivative_u(u3, c)
                v4 = self.func.derivative_u(u4, c)
                H += v1 + v2 + v3 + v4
        
        k += 4
        while k < N:
            H += self.func.derivative_u(Y[k], c)
            k += 1

        M = 4 * (N // 4)
        if M > 0:
            for k in prange(0, M, 4, nogil=True, num_threads=num_procs):
#             for k in range(0, M, 4):
                u1 = Y[k]
                u2 = Y[k+1]
                u3 = Y[k+2]
                u4 = Y[k+3]

                v1 = grad[k]
                v2 = grad[k+1]
                v3 = grad[k+2]
                v4 = grad[k+3]

                w1 = self.func.derivative(u1, c) +  H * v1
                w2 = self.func.derivative(u2, c) +  H * v2
                w3 = self.func.derivative(u3, c) +  H * v3
                w4 = self.func.derivative(u4, c) +  H * v4

                grad[k]   = N1 * w1
                grad[k+1] = N1 * w2
                grad[k+2] = N1 * w3
                grad[k+3] = N1 * w4
    
        k += 4
        while k < N:
            grad[k] = N1 * (self.func.derivative(Y[k], c) +  H * grad[k])
            k += 1
    
    #

include "avragg_it.pyx"
include "avragg_fg.pyx"

cdef class WMAverage(Average):
    #
    def __init__(self, Average avr):
        self.avr = avr
        self.gamma = 1
        self.u = 0
    #
    cpdef fit(self, double[::1] Y, u0=None):
        cdef double u, v, yk, avr_u
        cdef Py_ssize_t k, m, N = Y.shape[0], M
        cdef double y1,y2,y3,y4
        cdef double v1,v2,v3,v4
        cdef double s1,s2,s3,s4
        
        self.avr.fit(Y)
        avr_u = self.avr.u

        u = 0
        M = 4 * (N // 4)
        if M > 0:
            for k in prange(0, M, 4, nogil=True, num_threads=num_procs):
#             for k in range(0, M, 4):
                y1 = Y[k]
                y2 = Y[k+1]
                y3 = Y[k+2]
                y4 = Y[k+3]

                v1 = avr_u if y1 > avr_u else y1
                v2 = avr_u if y2 > avr_u else y2
                v3 = avr_u if y3 > avr_u else y3
                v4 = avr_u if y4 > avr_u else y4

                u += v1 + v2 + v3 + v4

        k += 4
        while k < N:
            yk = Y[k]
            v = avr_u if yk > avr_u else yk
            u += v
            k += 1

        self.u = u / N
        self.u_best = self.u
        self.K = self.avr.K
    #
    cdef gradient(self, double[::1] Y, double[::1] grad):
        cdef Py_ssize_t k, m, m4, N = Y.shape[0], M
        cdef double u, v, yk
        cdef double y1,y2,y3,y4
        cdef double u1,u2,u3,u4
        cdef double v1,v2,v3,v4
        cdef double s1,s2,s3,s4
        cdef double N1 = 1/N

        self.avr.gradient(Y, grad)
        u = self.avr.u

        m = 0
        for k in range(N):
            if Y[k] > u:
                m += 1

        M = 4 * (N // 4)
        if M > 0:
            for k in prange(0, M, 4, nogil=True, num_threads=num_procs):
#             for k in range(0, M, 4):
                y1 = Y[k]
                y2 = Y[k+1]
                y3 = Y[k+2]
                y4 = Y[k+3]

                u1 = grad[k]
                u2 = grad[k+1]
                u3 = grad[k+2]
                u4 = grad[k+3]

                v1 = (1 + m * u1) if y1 <= u else (m * u1)
                v2 = (1 + m * u2) if y2 <= u else (m * u2)
                v3 = (1 + m * u3) if y3 <= u else (m * u3)
                v4 = (1 + m * u4) if y4 <= u else (m * u4)

                grad[k]   = v1 * N1
                grad[k+1] = v2 * N1
                grad[k+2] = v3 * N1
                grad[k+3] = v4 * N1
            
        k += 4
        while k < N:
            yk = Y[k]
            u = grad[k]
            v = (1 + m * u) if yk <= u else (m * u)
            grad[k] = v * N1
            k += 1
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
    cpdef fit(self, double[::1] Y, u0=None):
        cdef double u
        cdef Py_ssize_t k, N = Y.shape[0]
        
        u = 0
#         for k in prange(N, nogil=True, num_threads=num_procs):
        for k in range(N):
            u += Y[k]
        self.u = u / N
        self.u_best = self.u
    #
    cdef gradient(self, double[::1] Y, double[::1] grad):
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double N1
                 
        N1 = 1./N
#         for k in prange(N, nogil=True, num_threads=num_procs):
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
        
        u = 0
#         for k in prange(N, nogil=True):
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
#             for k in prange(N, nogil=True):
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
