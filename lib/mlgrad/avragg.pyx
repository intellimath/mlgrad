# coding: utf-8

# The MIT License (MIT)
#
# Copyright (c) <2015-2023> <Shibzukhov Zaur, szport at gmail dot com>
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
from mlgrad.func2 cimport SoftMin
from libc.math cimport fabs, pow, sqrt, fmax, log, exp

from cython.parallel cimport parallel, prange

cimport mlgrad.inventory as inventory

cdef int num_threads = inventory.get_num_threads()

cdef double max_double = PyFloat_GetMax()

import numpy as np

cdef class Penalty(object):
    #
    cdef double evaluate(self, double[::1] Y, const double u):
        return 0
    #
    cdef double derivative(self, double[::1] Y, const double u):
        return 0
    #
    cdef void gradient(self, double[::1] Y, const double u, double[::1] grad):
        pass
    #
    cdef double iterative_next(self, double[::1] Y, const double u):
        return 0

@cython.final
cdef class PenaltyAverage(Penalty):
    #
    def __init__(self, Func func):
        self.func = func
    #
    @cython.cdivision(True)
    @cython.final
    cdef double evaluate(self, double[::1] Y, const double u):
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double S
        cdef Func func = self.func

        S = 0
        for k in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        # for k in range(N):
            S += func._evaluate(Y[k] - u)

        return S / N
    #
    @cython.cdivision(True)
    @cython.final
    cdef double derivative(self, double[::1] Y, const double u):
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double S
        cdef Func func = self.func

        S = 0
        for k in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        # for k in range(N):
            S += func._derivative(Y[k] - u)

        return -S / N
    #
    @cython.cdivision(True)
    @cython.final
    cdef double iterative_next(self, double[::1] Y, const double u):
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double S, V, v, yk
        cdef Func func = self.func

        S = 0
        V = 0
        for k in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        # for k in range(N):
            yk = Y[k]
            v = func._derivative_div_x(yk - u)
            V += v
            S += v * yk

        return S / V
    #
    @cython.cdivision(True)
    @cython.final
    cdef void gradient(self, double[::1] Y, const double u, double[::1] grad):
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double v, S
        cdef Func func = self.func

        S = 0
        for k in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        # for k in range(N):
            grad[k] = v = func._derivative2(Y[k] - u)
            S += v
 
        for k in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        # for k in range(N):
            grad[k] /= S

@cython.final
cdef class PenaltyScale(Penalty):
    #
    def __init__(self, Func func):
        self.func = func
    #
    @cython.cdivision(True)
    @cython.final
    cdef double evaluate(self, double[::1] Y, const double s):
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef Func func = self.func
        cdef double S
        cdef double v

        S = 0
        # for k in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        for k in range(N):
            v = Y[k]
            S += func._evaluate(v / s)

        return S / N + log(s)
    #
    @cython.cdivision(True)
    @cython.final
    cdef double derivative(self, double[::1] Y, const double s):
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double S, v
        cdef Func func = self.func
        #
        S = 0
        # for k in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        for k in range(N):
            v = Y[k] / s
            S += func._derivative(v) * v
        #
        return (1 - (S / N)) / s
    #
    @cython.cdivision(True)
    @cython.final
    cdef double iterative_next(self, double[::1] Y, const double s):
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double S, y_k
        cdef Func func = self.func
        #
        S = 0
        # for k in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        for k in range(N):
            y_k = Y[k]
            S += func._derivative(y_k / s) * y_k
        #
        return S / N
    #
    @cython.cdivision(True)
    @cython.final
    cdef void gradient(self, double[::1] Y, const double s, double[::1] grad):
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double S, v
        cdef Func func = self.func
        #
        S = 0
        # for k in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        for k in range(N):
            v = Y[k] / s
            S += func._derivative2(v) * v * v
        S += N
        #
        # for k in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        for k in range(N):
            v = Y[k] / s
            grad[k] = func._derivative(v) / S

cdef class Average(object):
    #
    cdef double _evaluate(self, double[::1] Y):
        self.fit(Y)
        return self.u
    #
    def evaluate(self, double[::1] Y):
        self.fit(Y)
        return self.u
    #
    def gradient(self, double[::1] Y):
        grad = np.zeros(len(Y), 'd')
        self._gradient(Y, grad)
        return grad
    #
    def weights(self, double[::1] Y):
        grad = np.zeros(len(Y), 'd')
        self._gradient(Y, grad)
        return grad
    #
    cdef double init_u(self, double[::1] Y):
        return 0
    #
    cdef _gradient(self, double[::1] Y, double[::1] grad):
        pass
    #
    cdef _weights(self, double[::1] Y, double[::1] weights):
        self._gradient(Y, weights)
    #
    cpdef fit(self, double[::1] Y):
        pass

cdef class AverageIterative(Average):
    #
    cdef double init_u(self, double[::1] Y):
        return 0
    #
    cpdef fit(self, double[::1] Y):
        cdef Py_ssize_t k=0, n_iter = self.n_iter
        cdef Penalty penalty = self.penalty
        cdef double tol = self.tol
        cdef double u, u_min, pval, pval_min
        cdef bint finish = 0

        u = u_min = self.init_u(Y)
        pval_min = pval = penalty.evaluate(Y, u)
        #
        for k in range(n_iter):
            u = penalty.iterative_next(Y, u)
            pval = penalty.evaluate(Y, u)

            if fabs(pval - pval_min) / (1 + fabs(pval_min)) < self.tol:
                finish = 1

            if pval < pval_min:
                pval_min = pval
                u_min = u

            if finish:
                break
            #

        self.K = k + 1
        self.u = u_min
        self.pval = pval_min
        self.evaluated = 1
    #
    cdef _gradient(self, double[::1] Y, double[::1] grad):
        if not self.evaluated:
            self.fit(Y)
        self.penalty.gradient(Y, self.u, grad)
        self.evaluated = 0
    #

# include "avragg_it.pyx"
include "avragg_fg.pyx"

@cython.final
cdef class MAverage(AverageIterative):
    #
    def __init__(self, Func func, tol=1.0e-9, n_iter=1000):
        self.penalty = PenaltyAverage(func)
        self.n_iter = n_iter
        self.tol = tol
        self.evaluated = 0
    #
    cdef double init_u(self, double[::1] Y):
        cdef Py_ssize_t N = Y.shape[0]
        return (Y[0] + Y[N//2] + Y[N-1]) / 3
    #
    # @cython.cdivision(True)
    # cdef double evaluate_next_u(self, double[::1] Y, const double u):
    #     cdef Py_ssize_t k, N = Y.shape[0]
    #     cdef double yk, v, S = 0, V = 0
    #     cdef Func func = self.func

    #     with cython.nogil, parallel():
    #         for k in prange(N, schedule='static'):
    #         # for k in range(N):
    #             yk = Y[k]
    #             v = func._derivative_div_x(yk - u)
    #             S += v * yk
    #             V += v

    #     return S / V
    # #
    # @cython.cdivision(True)
    # cdef double evaluate_penalty(self, double[::1] Y, const double u):
    #     cdef Py_ssize_t k, N = Y.shape[0]
    #     cdef Func func = self.func
    #     cdef double Z = 0

    #     with cython.nogil, parallel():
    #         for k in prange(N, schedule='static'):
    #         # for k in range(N):
    #             Z += func._evaluate(Y[k] - u)

    #     return Z / N
    # #
    # @cython.cdivision(True)
    # cpdef fit(self, double[::1] Y):
    #     cdef Py_ssize_t N = Y.shape[0]
    #     cdef Py_ssize_t k = 0
    #     cdef double u, u_min, z, z_min
    #     cdef double tol = self.tol
    #     cdef bint finish = 0

    #     u_min = u = (Y[0] + Y[N//2] + Y[N-1]) / 3
    #     z_min = z = self.evaluate_penalty(Y, u)

    #     for k in range(self.n_iter):
    #         u = self.evaluate_next_u(Y, u)
    #         z = self.evaluate_penalty(Y, u)

    #         if fabs(z - z_min) / (1 + fabs(z_min)) < tol:
    #             finish = 1

    #         if z < z_min:
    #             z_min = z
    #             u_min = u

    #         if finish:
    #             break

    #     self.u = u_min
    #     self.evaluated = 1
    #     self.K = k + 1
    # #
    # @cython.cdivision(True)
    # cdef _gradient(self, double[::1] Y, double[::1] grad):
    #     cdef Py_ssize_t k, N = Y.shape[0]
    #     cdef double V, v, u
    #     cdef Func func = self.func

    #     # if not self.evaluated:
    #     self.fit(Y)
    #     u = self.u
        
    #     V = 0
    #     with cython.nogil, parallel():
    #         for k in prange(N, schedule='static'):
    #         # for k in range(N):
    #             v = grad[k] = func._derivative2(Y[k] - u)
    #             V += v

    #     with cython.nogil, parallel():
    #         for k in prange(N, schedule='static'):
    #         # for k in range(N):
    #             grad[k] /= V

    #     self.evaluated = 0
 
# @cython.final
cdef class SAverage(AverageIterative):
    #
    def __init__(self, Func func, tol=1.0e-9, n_iter=1000):
        self.penalty = PenaltyScale(func)
        self.n_iter = n_iter
        self.tol = tol
        self.evaluated = 0
    #
    cdef double init_u(self, double[::1] Y):
        return 1
    # #
    # @cython.cdivision(True)
    # cdef double evaluate_next_u(self, double[::1] Y, const double u):
    #     cdef Py_ssize_t k, N = Y.shape[0]
    #     cdef double yk, S = 0
    #     cdef Func func = self.func
    #     cdef double S = 0

    #     with cython.nogil, parallel():
    #         for k in prange(N, schedule='static'):
    #         # for k in range(N):
    #             yk = Y[k]
    #             S += func._derivative(yk / u) * yk

    #     return S / N
    # #
    # @cython.cdivision(True)
    # cdef double evaluate_penalty(self, double[::1] Y, const double u):
    #     cdef Py_ssize_t k, N = Y.shape[0]
    #     cdef Func func = self.func
    #     cdef double Z = 0

    #     with cython.nogil, parallel():
    #         for k in prange(N, schedule='static'):
    #         # for k in range(N):
    #             Z += func._evaluate(Y[k] / u)

    #     return Z / N + log(u
    # #
    # @cython.cdivision(True)
    # cpdef fit(self, double[::1] Y):
    #     cdef Py_ssize_t k, N = Y.shape[0]
    #     cdef double u, u_min, z, z_min
    #     cdef double tol = self.tol
    #     cdef Func func = self.func
    #     cdef Py_ssize_t k
    #     cdef bint finish = 0

    #     # if not self.evaluated:
    #     #     u = 1.0
    #     # else:
    #     #     u = self.u

    #     u = u_min = 1.0
    #     z_min = z = self.evaluate_penalty(Y, u)

    #     for k in range(self.n_iter):
    #         u = self.evaluate_next_u(Y, u)
    #         z = self.evaluate_penalty(Y, u)

    #         if K > 0 and fabs(z - z_min) / (1 + fabs(z_min)) < tol:
    #             finish = 1

    #         if z < z_min:
    #             z_min = z
    #             u_min = u

    #         if finish:
    #             break

    #     self.u = u_min
    #     self.evaluated = 1
    #     self.K = k + 1
    # #
    # @cython.cdivision(True)
    # cdef _gradient(self, double[::1] Y, double[::1] grad):
    #     cdef Py_ssize_t k, N = Y.shape[0]
    #     cdef double S, v, u
    #     cdef Func func = self.func

    #     if not self.evaluated:
    #         self.fit(Y)
    #     u = self.u

    #     S = 0
    #     # for k in prange(N, nogil=True, schedule='static', num_threads=num_threads):
    #     for k in range(N):
    #         v = Y[k] / u
    #         S += func._derivative(v) * v * v
    #     S += N
    #     #
    #     # for k in prange(N, nogil=True, schedule='static', num_threads=num_threads):
    #     for k in range(N):
    #         grad[k] = func._derivative(Y[k] / u) / S

    #     self.evaluated = 0
        
@cython.final
cdef class ParameterizedAverage(Average):
    #
    def __init__(self, ParameterizedFunc func, Average avr):
        self.func = func
        self.avr = avr
        self.evaluated = 0
    #
    @cython.final
    @cython.cdivision(True)
    cpdef fit(self, double[::1] Y):
        cdef Py_ssize_t k
        cdef Py_ssize_t N = Y.shape[0], M
        cdef double c
        cdef double S = 0
        # cdef double *YY = &Y[0]
        cdef ParameterizedFunc func = self.func

        self.avr.fit(Y)
        c = self.avr.u

        # for k in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        for k in range(N):
            S += func._evaluate(Y[k], c)

        self.u = S / N
        self.evaluated = 1
    #
    @cython.cdivision(True)
    @cython.final
    cdef _gradient(self, double[::1] Y, double[::1] grad):
        cdef Py_ssize_t k
        cdef Py_ssize_t N = Y.shape[0], M
        cdef double c, v
        cdef double N1 = 1.0/N
        cdef double H, S
        # cdef double *YY = &Y[0]
        # cdef double *GG
        cdef ParameterizedFunc func = self.func

        self.avr._gradient(Y, grad)
        self.evaluated = 0
        c = self.avr.u

        H = 0
        # for k in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        for k in range(N):
            H += func.derivative_u(Y[k], c)
        H *= N1

        S = 0
        # GG = &grad[0]
        # for k in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        for k in range(N):
            v = N1 * func._derivative(Y[k], c) +  H * grad[k]
            grad[k] = v
            S += v

        v = S
        # for k in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        for k in range(N):
            grad[k] /= v
    #

@cython.final
cdef class WMAverage(Average):
    #
    def __init__(self, Average avr):
        self.avr = avr
        self.u = 0
        self.evaluated = 0
    #
    @cython.cdivision(True)
    @cython.final
    cpdef fit(self, double[::1] Y):
        cdef double v, yk, avr_u
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double S

        avr_u = self.avr._evaluate(Y)

        S = 0
        for k in prange(N, schedule='static', nogil=True, num_threads=num_threads):
        # for k in range(N):
            yk = Y[k]
            v = yk if yk <= avr_u else avr_u
            S += v
        self.u = S / N

        # self.u_min = self.u
        self.K = self.avr.K
        self.evaluated = 1
    #
    @cython.cdivision(True)
    @cython.final
    cdef _gradient(self, double[::1] Y, double[::1] grad):
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double u, v, m, fN = N

        if self.evaluated == 0:
            self.avr.fit(Y)
        self.avr._gradient(Y, grad)
        self.evaluated = 0
        u = self.avr.u

        m = 0
        for k in range(N):
            if Y[k] > u:
                m += 1

        for k in prange(N, schedule='static', nogil=True, num_threads=num_threads):
        # for k in range(N):
            v = m * grad[k]
            if Y[k] <= u:
                v = v + 1
            grad[k] = v / fN
    #

# cdef class WMAverageMixed(Average):
#     #
#     def __init__(self, Average avr, double gamma=1):
#         self.avr = avr
#         self.gamma = gamma
#         self.u = 0
#         self.evaluated = 0
#     #
#     cpdef fit(self, double[::1] Y):
#         cdef double u, v, yk, avr_u
#         cdef Py_ssize_t k, m, N = Y.shape[0]

#         self.avr.fit(Y)
#         self.evaluated = 1
#         avr_u = self.avr.u

#         m = 0
#         for k in range(N):
#             if Y[k] > avr_u:
#                 m += 1

#         u = 0
#         v = 0
#         # for k in prange(N, nogil=True, schedule='static', num_threads=num_threads):
#         for k in range(N):
#             yk = Y[k]
#             if yk <= avr_u:
#                 u += yk
#             else:
#                 v += yk

#         self.u = (1-self.gamma) * u / (N-m) + self.gamma * v / m

#         self.u_min = self.u
#     #
#     cdef _gradient(self, double[::1] Y, double[::1] grad):
#         cdef Py_ssize_t k, m, N = Y.shape[0]
#         cdef double v, N1, N2, yk, avr_u

#         self.avr._gradient(Y, grad)
#         self.evaluated = 0
#         avr_u = self.avr.u

#         m = 0
#         for k in range(N):
#             if Y[k] > avr_u:
#                 m += 1

#         N1 = (1-self.gamma) / (N-m)
#         N2 = self.gamma / m
#         # for k in prange(N, nogil=True, schedule='static', num_threads=num_threads):
#         for k in range(N):
#             yk = Y[k]
#             if yk <= avr_u:
#                 v = N1
#             else:
#                 v = N2
#             grad[k] = v
#     #

cdef class TMAverage(Average):
    #
    def __init__(self, Average avr):
        self.avr = avr
        self.u = 0
        self.evaluated = 0
    #
    cpdef fit(self, double[::1] Y):
        cdef double u, v, yk, avr_u
        cdef Py_ssize_t k, M, N = Y.shape[0]

        self.avr.fit(Y)
        self.evaluated = 1
        u = 0
        M = 0
        avr_u = self.avr.u
        # for k in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        for k in range(N):
            yk = Y[k]
            if yk <= avr_u:
                u += yk
                M += 1

        self.u = u / M
        self.u_min = self.u
        self.K = self.avr.K
    #
    cdef _gradient(self, double[::1] Y, double[::1] grad):
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double u, N1, M

        self.avr._gradient(Y, grad)
        self.evaluated = 0
        u = self.avr.u

        M = 0
        for k in range(N):
            if Y[k] <= u:
                M += 1

        N1 = 1./M
        # for k in prange(N, nogil=True, schedule='static', num_threads=num_threads):
        for k in range(N):
            if Y[k] <= u:
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
        self.evaluated = 0
    #
    @cython.cdivision(True)
    cpdef fit(self, double[::1] Y):
        cdef double v, w, yk, avr_z
        cdef double u, u_prev
        cdef Py_ssize_t k, m, N = Y.shape[0]
        cdef double q, S
        cdef double[::1] Z
        cdef double[::1] grad = np.zeros(N, 'd')
        cdef Average wm = self.avr

        if self.Z is None:
            self.Z = np.zeros(N, 'd')
        Z = self.Z

        wm.fit(Y)
        u = wm.u

        self.K = 1
        while self.K < self.n_iter:
            u_prev = u
            for k in range(N):
                w = Y[k] - u
                Z[k] = w * w

            wm.fit(Z)
            avr_z = sqrt(self.avr.u)
            wm._gradient(Z, grad)

            m = 0
            for k in range(N):
                if fabs(Y[k] - u) > avr_z:
                    m += 1

            v = 0
            # for k in prange(N, nogil=True, schedule='static', num_threads=num_threads):
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
        self.u_min = self.u
        self.evaluated = 1
    #
    @cython.cdivision(True)
    cdef _gradient(self, double[::1] Y, double[::1] grad):
        cdef Py_ssize_t k, m, N = Y.shape[0]
        cdef double u, v, w, N1, yk
        cdef double q, avr_z, S
        cdef double[::1] Z = self.Z

        u = self.u
        for k in range(N):
            w = Y[k] - u
            Z[k] = w * w

        self.avr.fit(Z)
        avr_z = sqrt(self.avr.u)
        self.avr._gradient(Z, grad)
        self.evaluated = 0

        m = 0
        for k in range(N):
            if fabs(Y[k] - u) > avr_z:
                m += 1

        N1 = 1./ N
        # for k in prange(N, nogil=True, schedule='static', num_threads=num_threads):
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
    cpdef fit(self, double[::1] Y):
        cdef double S
        cdef Py_ssize_t k, N = Y.shape[0]
        # cdef double *YY =&Y[0]

        S = 0
        for k in prange(N, schedule='static', nogil=True, num_threads=num_threads):
        # for k in range(N):
                S += Y[k]
        self.u = S / N
        self.u_min = self.u
        self.evaluated = 1
    #
    @cython.cdivision(True)
    cdef _gradient(self, double[::1] Y, double[::1] grad):
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double v

        v = 1./N
        for k in prange(N, schedule='static', nogil=True, num_threads=num_threads):
        # for k in range(N):
                grad[k] = v
        self.evaluated = 0

cdef class RArithMean(Average):
    #
    def __init__(self, Func func):
        self.func = func
    #
    @cython.cdivision(True)
    cpdef fit(self, double[::1] Y):
        cdef double S
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef Func func = self.func

        S = 0
        for k in prange(N, schedule='static', nogil=True, num_threads=num_threads):
        # for k in range(N):
                S += func._evaluate(Y[k])
        self.u = S / N
        self.u_min = self.u
        self.evaluated = 1
    #
    @cython.cdivision(True)
    cdef _gradient(self, double[::1] Y, double[::1] grad):
        cdef Py_ssize_t k, N = Y.shape[0]
        # cdef double *GG = &grad[0]
        cdef double v
        cdef Func func = self.func

        v = 1./N
        for k in prange(N, schedule='static', nogil=True, num_threads=num_threads):
        # for k in range(N):
                grad[k] = v * func._derivative(Y[k])
        self.evaluated = 0
    #
    @cython.cdivision(True)
    cdef _weights(self, double[::1] Y, double[::1] grad):
        cdef Py_ssize_t k, N = Y.shape[0]
        # cdef double *GG = &grad[0]
        cdef double v, S
        cdef Func func = self.func

        S = 0
        for k in prange(N, schedule='static', nogil=True, num_threads=num_threads):
        # for k in range(N):
                v = func._derivative_div_x(Y[k])
                grad[k] = v
                S += v

        # with cython.nogil, parallel():
        #     for k in prange(N, schedule='static'):
        for k in range(N):
                grad[k] /= S

        self.evaluated = 0

cdef class Minimal(Average):
    #
    cpdef fit(self, double[::1] Y):
        cdef double yk, y_min = Y[0]
        cdef Py_ssize_t k, N = Y.shape[0]
        # cdef double *y = &Y[0]

        for k in range(N):
            yk = Y[k]
            if yk < y_min:
                y_min = yk
        self.u = y_min
        self.u_min = self.u
        self.evaluated = 1
    #
    cdef _gradient(self, double[::1] Y, double[::1] grad):
        cdef Py_ssize_t k, m, N = Y.shape[0]
        cdef double u = self.u
        # cdef double *g = &grad[0]
        # cdef double *y = &Y[0]

        m = 0
        for k in range(N):
            if Y[k] == u:
                grad[k] = 1
                m += 1
            else:
                grad[k] = 0

        if m > 1:
            for k in range(N):
                if grad[k] > 0:
                    grad[k] /= m

        self.evaluated = 0

cdef class Maximal(Average):
    #
    cpdef fit(self, double[::1] Y):
        cdef double yk, y_max = Y[0]
        cdef Py_ssize_t k, N = Y.shape[0]
        for k in range(N):
            yk = Y[k]
            if yk > y_max:
                y_max = yk
        self.u = y_max
        self.u_min = self.u
        self.evaluated = 1
    #
    cdef _gradient(self, double[::1] Y, double[::1] grad):
        cdef Py_ssize_t k, m, N = Y.shape[0]

        m = 0
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

        self.evaluated = 0

cdef class KolmogorovMean(Average):
    #
    def __init__(self, Func func, Func invfunc):
        self.func = func
        self.invfunc = invfunc
    #
    cpdef fit(self, double[::1] Y):
        cdef double u, yk
        cdef Py_ssize_t k, N = Y.shape[0]

        u = 0
#         for k in prange(N, nogil=True):
        for k in range(N):
            yk = Y[k]
            u += self.func._evaluate(yk)
        u /= N
        self.uu = u
        self.u = self.invfunc._evaluate(u)
        self.u_min = self.u
        self.evaluated = 1
    #
    cdef _gradient(self, double[::1] Y, double[::1] grad):
        cdef Py_ssize_t k, N = Y.shape[0]
        cdef double V

        V = self.invfunc._derivative(self.uu)
#         for k in prange(N, nogil=True, schedule='static'):
        for k in range(N):
            grad[k] = self.func._derivative(Y[k]) * V
        self.evaluated = 0

cdef class SoftMinimal(Average):
    #
    def __init__(self, a):
        self.softmin = SoftMin(a)
        
    #
    cpdef fit(self, double[::1] Y):
#         cdef double u, yk
#         cdef Py_ssize_t k, N = Y.shape[0]
#         cdef double a = self.a

#         u = 0
# #         for k in prange(N, nogil=True):
#         for k in range(N):
#             yk = Y[k]
#             u += exp(-yk*a)
#         u /= N
        self.u = self.softmin._evaluate(Y)
        self.u_min = self.u
        self.evaluated = 1
    #
    cdef _gradient(self, double[::1] Y, double[::1] grad):
#         cdef Py_ssize_t k, l, N = Y.shape[0]
#         cdef double u, yk, yl
#         cdef double a = self.a

#         for l in range(N):
#             u = 0
#             yl = Y[l]
#             for k in range(N):
#                 yk = Y[k] - yl
#                 u += exp(-yk*a)

#             grad[l] = 1. / u
        self.softmin._gradient(Y, grad)
        self.evaluated = 0

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

