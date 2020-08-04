# coding: utf-8

# cython: language_level=3
# cython: boundscheck=True
# cython: wraparound=True
# cython: nonecheck=True
# cython: embedsignature=True
# cython: initializedcheck=True

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

from libc.math cimport fabs, pow, sqrt, fmax, log, exp
    
import numpy as np

cdef double double_max = PyFloat_GetMax()
cdef double double_min = PyFloat_GetMin()
cdef Func square_func = Square()

cdef class KAverage:
    
    def __init__(self, q, Func func = square_func, tol=1.0e-6, n_iter=1000):
        self.func = func
        self.q = q
        self.u = np.zeros(q, 'd')
        self.tol = tol
        self.n_iter = n_iter
    #
    cdef _evaluate_classes(self, double[::1] Y, int[::1] J):
        cdef double d, d_min, yk
        cdef double[::1] u = self.u
        cdef Py_ssize_t j, j_min, k, N = Y.shape[0]
        cdef Py_ssize_t q = self.q
        
        for k in range(N):
            yk = Y[k]
            
            d_min = max_double
            j_min = 0
            for j in range(q):
                d = fabs(yk - u[j])
                if d < d_min:
                    j_min = j
                    d_min = d
                    
            J[k] = j_min
    #
    def evaluate_classes(self, double[::1] Y):
        cdef int[::1] J = np.zeros(Y.shape[0], 'i')
        self._evaluate_classes(Y, J)
        return np.asarray(J)
    #
    cdef _evaluate_distances(self, double[::1] Y, double[::1] D):
        cdef double d, d_min, yk
        cdef Py_ssize_t j, j_min, k, N = Y.shape[0]
        cdef Py_ssize_t q = self.q
        cdef double[::1] u = self.u
        
        for k in range(N):
            yk = Y[k]
            
            d_min = max_double
            j_min = 0
            for j in range(q):
                d = fabs(yk - u[j])
                if d < d_min:
                    j_min = j
                    d_min = d
                    
            D[k] = d_min

        return D
    #
    def evaluate_distances(self, double[::1] Y):
        cdef double[::1] D = np.zeros(Y.shape[0], 'd')
        self._evaluate_distances(Y, D)
        return np.asarray(D)
    #
    cdef _init_u(self, double[::1] Y):
        cdef double y, yk, y_min, y_max, dy
        cdef Py_ssize_t j, k, N=Y.shape[0]
        
        y_min = double_max
        y_max = double_min
        for k in range(N):
            yk = Y[k]
            if yk < y_min:
                y_min = yk
            if yk > y_max:
                y_max = yk

        dy = (y_max - y_min) / self.q
        y = y_min
        for j in range(self.q):
            self.u[j] = y + 0.5*dy
            y += dy
    #
    cdef _fit(self, double[::1] Y):
        cdef Py_ssize_t j, N = Y.shape[0]
        cdef double y, yk, wj
        cdef int[::1] J = np.zeros(N, 'i')
#         cdef double[::1] D = np.zeros(N, 'd')
        cdef double[::1] vv = np.zeros(self.q, 'd')
        cdef double[::1] W = np.zeros(self.q, 'd')
        cdef double[::1] u = self.u
        cdef bint flag
        
#         if self.u is None or self.u.shape[0] != self.q:
#             self.u = np.zeros(self.q, 'd')            
        
        self._init_u(Y)
#         self._evaluate_classes(Y, J)
        
        self.K = 1
        while self.K <= self.n_iter:
            print(self.u.base)
            for j in range(self.q):
                vv[j] = 0
                W[j] = 0

            self._evaluate_classes(Y, J)
            print(J.base)
            
            for k in range(N):
                j = J[k]
                yk = Y[k]
                wj = self.func.derivative_div_x(yk - u[j])
                W[j] += wj
                vv[j] += wj * yk
            
            for j in range(self.q):
                vv[j] /= W[j]
            
            flag = 1
            for j in range(self.q):
                if fabs(u[j] - vv[j]) > self.tol:
                    flag = 0
                    break
            
            for j in range(self.q):
                u[j] = vv[j]
                    
            if flag:
                break
                
            self.K += 1
                
    def fit(self, Y):
        cdef double[::1] YY = np.ascontiguousarray(Y, 'd')
        self._fit(YY)

# cdef class ScalarKMeans:
    
#     def __init__(self, q, tol=1.0e-6, n_iter=1000):
#         self.q = q
#         self.u = None
#         self.tol = tol
#         self.n_iter = n_iter
#     #
#     cdef _evaluate_classes(self, double[::1] Y, int[::1] J):
#         cdef double d, d_min, yk
#         cdef double[::1] u = self.u
#         cdef Py_ssize_t j, j_min, k, N = Y.shape[0]
#         cdef Py_ssize_t q = self.q
        
#         for k in range(N):
#             yk = Y[k]
            
#             d_min = max_double
#             j_min = 0
#             for j in range(q):
#                 d = fabs(yk - u[j])
#                 if d < d_min:
#                     j_min = j
#                     d_min = d
                    
#             J[k] = j_min
#     #
#     def evaluate_classes(self, double[::1] Y):
#         cdef int[::1] J = np.zeros(Y.shape[0], 'i')
#         self._evaluate_classes(Y, J)
#         return np.asarray(J)
#     #
#     cdef _evaluate_distances(self, double[::1] Y, double[::1] D):
#         cdef double d, d_min, yk
#         cdef Py_ssize_t j, j_min, k, N = Y.shape[0]
#         cdef Py_ssize_t q = self.q
#         cdef double[::1] u = self.u
        
#         for k in range(N):
#             yk = Y[k]
            
#             d_min = max_double
#             j_min = 0
#             for j in range(q):
#                 d = fabs(yk - u[j])
#                 if d < d_min:
#                     j_min = j
#                     d_min = d
                    
#             D[k] = d_min
#     #
#     def evaluate_distances(self, double[::1] Y):
#         cdef double[::1] D = np.zeros(Y.shape[0], 'd')
#         self._evaluate_distances(Y, D)
#         return np.asarray(D)
#     #
#     cdef _fit(self, double[::1] Y):
#         cdef Py_ssize_t q = self.q
#         cdef Py_ssize_t j, N = Y.shape[0]
#         cdef double y, yk, y_min, y_max
#         cdef int[::1] J = np.zeros(N, 'i')
#         cdef double[::1] vv = np.zeros(q, 'd')
#         cdef int[::1] mm = np.zeros(q, 'i')
#         cdef double[::1] u = self.u
#         cdef bint flag
        
#         if self.u is None or self.u.shape[0] != q:
#             self.u = np.zeros(q, 'd')
            
#             y_min = double_max
#             y_max = double_min
#             for k in range(N):
#                 yk = Y[k]
#                 if yk < y_min:
#                     y_min = yk
#                 elif yk > y_max:
#                     y_max = yk
                    
#             dy = (y_max - y_min) / q
#             y = y_min
#             for j in range(q):
#                 self.u[j] = y + 0.5*dy
#                 y += dy
        
#         self.K = 1
#         while self.K <= self.n_iter:
#             for j in range(q):
#                 vv[j] = 0
#                 mm[j] = 0

#             self._evaluate_classes(Y, J)
            
#             for k in range(N):
#                 j = J[k]
#                 vv[j] += Y[k]
#                 mm[j] += 1
            
#             for j in range(q):
#                 vv[j] /= mm[j]
            
#             flag = 1
#             for j in range(q):
#                 if fabs(u[j] - vv[j]) > self.tol:
#                     flag = 0
#                     break
            
#             for j in range(q):
#                 u[j] = vv[j]
                    
#             if flag:
#                 break
                
#             self.K += 1
                
#         def fit(self, Y):
#             cdef double[::1] _Y = np.ascontiguousarray(Y, 'd')
#             self._fit(_Y)
            