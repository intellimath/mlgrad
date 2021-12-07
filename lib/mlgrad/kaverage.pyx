# coding: utf-8

# cython: language_level=3
# cython: boundscheck=True
# cython: wraparound=True
# cython: nonecheck=True
# cython: embedsignature=True
# cython: initializedcheck=True

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

from libc.math cimport fabs, pow, sqrt, fmax, log, exp

cimport cython
from cython.parallel cimport parallel, prange

from openmp cimport omp_get_num_procs, omp_get_thread_num

cdef int num_procs = omp_get_num_procs()
if num_procs >= 4:
    num_procs /= 2
else:
    num_procs = 2
    
import numpy as np

cdef float float_max = PyFloat_GetMax()
cdef float float_min = PyFloat_GetMin()
cdef Func square_func = Square()

cdef float max_l1_distance(float *a, float *b, Py_ssize_t n):
    cdef float val, max_val = 0
    cdef Py_ssize_t i
    
    for i in range(n):
        val = fabs(a[i] - b[i])
        if val > max_val:
            max_val = val
    return max_val

cdef float maxabs_array(float *a, Py_ssize_t n):
    cdef float val, max_val = 0
    cdef Py_ssize_t i
    
    for i in range(n):
        val = fabs(a[i])
        if val > max_val:
            max_val = val
    return max_val

cdef class KAverage:
    
    def __init__(self, q, Func func = square_func, tol=1.0e-6, n_iter=1000):
        self.func = func
        self.q = q
        self.u = np.zeros(q, 'f')
        self.u_prev = np.zeros(q, 'f')
        self.u_min = np.zeros(q, 'f')
        self.tol = tol
        self.n_iter = n_iter
    #
    cdef _evaluate_classes(self, float[::1] Y, Py_ssize_t[::1] J):
        cdef float d, d_min, y_k
        cdef float[::1] u = self.u
        cdef Py_ssize_t j, j_min, k, N = Y.shape[0]
        cdef Py_ssize_t q = self.q
        
        for k in prange(N, nogil=True, num_threads=num_procs):
#         for k in range(N):
            y_k = Y[k]
            
            d_min = float_max
            j_min = 0
            for j in range(q):
                d = fabs(y_k - u[j])
                if d < d_min:
                    j_min = j
                    d_min = d
                    
            J[k] = j_min
    #
    def evaluate_classes(self, float[::1] Y):
        cdef Py_ssize_t[::1] J = np.zeros(Y.shape[0], 'l')
        self._evaluate_classes(Y, J)
        return np.asarray(J)
    #
    cdef _evaluate_distances(self, float[::1] Y, float[::1] D):
        cdef float d, d_min, y_k
        cdef Py_ssize_t j, k, N = Y.shape[0]
        cdef Py_ssize_t q = self.q
        cdef float[::1] u = self.u
        
        for k in prange(N, nogil=True, num_threads=num_procs):
#         for k in range(N):
            y_k = Y[k]
            
            d_min = float_max
#             j_min = 0
            for j in range(q):
                d = fabs(y_k - u[j])
                if d < d_min:
#                     j_min = j
                    d_min = d
                    
            D[k] = d_min

        return D
    #
    cdef float _evaluate_qval(self, float[::1] Y):
        cdef float d, d_min, y_k
        cdef Py_ssize_t j, j_min, k, N = Y.shape[0]
        cdef Py_ssize_t q = self.q
        cdef float[::1] u = self.u
        cdef float s
        
        s = 0
        for k in prange(N, nogil=True, num_threads=num_procs):
#         for k in range(N):
            y_k = Y[k]
            
            d_min = float_max
            for j in range(q):
                d = self.func.evaluate(y_k - u[j])
                if d < d_min:
                    d_min = d
                    
            s += d_min

        return s / N
    #    
    def evaluate_distances(self, float[::1] Y):
        cdef float[::1] D = np.zeros(Y.shape[0], 'f')
        self._evaluate_distances(Y, D)
        return np.asarray(D)
    #
    cdef _init_u(self, float[::1] Y):
        cdef float y, y_k, y_min, y_max, dy
        cdef Py_ssize_t j, k, N = Y.shape[0]
        
        if self.u is None or self.u.shape[0] != self.q:
            self.u = np.zeros(self.q, 'f')            
            self.u_prev = np.zeros(self.q, 'f')        
            self.u_min = np.zeros(self.q, 'f')        
        
        y_min = float_max
        y_max = float_min
        for k in range(N):
            y_k = Y[k]
            if y_k < y_min:
                y_min = y_k
            if y_k > y_max:
                y_max = y_k

        dy = (y_max - y_min) / self.q
        y = y_min
        for j in range(self.q):
            self.u_min[j] = self.u_prev[j] = self.u[j] = y + 0.5*dy
            y += dy
    #
    cdef _fit(self, float[::1] Y):
        cdef Py_ssize_t j, k, N = Y.shape[0]
        cdef Py_ssize_t q = self.q
        cdef float y, y_k, w_j
        cdef Py_ssize_t[::1] J = np.zeros(N, 'l')
        cdef float[::1] D = np.zeros(N, 'f')
        cdef float[::1] W = np.zeros(q, 'f')
        cdef float[::1] u = self.u
        cdef float[::1] u_prev = self.u_prev
        cdef float[::1] u_min = self.u_min
        cdef float tol = self.tol
        cdef size_t qd_size = q * cython.sizeof(float)
        cdef float qval_prev, qval_min = float_max
            
        self._init_u(Y)
        self.qval = self._evaluate_qval(Y)
        self._evaluate_classes(Y, J)
        self.qvals = [self.qval]

        self.K = 1
        while self.K <= self.n_iter:
            qval_prev = self.qval
            memcpy(&u_prev[0], &u[0], qd_size)

#             print(self.u.base)
            for j in range(q):
                memset(&u[0], 0, qd_size)
                memset(&W[0], 0, qd_size)

            for k in range(N):
                j = J[k]
                y_k = Y[k]
                w_j = self.func.derivative_div_x(y_k - u[j])
                W[j] += w_j
                u[j] += w_j * y_k

            for j in range(q):
                u[j] /= W[j]

            self.qval = self._evaluate_qval(Y)
            self.qvals.append(self.qval)
            self._evaluate_classes(Y, J)
            
            if self.qval < qval_min:
                qval_min = self.qval
                memcpy(&u_min[0], &u[0], qd_size)
                        
            if fabs(self.qval - qval_prev) / (1+fabs(qval_prev)) < tol:
                break            
            if max_l1_distance(&u[0], &u_prev[0], q) / maxabs_array(&u[0], q) < tol:
                break                

            self.K += 1
            
        memcpy(&u[0], &u_min[0], qd_size)
                
    def fit(self, Y):
        cdef float[::1] YY = np.ascontiguousarray(Y, 'f')
        self._fit(YY)

