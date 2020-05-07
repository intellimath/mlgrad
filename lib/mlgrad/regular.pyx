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

import numpy as np
from mlgrad.model import as_array_1d

cdef class FuncMulti:

    cdef double evaluate(self, double[::1] param):
        return 0
    
    cdef void gradient(self, double[::1] param, double[::1] grad):
        pass
    
    def __call__(self, param):
        cdef double[::1] x1d

        x1d = as_array_1d(param)
        return self.evaluate(x1d)

cdef class Power(FuncMulti):
    
    def __init__(self, p=2.0):
        self.p = p
        self.all = all

    cdef double evaluate(self, double[::1] param):
        cdef int i, m
        cdef double s
        cdef double* param_ptr = &param[0]
        
        m = param.shape[0]
        s = 0
        
        for i in range(m):
            s += pow(fabs(param_ptr[i]), self.p)
        
        s /= self.p
        return s

    cdef void gradient(self, double[::1] param, double[::1] grad):
        cdef int i, m
        cdef double v
        cdef double* param_ptr = &param[0]
        cdef double* grad_ptr
    
        m = param.shape[0]
        # if grad is None:
        #     grad = np.empty((m,), dtype='d')
        grad_ptr = &grad[0]

        for i in range(m):
            v = pow(fabs(param_ptr[i]), self.p-1.0)
            if v < 0:
                grad_ptr[i] = -v
            else:
                grad_ptr[i] = v

    def _repr_latex_(self):
        return r"$||\mathbf{w}||_{%s}^{%s}=\sum_{i=0}^n w_i^{%s}$" % (self.p, self.p, self.p)

cdef class Square(FuncMulti):

    cdef double evaluate(self, double[::1] param):
        cdef int i, m
        cdef double s, v
        cdef double* param_ptr = &param[0]

        m = param.shape[0]
        s = 0
        for i in range(m):
            v = param_ptr[i]
            s += v * v

        s /= 2.
        return s

    cdef void gradient(self, double[::1] param, double[::1] grad):
        cdef int i, m
        cdef double* param_ptr = &param[0]
        cdef double* grad_ptr

        m = param.shape[0]
        # if grad is None:
        #     grad = np.empty((m,), dtype='d')
        grad_ptr = &grad[0]

        for i in range(m):
            grad_ptr[i] = param_ptr[i]    
        
    def _repr_latex_(self):
        return r"$||\mathbf{w}||_2^2=\sum_{i=0}^n w_i^2$"
        

cdef class Absolute(FuncMulti):

    cdef double evaluate(self, double[::1] param):
        cdef int i, m
        cdef double s
        cdef double* param_ptr = &param[0]

        m = param.shape[0]
        s = 0
        for i in range(m):
            s += fabs(param_ptr[i])
        return s

    cdef void gradient(self, double[::1] param, double[::1] grad):
        cdef int i, m
        cdef double* param_ptr = &param[0]
        cdef double* grad_ptr

        m = param.shape[0]
        if grad is None:
            grad = np.empty((m,), dtype='d')
        grad_ptr = &grad[0]

        for i in range(m):
            v = param_ptr[i]
            if v > 0:
                grad_ptr[i] = 1
            elif v < 0:
                grad_ptr[i] = -1
            else:
                grad_ptr[i] = 0
    
    def _repr_latex_(self):
        return r"$||\mathbf{w}||_1=\sum_{i=0}^n |w_i|$"

cdef class SquareForm(FuncMulti):
    
    def __init__(self, double[:,::1] matrix):
        if matrix.shape[0] != matrix.shape[1]-1:
            raise RuntimeError("Invalid shape: (%s,%s)" % (matrix.shape[0], matrix.shape[1]))
        self.matrix = matrix
    #
    cdef double evaluate(self, double[::1] x):
        cdef double[:,::1] mat = self.matrix
        cdef int n_row = mat.shape[0]
        cdef int n_col = mat.shape[1]
        cdef double s, val
        cdef int i, j
        
        val = 0
        for j in range(n_row):
            s = mat[j,0]
            for i in range(1, n_col):
                s += mat[j,i] * x[i-1]
            val += s*s
        return 0.5*val

    cdef void gradient(self, double[::1] x, double[::1] y):
        cdef double[:,::1] mat = self.matrix
        cdef int n_row = mat.shape[0]
        cdef int n_col = mat.shape[1]
        cdef double s
        cdef int i, j
        
        n_row = mat.shape[0]
        n_col = mat.shape[1]
        
        fill_memoryview(y, 0)
        for j in range(n_row):
            s = mat[j,0]
            for i in range(1, n_col):
                s += mat[j,i] * x[i-1]

            for i in range(1, n_col):
                y[i-1] += s*mat[j,i]

cdef class Rosenbrok(FuncMulti):

    cdef double evaluate(self, double[::1] param):
        return 10. * (param[1] - param[0]**2)**2 + 0.1*(1. - param[0])**2
    
    cdef void gradient(self, double[::1] param, double[::1] grad):
        grad[0] = -40. * (param[1] - param[0]**2) * param[0] - 0.2 * (1. - param[0])
        grad[1] = 20. * (param[1] - param[0]**2)
        
        
cdef class Himmelblau(FuncMulti):

    cdef double evaluate(self, double[::1] param):
        return (param[0]**2 + param[1] - 11)**2 + (param[0] + param[1]**2 - 7)**2
    
    cdef void gradient(self, double[::1] param, double[::1] grad):
        grad[0] = 4*(param[0]**2 + param[1] - 11) * param[0] + 2*(param[0] + param[1]**2 - 7)
        grad[1] = 2*(param[0]**2 + param[1] - 11) + 4*(param[0] + param[1]**2 - 7) * param[1]
        
# cdef class Func(FuncMulti):
    
#     def __init__(self, Func func, double[::1] Y):
#         self.Y = Y
#         self.func = func
        
#     cdef double evaluate(self, double[::1] param):
#         cdef doubel[::1] Y = self.Y
#         cdef int k, N = Y.shape[0]
#         cdef Func func = self.func
#         cdef double s
#         cdef double[::1]
        
#         s = 0
#         for k in range(N):
#             s += func(param[0] - Y[k])
            