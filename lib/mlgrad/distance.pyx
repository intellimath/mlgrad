# coding: utf-8

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

from libc.math cimport fabs, pow, sqrt, fmax, log

import numpy as np

# cdef inline double sign(double x):
#     if x >= 0:
#         return x
#     else:
#         return -x

cdef class Distance:
    cdef double evaluate(self, double[::1] x, double[::1] y) noexcept nogil:
        return 0
    cdef double _evaluate(self, const double *x, const double *y, const Py_ssize_t n) noexcept nogil:
        return 0
    cdef void gradient(self, double[::1] x, double[::1] y, double[::1] grad) noexcept nogil:
        pass
    def __call__(self, double[::1] x, double[::1] y):
        return self.evaluate(x, y)
    def grad(self, double[::1] x, double[::1] y):
        g = np.zeros(x.shape[0], 'd')
        self.gradient(x, y, g)
        return g
    cdef set_param(self, name, val):
        pass
    
cdef class AbsoluteDistance(Distance):

    cdef double evaluate(self, double[::1] x, double[::1] y) noexcept nogil:
        cdef int i, m = x.shape[0]
        cdef double s

        s = 0
        for i in range(m):
            s += fabs(x[i] - y[i])
        return s

    cdef void gradient(self, double[::1] x, double[::1] y, double[::1] grad) noexcept nogil:
        cdef int i, m = grad.shape[0]
        cdef double v

        for i in range(m):
            v = x[i] - y[i]
            if v > 0:
                grad[i] = 1
            elif v < 0:
                grad[i] = -1
            else:
                grad[i] = 0

cdef class EuclidDistance(Distance):

    cdef double _evaluate(self, double *x, double *y, Py_ssize_t m) noexcept nogil:
        cdef Py_ssize_t i
        cdef double s, v
        
        s = 0
        for i in range(m):
            v = x[i] - y[i]
            s += v * v
        return s
    
    cdef double evaluate(self, double[::1] x, double[::1] y) noexcept nogil:
        cdef Py_ssize_t i, m = x.shape[0]
        cdef double s, v
        
        s = 0
        for i in range(m):
            v = x[i] - y[i]
            s += v * v
        return s

    cdef void gradient(self, double[::1] x, double[::1] y, double[::1] grad) noexcept nogil:
        cdef Py_ssize_t i, m = grad.shape[0]
        cdef double v
    
        for i in range(m):
            grad[i] = 2 * (x[i] - y[i])
            
cdef class PowerDistance(Distance):
    
    def __init__(self, p):
        self.p = p
        
    cdef double evaluate(self, double[::1] x, double[::1] y) noexcept nogil:
        cdef int i, m = x.shape[0]
        cdef double s, v
        
        s = 0.0
        for i in range(m):
            v = x[i]-y[i]
            if v >= 0:
                s += pow(v, self.p)
            else:
                s += pow(-v, self.p)
        return s / self.p

    cdef void gradient(self, double[::1] x, double[::1] y, double[::1] grad) noexcept nogil:
        cdef int i, m = grad.shape[0]
        cdef double v
    
        for i in range(m):
            v = x[i] - y[i]
            if v >= 0:
                grad[i] = pow(fabs(v), self.p-1.0)
            else:
                grad[i] = -pow(fabs(v), self.p-1.0)                

cdef class DistanceWithScale(Distance):
    pass            
            
cdef class MahalanobisDistance(DistanceWithScale):
    
    def __init__(self, double[:,::1] S):
        self.S = S

    cdef double _evaluate(self, const double *x, const double *y, const Py_ssize_t n) noexcept nogil:
        cdef double *S = &self.S[0,0]
        cdef double xy1, xy2
        cdef Py_ssize_t i, j
        cdef double s, vi, vj, sj
        cdef double *S_i
        
        if n == 2:
            xy1 = x[0] - y[0]
            xy2 = x[1] - y[1]
            return S[0] * xy1 * xy1 + \
                   S[3] * xy2 * xy2 + \
                   2 * (S[1] * xy1 * xy2)
        
        s = 0
        S_i = S
        for i in range(n):
            vi = x[i] - y[i]            
            sj = S_i[i] * vi

            j = i + 1
            while j < n:
                vj = x[j] - y[j]
                sj += S_i[j] * vj
                j += 1

            s += 2 * vi * sj
            S_i += n

        return s
        
    cdef double evaluate(self, double[::1] x, double[::1] y) noexcept nogil:
        return self._evaluate(&x[0], &y[0], x.shape[0])

    cdef void gradient(self, double[::1] x, double[::1] y, double[::1] grad) noexcept nogil:
        cdef double[:,::1] S = self.S
        cdef double *S_i
        cdef Py_ssize_t i, j, m = grad.shape[0]
        cdef double s, xi
    
        for i in range(m):
            S_i = &S[i,0]
            s = 0
            for j in range(m):
                s += S_i[j] * (x[j] - y[j])
            grad[i] = 2*s
            
    cdef set_param(self, name, val):
        if name == "S":
            self.S = val
        else:
            raise NameError("invalid param's name")

