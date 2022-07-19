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

#
# utility functions
#

cimport cython
from libc.math cimport fabs, pow, sqrt, fmax
from libc.stdlib cimport rand as stdlib_rand, srand
from libc.time cimport time


#cdef extern from "math.h":
#    bint isnan(double x)
#    bint isinf(double x)
#    bint signbit(double x)
#    bint isfinite(double x)
    
#cdef extern from "sys/time.h":
#    cdef long time(long*)

cdef void init_rand():
    srand(time(NULL))    

cdef long rand(long N):
    return stdlib_rand() % N

ctypedef double[:] array1d
ctypedef double[:,::1] array2d
ctypedef long[:] array1d_long    

# cdef double[:] as_memoryview_1d(object X):
#     cdef arraybuffer_dsc dsc
#     tp = type(X)
#     if tp is list:
#         dsc = arraybuffer_dsc((len(X),), 'd')
#         return arraybuffer(dsc)
#     elif tp is tuple:
#         dsc = arraybuffer_dsc((len(X),), 'd')
#         return arraybuffer(dsc)
#     else:
#         return X

# def fmax1(x):
#     if x < 1 and x > -1:
#         return 1
#     return x
#
# def fabsmax1(x):
#     if x >= 0:
#         if x < 1.0:
#             return 1.0
#         else:
#             return x
#     else:
#         if x > -1.0:
#             return 1.0
#         else:
#             return -x
#
# def isfinite1d(X):
#     N = X.shape[0]
#     for i in range(N):
#         if not isfinite(X[i]):
#             return 0
#     return 1
#

cdef void  multiply_scalar(double[::1] X, double c):
    cdef Py_ssize_t i
    cdef double *XX = &X[0]
    for i in range(X.shape[0]):
        XX[i] *= c

cdef void fill(double[::1] X, double c):
    cdef Py_ssize_t i
    cdef double *XX = &X[0]
    for i in range(X.shape[0]):
        XX[i] = c

#
# def add_inplace(X, Y):
#     m = X.shape[0]
#     for i in range(m):
#         X[i] += Y[i]
#     return X
#
# def sub_inplace(X, Y):
#     m = X.shape[0]
#     for i in range(m):
#         X[i] -= Y[i]
#     return X
#
# def mean1d(X):
#     N = X.shape[0]
#     s = 0.0
#     for i in range(N):
#         s += X[i]
#     return s / N
#
# def sum1d(X):
#     N = X.shape[0]
#     s = 0.0
#     for i in range(N):
#         s += X[i]
#     return s
#
# def absnorm(X):
#     N = X.shape[0]
#     s = 0.0
#     for i in range(N):
#         s += fabs(X[i])
#     return s
#
# def max1d(X):
#     N = X.shape[0]
#     s = X[0]
#     for i in range(N):
#         if X[i] > s:
#             s = X[i]
#     return s

cdef double absmax_1d(double[::1] X):
    cdef Py_ssize_t i
    cdef double s, v
    cdef double *XX = &X[0]
    
    s = XX[0]
    for i in range(X.shape[0]):
        v = XX[i]
        if v < 0:
            v = -v
        if v > s:
            s = v
    return s

# def mean2d(X):
#     N = X.shape[0]
#     m = X.shape[1]
#     Y = np.zeros((m,), np.double)
#     for j in range(m):
#         s = 0.0
#         for i in range(N):
#             s += X[i,j]
#         Y[j] = s / N
#     return Y
#
# def sum2d(X):
#     N = X.shape[0]
#     m = X.shape[1]
#     Y = np.zeros((m,), np.double)
#     for j in range(m):
#         s = 0.0
#         for i in range(N):
#             s += X[i,j]
#     return Y
#
# def max_abs_diff(X, Y):
#     m = X.shape[0]
#     v_max = 0.0
#     for i in range(m):
#         v = X[i] - Y[i]
#         if v < 0:
#             v = -v
#         if v > v_max:
#             v_max = v
#     return v_max
#
# def rel_diff(x, y):
#     if x < 0:
#         x = -x
#     if y < 0:
#         y = -y
#
#     if y >= 1:
#         z = x / y
#     else:
#         z = x
#
#     return z
