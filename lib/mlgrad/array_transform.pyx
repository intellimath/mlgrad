from libc.math cimport fabs, pow, sqrt, fmax
cimport mlgrad.inventory as inventory

import numpy as np

cdef void _array_zscore(double *a, double *b, Py_ssize_t n):
    cdef Py_ssize_t i
    cdef double mu = inventory._mean(a, n)
    cdef double sigma = inventory._std(a, mu, n)

    for i in range(n):
        b[i] = (a[i] - mu) / sigma

def array_zscore(double[::1] a, double[::1] b=None):
    cdef Py_ssize_t n = a.shape[0] 
    if b is None:
        b = inventory.empty_array(n)
    _array_zscore(&a[0], &b[0], a.shape[0])
    return np.asarray(b)
    
cdef void _array_modified_zscore(double *a, double *b, Py_ssize_t n):
    cdef Py_ssize_t i
    cdef double mu, sigma
    cdef double[::1] aa = inventory.empty_array(n)

    inventory._move(&aa[0], &a[0], n)
    mu = inventory._median_1d(aa)
    for i in range(n):
        aa[i] = fabs(aa[i] - mu)
    sigma = inventory._median_1d(aa)
    
    for i in range(n):
        b[i] = 0.6745 * (a[i] - mu) / sigma

def array_modified_zscore(double[::1] a, double[::1] b=None):
    cdef Py_ssize_t n = a.shape[0] 
    if b is None:
        b = inventory.empty_array(n)
    _array_modified_zscore(&a[0], &b[0], n)
    return np.asarray(b)

cdef void _array_diff2(double *x, double *y, const Py_ssize_t n):
    cdef Py_ssize_t i

    y[0] = 0
    y[n-1] = 0
    for i in range(1, n-1):
        y[i] = y[i-1] - 2*y[i] +y[i+1]

def array_diff2(double[::1] a, double[::1] b=None):
    cdef Py_ssize_t n = a.shape[0] 
    if b is None:
        b = inventory.empty_array(n)
    _array_diff2(&a[0], &b[0], n)
    return np.asarray(b)

cdef void _array_diff1(double *x, double *y, const Py_ssize_t n):
    cdef Py_ssize_t i

    y[0] = 0
    for i in range(1, n):
        y[i] = y[i] - y[i-1]

def array_diff1(double[::1] a, double[::1] b=None):
    cdef Py_ssize_t n = a.shape[0] 
    if b is None:
        b = inventory.empty_array(n)
    _array_diff1(&a[0], &b[0], n)
    return np.asarray(b)


# cdef class ArrayTransformer:
#     cdef transform(self, double[::1] a):
#         pass

# cdef class ArrayNormalizer2(ArrayTransformer):
#     def __init__(self, i0=0):
#         self.i0 = i0

#     cdef transform(self, double[::1] a):
#         cdef Py_ssize_t i
#         cdef double s, v

#         s = 0
#         for i in range(self.i0, a.shape[0]):
#             v = a[i]
#             s += v*v

#         s = sqrt(s)

#         for i in range(a.shape[0]):
#             a[i] /= s
