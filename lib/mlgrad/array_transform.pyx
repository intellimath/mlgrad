from libc.math cimport fabs, pow, sqrt, fmax
cimport mlgrad.inventory as inventory
from scipy.special import expit

import numpy as np

# cdef void _array_zscore(double[::1] a, double *b, Py_ssize_t n):
#     cdef Py_ssize_t i
#     cdef double mu = inventory._mean(a, n)
#     cdef double sigma = inventory._std(a, mu, n)

#     for i in range(n):
#         b[i] = (a[i] - mu) / sigma

# def array_zscore(double[::1] a, double[::1] b=None):
#     cdef Py_ssize_t n = a.shape[0] 
#     if b is None:
#         b = inventory.empty_array(n)
#     inventory._zscore(a, b)
#     return np.asarray(b)
    
# cdef void _array_modified_zscore(double *a, double *b, Py_ssize_t n):
#     cdef Py_ssize_t i
#     cdef double mu, sigma
#     cdef double[::1] aa = inventory.empty_array(n)
#     # cdef double[::1] ss = inventory.empty_array(n)

#     inventory._move(&aa[0], &a[0], n)
#     mu = inventory._median_1d(aa)
#     for i in range(n):
#         aa[i] = fabs(aa[i] - mu)
#     sigma = inventory._median_1d(aa)
    
#     for i in range(n):
#         b[i] = 0.6745 * (a[i] - mu) / sigma

# cdef void _array_modified_zscore_mu(double *a, double *b, Py_ssize_t n, double mu):
#     cdef Py_ssize_t i
#     cdef double sigma
#     cdef double[::1] aa = inventory.empty_array(n)

#     # inventory._move(&aa[0], &a[0], n)
#     for i in range(n):
#         aa[i] = fabs(a[i] - mu)
#     sigma = inventory._median_1d(aa)
    
#     for i in range(n):
#         b[i] = 0.6745 * (a[i] - mu) / sigma
        
# def array_modified_zscore2(double[:,::1] a, double[:,::1] b=None):
#     cdef Py_ssize_t i, n = a.shape[0], m = a.shape[1]
#     if b is None:
#         b = inventory.empty_array2(n,m)
#     for i in range(n):
#         inventory._zscore(a[i], b)
#     return np.asarray(b)

# def array_modified_zscore(double[::1] a, double[::1] b=None, mu=None):
#     cdef Py_ssize_t n = a.shape[0]
#     cdef double d_mu
#     if b is None:
#         b = inventory.empty_array(n)
#     if mu is None:
#         inventory._zscore(a, b)
#     else:
#         d_mu = mu
#         _array_modified_zscore_mu(&a[0], &b[0], n, d_mu)
#     return np.asarray(b)

cdef void _array_diff4(double *x, double *y, const Py_ssize_t n4):
    cdef Py_ssize_t i

    for i in range(n4):
        y[i] = x[i] - 4*x[i+1] + 6*x[i+2] - 4*x[i+3] + x[i+4]

def array_diff4(double[::1] a, double[::1] b=None):
    cdef Py_ssize_t n = a.shape[0]
    if b is None:
        b = inventory.empty_array(n-4)
    _array_diff4(&a[0], &b[0], n-4)
    return np.asarray(b)

cdef void _array_diff3(double *x, double *y, const Py_ssize_t n3):
    cdef Py_ssize_t i

    for i in range(n3):
        y[i] = x[i] - 3*x[i+1] + 3*x[i+2] - x[i+3]

def array_diff3(double[::1] a, double[::1] b=None):
    cdef Py_ssize_t n = a.shape[0]
    if b is None:
        b = inventory.empty_array(n-3)
    _array_diff3(&a[0], &b[0], n-3)
    return np.asarray(b)

cdef void _array_diff2(double *x, double *y, const Py_ssize_t n2):
    cdef Py_ssize_t i

    for i in range(n2):
        y[i] = x[i] - 2*x[i+1] + x[i+2]

def array_diff2(double[::1] a, double[::1] b=None):
    cdef Py_ssize_t n = a.shape[0]
    if b is None:
        b = inventory.empty_array(n-2)
    _array_diff2(&a[0], &b[0], n-2)
    return np.asarray(b)

cdef void _array_diff1(double *x, double *y, const Py_ssize_t n1):
    cdef Py_ssize_t i

    for i in range(n1):
        y[i] = x[i+1] - x[i]

def array_diff1(double[::1] a, double[::1] b=None):
    cdef Py_ssize_t n = a.shape[0] 
    if b is None:
        b = inventory.empty_array(n-1)
    _array_diff1(&a[0], &b[0], n-1)
    return np.asarray(b)

def array_rel_max(E):
    abs_E = abs(E)
    max_E = max(abs_E)
    min_E = min(abs_E)
    rel_E =  (abs_E - min_E) / (max_E - min_E)
    return rel_E

def array_expit_sym(E):
    return expit(-5*E)
def array_expit(E):
    return expit(E)

def array_sqrtit(E):
    return (1 - E / np.sqrt(1 + E*E)) / 2
