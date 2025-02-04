from libc.math cimport fabs, pow, sqrt, fmax
cimport mlgrad.inventory as inventory
from scipy.special import expit

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
    # cdef double[::1] ss = inventory.empty_array(n)

    inventory._move(&aa[0], &a[0], n)
    mu = inventory._median_1d(aa)
    for i in range(n):
        aa[i] = fabs(aa[i] - mu)
    sigma = inventory._median_1d(aa)
    
    for i in range(n):
        b[i] = 0.6745 * (a[i] - mu) / sigma

cdef void _array_modified_zscore_mu(double *a, double *b, Py_ssize_t n, double mu):
    cdef Py_ssize_t i
    cdef double sigma
    cdef double[::1] aa = inventory.empty_array(n)

    # inventory._move(&aa[0], &a[0], n)
    for i in range(n):
        aa[i] = fabs(a[i] - mu)
    sigma = inventory._median_1d(aa)
    
    for i in range(n):
        b[i] = 0.6745 * (a[i] - mu) / sigma
        
def array_modified_zscore2(double[:,::1] a, double[:,::1] b=None):
    cdef Py_ssize_t i, n = a.shape[0], m = a.shape[1]
    if b is None:
        b = inventory.empty_array2(n,m)
    for i in range(n):
        _array_modified_zscore(&a[i,0], &b[i,0], m)
    return np.asarray(b)

def array_modified_zscore(double[::1] a, double[::1] b=None, mu=None):
    cdef Py_ssize_t n = a.shape[0]
    cdef double d_mu
    if b is None:
        b = inventory.empty_array(n)
    if mu is None:
        _array_modified_zscore(&a[0], &b[0], n)
    else:
        d_mu = mu
        _array_modified_zscore_mu(&a[0], &b[0], n, d_mu)
    return np.asarray(b)

cdef void _array_diff2(double *x, double *y, const Py_ssize_t n):
    cdef Py_ssize_t i

    y[0] = 0 # x[1] - x[0]
    y[n-1] = 0 # x[n-1] - x[n-2]
    for i in range(1, n-1):
        y[i] = x[i-1] - 2*x[i] + x[i+1]

def array_diff2(double[::1] a, double[::1] b=None):
    cdef Py_ssize_t n = a.shape[0]
    if b is None:
        b = inventory.empty_array(n)
    _array_diff2(&a[0], &b[0], n)
    return np.asarray(b)

cdef void _array_diff1(double *x, double *y, const Py_ssize_t n):
    cdef Py_ssize_t i

    for i in range(n-1):
        y[i] = x[i+1] - x[i]
    y[n-1] = 0.5 * (x[n-1] - x[n-3])

def array_diff1(double[::1] a, double[::1] b=None):
    cdef Py_ssize_t n = a.shape[0] 
    if b is None:
        b = inventory.empty_array(n)
    _array_diff1(&a[0], &b[0], n)
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
