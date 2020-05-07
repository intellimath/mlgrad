# cython: language_level=3

cimport cython
from cython.view cimport indirect_contiguous

from libc.math cimport fabs, pow, sqrt, fmax, log
from libc.string cimport memcpy, memset

from mlgrad.miscfuncs cimport init_rand, rand, fill
from mlgrad.avragg cimport Average
from mlgrad.distance cimport Distance, DistanceWithScale, MahalanobisDistance
from mlgrad.func cimport Func

cdef extern from "Python.h":
    double PyFloat_GetMax()
    double PyFloat_GetMin()

cdef inline void copy_memoryview(double[::1] Y, double[::1] X):
    cdef int m = X.shape[0], n = Y.shape[0]
    if n < m:
        m = n
    memcpy(&Y[0], &X[0], m*cython.sizeof(double))    

cdef inline void copy_memoryview2(double[:,::1] Y, double[:,::1] X):
    cdef int m = X.shape[0], n = X.shape[1]
    memcpy(&Y[0,0], &X[0,0], n*m*cython.sizeof(double))

cdef inline void copy_memoryview3(double[:,:,::1] Y, double[:,:,::1] X):
    cdef int m = X.shape[0], n = X.shape[1], l = X.shape[2]
    memcpy(&Y[0,0,0], &X[0,0,0], n*m*l*cython.sizeof(double))
    
cdef inline void fill_memoryview(double[::1] X, double c):
    cdef int m = X.shape[0]
    memset(&X[0], 0, m*cython.sizeof(double))    

cdef inline void fill_memoryview2(double[:,::1] X, double c):
    cdef int i, j
    cdef int m = X.shape[0], n = X.shape[1]
    memset(&X[0,0], 0, m*n*cython.sizeof(double))     

cdef inline void multiply_memoryview(double[::1] X, double c):
    cdef Py_ssize_t i
    cdef Py_ssize_t n = X.shape[0]
    cdef double *ptr = &X[0]

    for i in range(n):
        ptr[i] *= c
    
cdef inline void multiply_memoryview2(double[:,::1] X, double c):
    cdef Py_ssize_t i
    cdef Py_ssize_t mn = X.shape[0] * X.shape[1]
    cdef double *ptr = &X[0,0]

    for i in range(mn):
        ptr[i] *= c

cdef class MLSE2:
    cdef public DistanceWithScale[::1] distfuncs
    cdef public double[:,::1] X
    cdef public double[:,::1] locs, locs_min
    cdef public int n_locs
    cdef public double[:,:,::1] scatters, scatters_min
    cdef double[::1] D
    cdef double[::1] logdet
    cdef Average avg
    cdef public double[::1] weights
    cdef double[::1] W
    cdef int[::1] Y, Ns

    cdef double dval, dval_prev, dval_min
    cdef double dval2, dval2_prev, dval2_min
    cdef double h
    cdef public int K, Ks, n_iter, n_step
    cdef public double tol
    cdef public list dvals, dvals2

    cdef bint normalize, pinv
    
    cpdef calc_distances(self)
    cpdef double Q(self)
#     cpdef get_weights(self)

cdef class MLocationsScattersEstimator(MLSE2):
    pass