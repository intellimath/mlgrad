# cython: language_level=3

cimport cython

from libc.math cimport fabs, pow, sqrt, fmax, log
from libc.string cimport memcpy, memset

from mlgrad.avragg cimport Average
from mlgrad.distance cimport Distance

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

cdef class KMeans_MLSE:
    cdef public Distance[::1] distfunc
    cdef public double[:,::1] X
    cdef public int[::1] Y
    cdef public list loc
    cdef public list S
    cdef double[:,::1] D
    cdef double[::1] D_min
    cdef int[::1] count
    cdef Average avg

    cdef double dval, dval_prev, dval_min
    cdef double h
    cdef public int K, n_iter
    cdef public double tol
    cdef public list dvals
    
    cpdef calc_distances(self)
    cpdef double Q(self)
    
    
cdef class KMeans_MLocationEstimator(KMeans_MLSE):
#     cdef double[::1] loc_min
    cdef public double[::1] weights

cdef class KMeans_MScatterEstimator(KMeans_MLSE):
#     cdef double[:,::1] S_min
    cdef public double[::1] weights
#     cdef double[:,::1] V

cdef class KMeans_MLocationScatterEstimator(KMeans_MLSE):
    cdef public MLocationEstimator mlocation
    cdef public MScatterEstimator mscatter
#     cdef double[::1] loc_min
#     cdef double[:,::1] S_min
