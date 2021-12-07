# cython: language_level=3

cimport cython

from libc.math cimport fabs, pow, sqrt, fmax, log
from libc.string cimport memcpy, memset

from mlgrad.avragg cimport Average
from mlgrad.distance cimport Distance

cdef extern from "Python.h":
    float PyFloat_GetMax()
    float PyFloat_GetMin()

cdef inline void copy_memoryview(float[::1] Y, float[::1] X):
    cdef int m = X.shape[0], n = Y.shape[0]

    if n < m:
        m = n
    memcpy(&Y[0], &X[0], m*cython.sizeof(float))    

cdef inline void copy_memoryview2(float[:,::1] Y, float[:,::1] X):
    cdef int m = X.shape[0], n = X.shape[1]
    memcpy(&Y[0,0], &X[0,0], n*m*cython.sizeof(float))    

cdef class KMeans_MLSE:
    cdef public Distance[::1] distfunc
    cdef public float[:,::1] X
    cdef public int[::1] Y
    cdef public list loc
    cdef public list S
    cdef float[:,::1] D
    cdef float[::1] D_min
    cdef int[::1] count
    cdef Average avg

    cdef float dval, dval_prev, dval_min
    cdef float h
    cdef public int K, n_iter
    cdef public float tol
    cdef public list dvals
    
    cpdef calc_distances(self)
    cpdef float Q(self)
    
    
cdef class KMeans_MLocationEstimator(KMeans_MLSE):
#     cdef float[::1] loc_min
    cdef public float[::1] weights

cdef class KMeans_MScatterEstimator(KMeans_MLSE):
#     cdef float[:,::1] S_min
    cdef public float[::1] weights
#     cdef float[:,::1] V

cdef class KMeans_MLocationScatterEstimator(KMeans_MLSE):
    cdef public MLocationEstimator mlocation
    cdef public MScatterEstimator mscatter
#     cdef float[::1] loc_min
#     cdef float[:,::1] S_min
