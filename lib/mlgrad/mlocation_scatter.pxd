# cython: language_level=3

cimport cython
from cython.view cimport indirect_contiguous

from libc.math cimport fabs, pow, sqrt, fmax, log
from libc.string cimport memcpy, memset

from mlgrad.miscfuncs cimport init_rand, rand, fill
from mlgrad.avragg cimport Average
from mlgrad.distance cimport Distance, DistanceWithScale, MahalanobisDistance
from mlgrad.funcs cimport Func

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
    
cdef class MLSE:
    cdef public DistanceWithScale distfunc
    cdef public double[:,::1] X
    cdef public double[::1] loc
    cdef public double[:,::1] S
    cdef public double[::1] D
    cdef Average avg
    cdef public double[::1] weights
    cdef double tau
    cdef double logdet
    cdef Func reg

    cdef double dval, dval_prev, dval_min
    cdef double h
    cdef public int K, n_iter
    cdef public double tol
    cdef public list dvals
    
    cpdef _calc_distances(self)
    cpdef double Q(self)
    cpdef get_weights(self)        
    cpdef update_distfunc(self, S)
    
    
cdef class MLocationEstimator(MLSE):
    cdef double[::1] loc_min
    cdef double hh
    
    cpdef fit_step(self)
    cpdef bint stop_condition(self)

cdef class MScatterEstimator(MLSE):
    cdef double[:,::1] S_min
    cdef double hh
    cdef bint normalize

    cpdef fit_step(self)
    cpdef bint stop_condition(self)
    
cdef class MLocationScatterEstimator(MLSE):
    cdef public MLocationEstimator mlocation
    cdef public MScatterEstimator mscatter
    cdef double[::1] loc_min
    cdef double[:,::1] S_min
    cdef int n_step
    cdef bint normalize_S
    
    cpdef fit_step(self)
    cpdef bint stop_condition(self)
    

