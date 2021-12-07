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

cdef inline void copy_memoryview3(float[:,:,::1] Y, float[:,:,::1] X):
    cdef int m = X.shape[0], n = X.shape[1], l = X.shape[2]
    memcpy(&Y[0,0,0], &X[0,0,0], n*m*l*cython.sizeof(float))
    
cdef inline void fill_memoryview(float[::1] X, float c):
    cdef int m = X.shape[0]
    memset(&X[0], 0, m*cython.sizeof(float))    

cdef inline void fill_memoryview2(float[:,::1] X, float c):
    cdef int i, j
    cdef int m = X.shape[0], n = X.shape[1]
    memset(&X[0,0], 0, m*n*cython.sizeof(float))     

cdef inline void multiply_memoryview(float[::1] X, float c):
    cdef Py_ssize_t i
    cdef Py_ssize_t n = X.shape[0]
    cdef float *ptr = &X[0]

    for i in range(n):
        ptr[i] *= c
    
cdef inline void multiply_memoryview2(float[:,::1] X, float c):
    cdef Py_ssize_t i
    cdef Py_ssize_t mn = X.shape[0] * X.shape[1]
    cdef float *ptr = &X[0,0]

    for i in range(mn):
        ptr[i] *= c
    
cdef class MLSE:
    cdef public DistanceWithScale distfunc
    cdef public float[:,::1] X
    cdef public float[::1] loc
    cdef public float[:,::1] S
    cdef public float[::1] D
    cdef Average avg
    cdef public float[::1] weights
    cdef float tau
    cdef float logdet
    cdef Func reg

    cdef float dval, dval_prev, dval_min
    cdef float h
    cdef public int K, n_iter
    cdef public float tol
    cdef public list dvals
    
    cpdef _calc_distances(self)
    cpdef float Q(self)
    cpdef get_weights(self)        
    cpdef update_distfunc(self, S)
    
    
cdef class MLocationEstimator(MLSE):
    cdef float[::1] loc_min
    cdef float hh
    
    cpdef fit_step(self)
    cpdef bint stop_condition(self)

cdef class MScatterEstimator(MLSE):
    cdef float[:,::1] S_min
    cdef float hh
    cdef bint normalize

    cpdef fit_step(self)
    cpdef bint stop_condition(self)
    
cdef class MLocationScatterEstimator(MLSE):
    cdef public MLocationEstimator mlocation
    cdef public MScatterEstimator mscatter
    cdef float[::1] loc_min
    cdef float[:,::1] S_min
    cdef int n_step
    cdef bint normalize_S
    
    cpdef fit_step(self)
    cpdef bint stop_condition(self)
    

