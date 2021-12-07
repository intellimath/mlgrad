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
    
cdef inline int int_min(int a, int b) nogil:
    if a < b:
        return a
    else:
        return b

cdef inline void copy_memoryview(float[::1] Y, float[::1] X) nogil:
    cdef Py_ssize_t m = X.shape[0], n = Y.shape[0]
    if n < m:
        m = n
    memcpy(&Y[0], &X[0], m*cython.sizeof(float))    

cdef inline void copy_memoryview2(float[:,::1] Y, float[:,::1] X) nogil:
    cdef Py_ssize_t m = X.shape[0], n = X.shape[1]
    memcpy(&Y[0,0], &X[0,0], n*m*cython.sizeof(float))

cdef inline void copy_memoryview3(float[:,:,::1] Y, float[:,:,::1] X) nogil:
    cdef Py_ssize_t m = X.shape[0], n = X.shape[1], l = X.shape[2]
    memcpy(&Y[0,0,0], &X[0,0,0], n*m*l*cython.sizeof(float))
    
cdef inline void fill_memoryview(float[::1] X, float c) nogil:
    cdef Py_ssize_t m = X.shape[0]
    memset(&X[0], 0, m*cython.sizeof(float))    

cdef inline void fill_memoryview2(float[:,::1] X, float c) nogil:
    cdef Py_ssize_t i, j
    cdef Py_ssize_t m = X.shape[0], n = X.shape[1]
    memset(&X[0,0], 0, m*n*cython.sizeof(float))     

cdef inline void fill_memoryview3(float[:,:,::1] X, float c) nogil:
    cdef Py_ssize_t i, j
    cdef Py_ssize_t m = X.shape[0], n = X.shape[1], l = X.shape[2]
    memset(&X[0,0,0], 0, m*n*l*cython.sizeof(float))     
    
cdef inline void multiply_memoryview(float[::1] X, float c) nogil:
    cdef Py_ssize_t i
    cdef Py_ssize_t n = X.shape[0]
    cdef float *ptr = &X[0]

    for i in range(n):
        ptr[i] *= c
        
cdef inline void multiply_memoryview2(float[:,::1] X, float c) nogil:
    cdef Py_ssize_t i
    cdef Py_ssize_t mn = X.shape[0] * X.shape[1]
    cdef float *ptr = &X[0,0]

    for i in range(mn):
        ptr[i] *= c

cdef inline void multiply_memoryview3(float[:,:,::1] X, float c) nogil:
    cdef Py_ssize_t i, j
    cdef Py_ssize_t m = X.shape[0], n = X.shape[1], l = X.shape[2]
    cdef Py_ssize_t N = m*n*l
    cdef float *ptr = &X[0,0,0]
    
    for i in range(N):
        ptr[i] *= c
        
cdef class MLSE2:
    cdef public DistanceWithScale[::1] distfuncs
    cdef public float[:,::1] X
    cdef public float[:,::1] locs, locs_min
    cdef public Py_ssize_t n_locs
    cdef public float[:,:,::1] scatters, scatters_min
    cdef float[:,::1] DD
    cdef float[::1] D
    cdef float[:,::1] GG
    cdef Average avg
    cdef Average avg_min
    cdef public float[::1] weights
    cdef float[::1] W
    cdef float[::1] xy
    cdef Py_ssize_t[::1] Ns

    cdef float dval, dval_prev, dval_min
    cdef float dval2, dval2_prev, dval2_min
    cdef float h
    cdef public int K, Ks, n_iter, n_iter_s, n_step
    cdef public float tol
    cdef public list dvals, dvals2
    cdef float alpha

    cpdef calc_distances(self)
    cpdef calc_weights(self)
    cpdef calc_update_GG(self)
    cpdef float Q(self)
    cpdef float local_Q(self)
#     cpdef get_weights(self)


cdef class MLocationsScattersEstimator(MLSE2):
    cdef bint stop_condition(self)
    cdef bint stop_condition2(self)
