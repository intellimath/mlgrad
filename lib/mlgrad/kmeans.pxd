# cython: language_level=3

cimport cython

from libc.math cimport fabs, pow, sqrt, fmax
from mlgrad.miscfuncs cimport init_rand, rand, fill
from libc.string cimport memcpy, memset

from mlgrad.func cimport Func
from mlgrad.distance cimport Distance
from mlgrad.risk cimport ED
from mlgrad.gd cimport FG
from mlgrad.avragg cimport Average

cdef extern from "Python.h":
    float PyFloat_GetMax()
    float PyFloat_GetMin()

cdef inline void fill_memoryview(float[::1] X, float c):
    cdef int m = X.shape[0]
    memset(&X[0], 0, m*cython.sizeof(float))    

cdef inline void fill_memoryview_int(int[::1] X, int c):
    cdef int m = X.shape[0]
    memset(&X[0], 0, m*cython.sizeof(int))    
    
cdef inline void fill_memoryview2(float[:,::1] X, float c):
    cdef int i, j
    cdef int m = X.shape[0], n = X.shape[1]
    memset(&X[0,0], 0, m*n*cython.sizeof(float))    

cdef inline void copy_memoryview(float[::1] Y, float[::1] X):
    cdef int m = X.shape[0], n = Y.shape[0]

    if n < m:
        m = n
    memcpy(&Y[0], &X[0], m*cython.sizeof(float))    

cdef inline void copy_memoryview2(float[:,::1] Y, float[:,::1] X):
    cdef int i, j
    cdef int m = X.shape[0], n = X.shape[1]
    memcpy(&Y[0,0], &X[0,0], n*m*cython.sizeof(float))    

# @cython.final
# cdef class ArrayRef:
#     cdef float[::1] data
    
#cpdef float[:, ::1] init_centers(float[:,::1] X, int n_class)    
#cpdef float[:, ::1] init_centers2(float[:,::1] X, int n_class)    
    
cdef class HCD:
    cdef public int n_class, n_param, n_sample
    cdef public Func func
    cdef public float[:,::1] X
    cdef public int[::1] Y
    cdef public float[::1] weights
    
    cdef public float[:,::1] params
    cdef float[:,::1] prev_params
    
    cdef public int n_iter, K
    cdef public float tol
    
    cpdef init(self)

    cdef bint stop_condition(self)

cdef class HCD_M1:
    cdef public int n_class, n_param, n_sample
    cdef public Average avrfunc
    cdef public float[:,::1] X
    cdef public int[::1] Y
    cdef public float[::1] weights
    cdef public float[::1] dist
    cdef public float[::1] grad

    cdef public float[:,::1] params
    cdef float[:,::1] prev_params
    
    cdef public int n_iter, K
    cdef public float tol

    #cdef float[::1] dist, grad

    cpdef init(self)
    
    cdef bint stop_condition(self)

cdef class HCD_M2:
    cdef public int n_class, n_param, n_sample
    cdef public Average avrfunc, minfunc
    cdef public float[:,::1] X
    cdef public float[:, ::1] Y
    cdef public float[:, ::1] D
    cdef public float[::1] weights
    cdef public float[::1] grad

    cdef public float[:,::1] params
    cdef float[:,::1] prev_params
    
    cdef public int n_iter, K
    cdef public float tol

    #cdef float[::1] dist, grad

    cpdef init(self)
    
    cdef bint stop_condition(self)
