# cython: language_level=3

cimport cython

from libc.math cimport fabs, pow, sqrt, fmax
from mlgrad.inventory cimport init_rand, rand
from libc.string cimport memcpy, memset

from mlgrad.funcs cimport Func
from mlgrad.distance cimport Distance
from mlgrad.risks cimport ED
from mlgrad.gd cimport FG
from mlgrad.avragg cimport Average

cdef extern from "Python.h":
    double PyFloat_GetMax()
    double PyFloat_GetMin()

cdef inline void fill_memoryview(double[::1] X, double c):
    cdef int m = X.shape[0]
    memset(&X[0], 0, m*cython.sizeof(double))    

cdef inline void fill_memoryview_int(int[::1] X, int c):
    cdef int m = X.shape[0]
    memset(&X[0], 0, m*cython.sizeof(int))    
    
cdef inline void fill_memoryview2(double[:,::1] X, double c):
    cdef int i, j
    cdef int m = X.shape[0], n = X.shape[1]
    memset(&X[0,0], 0, m*n*cython.sizeof(double))    

cdef inline void copy_memoryview(double[::1] Y, double[::1] X):
    cdef int m = X.shape[0], n = Y.shape[0]

    if n < m:
        m = n
    memcpy(&Y[0], &X[0], m*cython.sizeof(double))    

cdef inline void copy_memoryview2(double[:,::1] Y, double[:,::1] X):
    cdef int i, j
    cdef int m = X.shape[0], n = X.shape[1]
    memcpy(&Y[0,0], &X[0,0], n*m*cython.sizeof(double))    

# @cython.final
# cdef class ArrayRef:
#     cdef double[::1] data
    
#cpdef double[:, ::1] init_centers(double[:,::1] X, int n_class)    
#cpdef double[:, ::1] init_centers2(double[:,::1] X, int n_class)    
    
cdef class HCD:
    cdef public int n_class, n_param, n_sample
    cdef public Func func
    cdef public double[:,::1] X
    cdef public int[::1] Y
    cdef public double[::1] weights
    
    cdef public double[:,::1] params
    cdef double[:,::1] prev_params
    
    cdef public int n_iter, K
    cdef public double tol
    
    cpdef init(self)

    cdef bint stop_condition(self)

cdef class HCD_M1:
    cdef public int n_class, n_param, n_sample
    cdef public Average avrfunc
    cdef public double[:,::1] X
    cdef public int[::1] Y
    cdef public double[::1] weights
    cdef public double[::1] dist
    cdef public double[::1] grad

    cdef public double[:,::1] params
    cdef double[:,::1] prev_params
    
    cdef public int n_iter, K
    cdef public double tol

    #cdef double[::1] dist, grad

    cpdef init(self)
    
    cdef bint stop_condition(self)

cdef class HCD_M2:
    cdef public int n_class, n_param, n_sample
    cdef public Average avrfunc, minfunc
    cdef public double[:,::1] X
    cdef public double[:, ::1] Y
    cdef public double[:, ::1] D
    cdef public double[::1] weights
    cdef public double[::1] grad

    cdef public double[:,::1] params
    cdef double[:,::1] prev_params
    
    cdef public int n_iter, K
    cdef public double tol

    #cdef double[::1] dist, grad

    cpdef init(self)
    
    cdef bint stop_condition(self)
