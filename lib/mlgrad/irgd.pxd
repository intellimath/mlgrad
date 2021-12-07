# cython: language_level=3

cimport cython

from mlgrad.func cimport Func
from mlgrad.model cimport Model
from mlgrad.regnorm cimport FuncMulti

from mlgrad.avragg cimport Average
from mlgrad.gd cimport GD
from mlgrad.risk cimport Risk, Functional
from mlgrad.weights cimport Weights
# from mlgrad.averager cimport ScalarAverager, ArrayAverager

from mlgrad.avragg cimport array_min
from mlgrad.miscfuncs cimport init_rand, rand, fill

from libc.math cimport fabs, pow, sqrt, fmax, log, log2
from libc.string cimport memcpy, memset
#from libc.math cimport isnan, isinf

#cdef extern from "math.h":
#    bint isnan(float x)
#    bint isinf(float x)
#    bint signbit(float x)
#    bint isfinite(float x)

cdef extern from "Python.h":
    float PyFloat_GetMax()

cdef extern from "pymath.h" nogil:
    bint Py_IS_FINITE(float x)
    bint Py_IS_INFINITY(float x)
    bint Py_IS_NAN(float x)
    bint copysign(float x, float x)


cdef inline float min3(float v1, float v2, float v3):
    cdef float vmin = v1
    if v2 < vmin:
        vmin = v2
    if v3 < vmin:
        vmin = v3
    return vmin

cdef inline void fill_memoryview(float[::1] X, float c):
    cdef int m = X.shape[0]
    memset(&X[0], 0, m*cython.sizeof(float))    

cdef inline void copy_memoryview(float[::1] Y, float[::1] X):
    cdef int m = X.shape[0], n = Y.shape[0]

    if n < m:
        m = n
    memcpy(&Y[0], &X[0], m*cython.sizeof(float))    

cdef class IRGD(object):
    #
    cdef public Weights weights
    cdef public GD gd
    cdef public int n_iter, K
    cdef float h_anneal
    
    cdef bint with_scale
    
    cdef float decay
    
    cdef public float tol

#     cdef public float[::1] lval_all
    
    cdef public list lvals
    #cdef public list qvals
    cdef public list n_iters
    
#     cdef float[::1] grad
        
    #cdef public float[::1] param_best
    cdef public float lval, lval1, lval2
    cdef int m, M
    cdef bint u_only, is_warm_start
    cdef public bint completed
    
    cdef float[::1] param_prev
    cdef float[::1] param_best
    cdef float lval_best
    #cdef float u_prev
    #
    cdef public object callback
    #
    #cdef ArrayAverager param_averager
    #cdef ScalarAverager u_averager
    #
    #def fit(self)
    #
    cdef finalize(self)
    #
    cdef inline bint stop_condition(self)
