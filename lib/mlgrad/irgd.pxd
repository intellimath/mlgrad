# cython: language_level=3

cimport cython

from mlgrad.func cimport Func
from mlgrad.model cimport Model
from mlgrad.regular cimport FuncMulti

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
#    bint isnan(double x)
#    bint isinf(double x)
#    bint signbit(double x)
#    bint isfinite(double x)

cdef extern from "Python.h":
    double PyFloat_GetMax()

cdef extern from "pymath.h" nogil:
    bint Py_IS_FINITE(double x)
    bint Py_IS_INFINITY(double x)
    bint Py_IS_NAN(double x)
    bint copysign(double x, double x)


cdef inline double min3(double v1, double v2, double v3):
    cdef double vmin = v1
    if v2 < vmin:
        vmin = v2
    if v3 < vmin:
        vmin = v3
    return vmin

cdef inline void fill_memoryview(double[::1] X, double c):
    cdef int m = X.shape[0]
    memset(&X[0], 0, m*cython.sizeof(double))    

cdef inline void copy_memoryview(double[::1] Y, double[::1] X):
    cdef int m = X.shape[0], n = Y.shape[0]

    if n < m:
        m = n
    memcpy(&Y[0], &X[0], m*cython.sizeof(double))    

cdef class IRGD(object):
    #
    cdef public Weights weights
    cdef public GD gd
    cdef public int n_iter, K
    cdef double h_anneal
    
    cdef bint with_scale
    
    cdef double decay
    
    cdef public double tol

    cdef double[::1] lval_all
    
    cdef public list lvals
    #cdef public list qvals
    cdef public list n_iters
    
#     cdef double[::1] grad
        
    #cdef public double[::1] param_best
    cdef public double lval, lval1, lval2
    cdef int m, M
    cdef bint u_only, is_warm_start
    cdef bint completed
    
    cdef double[::1] param_prev
    cdef double[::1] param_best
    cdef double lval_best
    #cdef double u_prev
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
