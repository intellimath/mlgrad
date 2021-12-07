# cython: language_level=3

cimport cython

from mlgrad.model cimport Model
from mlgrad.func cimport Func, Square
from mlgrad.loss cimport Loss
from mlgrad.regnorm cimport FuncMulti
from mlgrad.averager cimport ArrayAverager, ArraySave
from mlgrad.avragg cimport Average, ArithMean
from mlgrad.weights cimport Weights
from mlgrad.risk cimport Functional, Risk, ERisk
from mlgrad.normalizer cimport Normalizer

from mlgrad.normalizer cimport LinearModelNormalizer

from mlgrad.miscfuncs cimport init_rand, rand, fill

from libc.math cimport fabs, pow, sqrt, fmax
# from libc.math cimport fabsf, powf, sqrtf, fmaxf
from libc.string cimport memcpy, memset

cdef extern from "Python.h":
    float PyFloat_GetMax()
    float PyFloat_GetMin()

cdef inline void fill_memoryview(float[::1] X, float c):
    cdef int m = X.shape[0]
    memset(&X[0], 0, m*cython.sizeof(float))    

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

# cdef class Fittable(object):
#     #
#     cpdef fit(self)

cdef class GD:

    cdef public Functional risk
    cdef public StopCondition stop_condition
    cdef public ParamRate h_rate
    cdef Normalizer normalizer

    cdef public float tol
    cdef public int n_iter, K, M, m
    cdef public bint completed

    cdef public float h
    
    cdef public list lvals
        
    cdef float[::1] param_min
    cdef float lval, lval_prev, lval_min
        
    cdef ArrayAverager grad_averager
    #cdef ArrayAverager param_averager

    cdef public object callback
    #
    cpdef init(self)
    #
    cpdef gradient(self)
    #
    cpdef fit_epoch(self)
    #
    cpdef finalize(self)

@cython.final
cdef class FG(GD):
    cdef float gamma
    #
    pass

@cython.final
cdef class FG_RUD(GD):
    cdef float[::1] param_prev
    cdef float gamma
    #
    pass

# cdef class RK4(GD):
#     #
#     cdef float[::1] param_k1, param_k2, param_k3, param_k4
#     cdef float[::1] grad_k1, grad_k2, grad_k3, grad_k4
    #
#     cpdef fit(self, float[:,::1] X, float[::1] Y, float[::1] W=*)
    #
#     cdef object fit_epoch(self, float[:,::1] X, float[::1] Y)
    #
#     #cdef float line_search(self, float[::1] Xk, float yk, float[::1] G)
    #
    #cdef bint stop_condition(self)

# cdef class AdaM(GD):
    #
#     cpdef fit(self, float[:,::1] X, float[::1] Y, float[::1] W=*)
    #
#     cdef object fit_epoch(self, float[:,::1] X, float[::1] Y)
    #
    #cdef float line_search(self, float[::1] Xk, float yk, float[::1] G)
    #
    #cdef bint stop_condition(self)
    #
#     cdef adamize(self)
    
cdef class SGD(GD):
    cdef float h0

#cdef class SAG(GD):

    #cdef float[:,::1] grad_all

    #cdef fill_tables(self, float[:,::1] X, float[::1] Y)

    #cdef fit_epoch(self, float[:,::1] X, float[::1] Y)

    #cdef fit_step_param(self, float[::1] Xk, float yk, int k, float Nd)

    #cdef float line_search(self, float[::1] Xk, float yk, float[::1] G, float g2, float h)

    #cdef bint stop_condition(self)
    #cdef bint check_condition(self)
    
#######################################################    

include "stopcond.pxd"
include "paramrate.pxd"

##########################################################
