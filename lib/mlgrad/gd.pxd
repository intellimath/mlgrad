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

from mlgrad.miscfuncs cimport init_rand, rand, fill

from libc.math cimport fabs, pow, sqrt, fmax
# from libc.math cimport fabsf, powf, sqrtf, fmaxf
from libc.string cimport memcpy, memset

cdef extern from "Python.h":
    double Pydouble_GetMax()
    double Pydouble_GetMin()

cdef inline void fill_memoryview(double[::1] X, double c):
    memset(&X[0], 0, X.shape[0]*cython.sizeof(double))    

cdef inline void fill_memoryview2(double[:,::1] X, double c):
    memset(&X[0,0], 0, X.shape[0]*X.shape[1]*cython.sizeof(double))    

cdef inline void copy_memoryview(double[::1] Y, double[::1] X):
    memcpy(&Y[0], &X[0], Y.shape[0]*cython.sizeof(double))    

cdef inline void copy_memoryview2(double[:,::1] Y, double[:,::1] X):
    memcpy(&Y[0,0], &X[0,0], X.shape[0]*X.shape[1]*cython.sizeof(double))    
    
# cdef class Fittable(object):
#     #
#     cpdef fit(self)

cdef class GD:

    cdef public Functional risk
    cdef public StopCondition stop_condition
    cdef public ParamRate h_rate
    cdef Normalizer normalizer

    cdef public double tol
    cdef public int n_iter, K, M
    # cdef public int m
    cdef public bint completed

    cdef public double h
    
    cdef public list lvals
        
    cdef double[::1] param_min
    cdef double lval, lval_prev, lval_min
        
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
    cdef double gamma
    #
    pass

@cython.final
cdef class FG_RUD(GD):
    cdef double[::1] param_prev
    cdef double gamma
    #
    pass

# cdef class RK4(GD):
#     #
#     cdef double[::1] param_k1, param_k2, param_k3, param_k4
#     cdef double[::1] grad_k1, grad_k2, grad_k3, grad_k4
    #
#     cpdef fit(self, double[:,::1] X, double[::1] Y, double[::1] W=*)
    #
#     cdef object fit_epoch(self, double[:,::1] X, double[::1] Y)
    #
#     #cdef double line_search(self, double[::1] Xk, double yk, double[::1] G)
    #
    #cdef bint stop_condition(self)

# cdef class AdaM(GD):
    #
#     cpdef fit(self, double[:,::1] X, double[::1] Y, double[::1] W=*)
    #
#     cdef object fit_epoch(self, double[:,::1] X, double[::1] Y)
    #
    #cdef double line_search(self, double[::1] Xk, double yk, double[::1] G)
    #
    #cdef bint stop_condition(self)
    #
#     cdef adamize(self)
    
cdef class SGD(GD):
    cdef double h0

#cdef class SAG(GD):

    #cdef double[:,::1] grad_all

    #cdef fill_tables(self, double[:,::1] X, double[::1] Y)

    #cdef fit_epoch(self, double[:,::1] X, double[::1] Y)

    #cdef fit_step_param(self, double[::1] Xk, double yk, int k, double Nd)

    #cdef double line_search(self, double[::1] Xk, double yk, double[::1] G, double g2, double h)

    #cdef bint stop_condition(self)
    #cdef bint check_condition(self)
    
#######################################################    

include "stopcond.pxd"
include "paramrate.pxd"
include "normalizer.pxd"

##########################################################
