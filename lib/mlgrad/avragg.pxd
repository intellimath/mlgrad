# cython: language_level=3

cimport cython

from mlgrad.func cimport Func, ParametrizedFunc
from mlgrad.averager cimport ScalarAverager

from mlgrad.miscfuncs cimport init_rand, rand, fill

from libc.math cimport fabs, pow, sqrt, fmax, log, exp

cdef extern from "Python.h":
    double PyFloat_GetMax()

cdef extern from "pymath.h" nogil:
    bint Py_IS_FINITE(double x)
    bint Py_IS_INFINITY(double x)
    bint Py_IS_NAN(double x)
    bint copysign(double x, double x)

cdef inline double array_min(double[::1] arr):
    cdef Py_ssize_t i, N = arr.shape[0]
    cdef double v, min_val = arr[0]

    for i in range(N):
        v = arr[i]
        if v < min_val:
            min_val = v

    return min_val

cdef inline double array_mean(double[::1] arr):
    cdef Py_ssize_t i, N = arr.shape[0]
    cdef double v

    v = 0
    for i in range(N):
        v += arr[i]

    return v / N

cdef inline void array_add_scalar(double[::1] arr, double v):
    cdef Py_ssize_t i, N = arr.shape[0]

    for i in range(N):
        arr[i] += v

cdef class Penalty:
    cdef public Func func

    cdef double evaluate(self, double[::1] Y, double u)
    cdef double derivative(self, double[::1] Y, double u)
    cdef void gradient(self, double[::1] Y, double u, double[::1] grad)
    cdef double iterative_next(self, double[::1] Y, double u)
    

@cython.final
cdef class PenaltyAverage(Penalty):
    pass

@cython.final
cdef class PenaltyScale(Penalty):
    pass

#############################################################

cdef class Average:
    cdef public Penalty penalty
    cdef public double tol
    cdef public int n_iter, m_iter, K, L
    cdef public double h
    cdef public bint success    
    cdef ScalarAverager deriv_averager
    cdef public double u_best
    cdef public double pmin
    cdef public double u
    cdef public double pval
    cdef int m, M
    cdef double u_prev, pval_prev
    cdef bint first
    #cdef double u1, u2, u3, u4    
    #
    #cpdef evaluate(self, double[::1] Y)
    cdef gradient(self, double[::1] Y, double[::1] grad)
    #    
    cdef init(self, double[::1] Y, u0=*)
    #
    cpdef fit(self, double[::1] Y, u0=*)
    #
    #cdef c_fit(self, double[::1] Y)
    #
    cdef fit_epoch(self, double[::1] Y)
    #
    cdef bint stop_condition(self)

@cython.final
cdef class Average_Iterative(Average):
    pass

@cython.final
cdef class Average_FG(Average):
    #
    pass
    
cdef class ParametrizedAverage(Average):
    cdef ParametrizedFunc func
    cdef Average avr

@cython.final
cdef class MWAverage(Average):
    cdef Average avr
    cdef double beta

@cython.final
cdef class MWAverage2(Average):
    cdef Average avr

@cython.final
cdef class MHAverage(Average):
    cdef Average avr
    cdef double[::1] Z
    
@cython.final
cdef class ArithMean(Average):
    pass

@cython.final
cdef class KolmogorovMean(Average):
    cdef Func func, invfunc
    cdef double uu

@cython.final
cdef class SoftMinimal(Average):
    cdef double a
    
@cython.final
cdef class Minimal(Average):
    pass

@cython.final
cdef class Maximal(Average):
    pass
    

