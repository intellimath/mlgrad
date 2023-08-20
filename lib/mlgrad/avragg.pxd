# cython: language_level=3

cimport cython

from mlgrad.func cimport Func, ParameterizedFunc
from mlgrad.func2 cimport SoftMin
from mlgrad.averager cimport ScalarAverager

# from mlgrad.miscfuncs cimport init_rand, rand, fill

ctypedef double (*FuncEvaluate)(Func, double) nogil
ctypedef double (*FuncDerivative)(Func, double) nogil
ctypedef double (*FuncDerivative2)(Func, double) nogil
ctypedef double (*FuncDerivativeDivX)(Func, double) nogil

ctypedef fused number:
    float
    double
#     double complex
#     double complex

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

cdef inline void array_add_scalar(double[::1] arr, const double v):
    cdef Py_ssize_t i, N = arr.shape[0]

    for i in range(N):
        arr[i] += v

cdef class Penalty:
    cdef readonly Func func

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
    cdef readonly double tol
    cdef readonly Py_ssize_t n_iter 
    cdef readonly Py_ssize_t K 
    cdef public bint success    
    cdef public double u
    cdef public double pval
    cdef bint evaluated
    #
    cdef double init_u(self, double[::1] Y)
    #
    cdef double _evaluate(self, double[::1] Y)
    cdef _gradient(self, double[::1] Y, double[::1] grad)
    cdef _weights(self, double[::1] Y, double[::1] weights)
    #    
    cpdef fit(self, double[::1] Y)
    #
    # cdef bint stop_condition(self)

cdef class AverageIterative(Average):
    cdef readonly Penalty penalty
    
# @cython.final
cdef class MAverage(AverageIterative):
    cdef Func func
    #
    # cdef double evaluate_next_u(self, double[::1] Y, const double u)
    # cdef double evaluate_penalty(self, double[::1] Y, const double u)

# @cython.final
cdef class SAverage(AverageIterative):
    cdef Func func
    
# @cython.final
# cdef class Average_Iterative(Average):
#     pass

# @cython.final
# cdef class MAverage_Iterative(Average):
#     cdef Func func

# @cython.final
# cdef class Average_FG(Average):
#     #
#     cdef ScalarAverager deriv_averager
    
@cython.final
cdef class ParameterizedAverage(Average):
    cdef ParameterizedFunc func
    cdef Average avr

@cython.final
cdef class WMAverage(Average):
    cdef Average avr
    cdef bint initial

# @cython.final
# cdef class WMAverageMixed(Average):
#     cdef Average avr
#     cdef double gamma
    
@cython.final
cdef class TMAverage(Average):
    cdef Average avr

@cython.final
cdef class HMAverage(Average):
    cdef Average avr
    cdef double[::1] Z
    
@cython.final
cdef class ArithMean(Average):
    pass

@cython.final
cdef class RArithMean(Average):
    cdef Func func

@cython.final
cdef class KolmogorovMean(Average):
    cdef Func func, invfunc
    cdef double uu

@cython.final
cdef class SoftMinimal(Average):
    cdef SoftMin softmin
    
@cython.final
cdef class Minimal(Average):
    pass

@cython.final
cdef class Maximal(Average):
    pass
