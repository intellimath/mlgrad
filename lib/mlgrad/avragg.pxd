# cython: language_level=3

cimport cython

from mlgrad.func cimport Func, ParameterizedFunc
from mlgrad.averager cimport ScalarAverager

# from mlgrad.miscfuncs cimport init_rand, rand, fill

# from libc.math cimport fabs, pow, sqrt, fmax, log, exp
from libc.math cimport fabsf, powf, sqrtf, fmaxf, logf, expf

ctypedef float (*FuncEvaluate)(Func, float) nogil
ctypedef float (*FuncDerivative)(Func, float) nogil
ctypedef float (*FuncDerivative2)(Func, float) nogil
ctypedef float (*FuncDerivativeDivX)(Func, float) nogil

ctypedef fused number:
    float
    double
#     float complex
#     float complex

cdef extern from "Python.h":
    float PyFloat_GetMax()

cdef extern from "pymath.h" nogil:
    bint Py_IS_FINITE(float x)
    bint Py_IS_INFINITY(float x)
    bint Py_IS_NAN(float x)
    bint copysign(float x, float x)

cdef inline float array_min(float[::1] arr):
    cdef Py_ssize_t i, N = arr.shape[0]
    cdef float v, min_val = arr[0]

    for i in range(N):
        v = arr[i]
        if v < min_val:
            min_val = v

    return min_val

cdef inline float array_mean(float[::1] arr):
    cdef Py_ssize_t i, N = arr.shape[0]
    cdef float v

    v = 0
    for i in range(N):
        v += arr[i]

    return v / N

cdef inline void array_add_scalar(float[::1] arr, const float v):
    cdef Py_ssize_t i, N = arr.shape[0]

    for i in range(N):
        arr[i] += v

cdef class Penalty:
    cdef readonly Func func

    cdef float evaluate(self, float[::1] Y, float u)
    cdef float derivative(self, float[::1] Y, float u)
    cdef void gradient(self, float[::1] Y, float u, float[::1] grad)
    cdef float iterative_next(self, float[::1] Y, float u)
    

@cython.final
cdef class PenaltyAverage(Penalty):
    cdef float *temp


@cython.final
cdef class PenaltyScale(Penalty):
    cdef float *temp


#############################################################

cdef class Average:
    cdef readonly Penalty penalty
    cdef readonly float tol
    cdef readonly int n_iter 
    cdef readonly int K 
    cdef public int m_iter, L
    cdef public float h
    cdef public bint success    
    cdef public float u_best
    cdef public float pmin
    cdef public float u
    cdef public float pval
    cdef int m, M
    cdef float u_prev, pval_prev
    cdef bint first
    #cdef float u1, u2, u3, u4    
    #
    #cpdef evaluate(self, float[::1] Y)
    cdef gradient(self, float[::1] Y, float[::1] grad)
    #    
    cdef init(self, float[::1] Y, u0=*)
    #
    cpdef fit(self, float[::1] Y, u0=*)
    #
    #cdef c_fit(self, float[::1] Y)
    #
    cdef fit_epoch(self, float[::1] Y)
    #
    cdef bint stop_condition(self)

@cython.final
cdef class Average_Iterative(Average):
    pass

@cython.final
cdef class MAverage_Iterative(Average):
    cdef Func func

@cython.final
cdef class Average_FG(Average):
    #
    cdef ScalarAverager deriv_averager
    
@cython.final
cdef class ParameterizedAverage(Average):
    cdef ParameterizedFunc func
    cdef Average avr

@cython.final
cdef class WMAverage(Average):
    cdef Average avr
    cdef bint initial

@cython.final
cdef class WMAverageMixed(Average):
    cdef Average avr
    cdef float gamma
    
@cython.final
cdef class TMAverage(Average):
    cdef Average avr

@cython.final
cdef class HMAverage(Average):
    cdef Average avr
    cdef float[::1] Z
    
@cython.final
cdef class ArithMean(Average):
    pass

@cython.final
cdef class KolmogorovMean(Average):
    cdef Func func, invfunc
    cdef float uu

@cython.final
cdef class SoftMinimal(Average):
    cdef float a
    
@cython.final
cdef class Minimal(Average):
    pass

@cython.final
cdef class Maximal(Average):
    pass
    

