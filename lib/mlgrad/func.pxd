
# cython: language_level=3

cimport cython

cdef class Func(object):
    cdef public unicode label
    #
    cdef float evaluate(self, const float x) nogil
    cdef float derivative(self, const float x) nogil
    cdef float derivative2(self, const float x) nogil
    cdef float derivative_div_x(self, const float x) nogil
    cdef float evaluate_array(self, const float *x, float *y, const Py_ssize_t n) nogil
    cdef float derivative_array(self, const float *x, float *y, const Py_ssize_t n) nogil
    cdef float derivative2_array(self, const float *x, float *y, const Py_ssize_t n) nogil
    cdef float derivative_div_x_array(self, const float *x, float *y, const Py_ssize_t n) nogil
    #

@cython.final
cdef class Comp(Func):
    #
    cdef public Func f, g
    #

@cython.final
cdef class CompSqrt(Func):
    #
    cdef public Func f, g
    #

@cython.final
cdef class ZeroOnPositive(Func):
    #
    cdef public Func f

@cython.final
cdef class FuncExp(Func):
    cdef public Func f
    
@cython.final
cdef class Id(Func):
    #
    pass

@cython.final
cdef class Neg(Func):
    #
    pass

@cython.final
cdef class Arctang(Func):
    cdef public float a

@cython.final
cdef class Sigmoidal(Func):
    cdef public float p

@cython.final
cdef class ModSigmoidal(Func):
    cdef public float a

@cython.final
cdef class Softplus(Func):
    cdef public float a
    cdef float log_a

@cython.final
cdef class Threshold(Func):
    cdef public float theta

@cython.final
cdef class Sign(Func):
    cdef public float theta

@cython.final
cdef class Quantile(Func):
    #
    cdef public float alpha
    #

@cython.final
cdef class QuantileFunc(Func):
    cdef public float alpha
    cdef public Func f
    
@cython.final
cdef class Expectile(Func):
    #
    cdef public float alpha
    #

@cython.final
cdef class Power(Func):
    #
    cdef public float alpha, p, alpha_p
    #

@cython.final
cdef class Square(Func):
    #
    pass

@cython.final
cdef class SquareSigned(Func):
    #
    pass

@cython.final
cdef class Absolute(Func):
    #
    pass

@cython.final
cdef class Quantile_AlphaLog(Func):
    #
    cdef public float alpha
    cdef float alpha2, q
    #
    
@cython.final
cdef class SoftAbs(Func):
    #
    cdef float eps
    
@cython.final
cdef class Sqrt(Func):
    #
    cdef float eps
    cdef float eps2
    cdef float alpha
    cdef float zero
    #
    
@cython.final
cdef class Quantile_Sqrt(Func):
    #
    cdef float eps
    cdef float eps2
    cdef float alpha
    #

@cython.final
cdef class Logistic(Func):
    #
    cdef public float p
    #

@cython.final
cdef class Huber(Func):
    #
    cdef public float C
    #

@cython.final
cdef class TM(Func):
    #
    cdef public float a

@cython.final
cdef class LogSquare(Func):
    #
    cdef public float a
    cdef public float a2
    
@cython.final
cdef class Tukey(Func):
    #
    cdef public float C
    cdef float C2
    #

@cython.final
cdef class Hinge(Func):
    #
    cdef public float C
    #

@cython.final
cdef class HingeSqrt(Func):
    #
    cdef public float alpha
    cdef float alpha2
    #

@cython.final
cdef class  Exp(Func):
    #
    cdef public float alpha

@cython.final
cdef class  Log(Func):
    #
    cdef public float alpha
    
@cython.final
cdef class KMinSquare(Func):
    #
    cdef float[::1] c
    cdef int n_dim, j_min
    
cdef class ParameterizedFunc:
    #
    cdef float evaluate(self, float x, float u) nogil
    #
    cdef float derivative(self, float x, float u) nogil
    #
    cdef float derivative_u(self, float x, float u) nogil

@cython.final
cdef class WinsorizedFunc(ParameterizedFunc):
    pass

@cython.final
cdef class WinsorizedSmoothFunc(ParameterizedFunc):
    cdef Func f

cdef class SoftMinFunc(ParameterizedFunc):
    cdef float a

    