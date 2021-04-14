
# cython: language_level=3

cimport cython

cdef class Func(object):
    cdef public unicode label
    #
    cdef double evaluate(self, const double x) nogil
    cdef double derivative(self, const double x) nogil
    cdef double derivative2(self, const double x) nogil
    cdef double derivative_div_x(self, const double x) nogil
    #

@cython.final
cdef class Comp(Func):
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
    cdef public double a

@cython.final
cdef class Sigmoidal(Func):
    cdef public double p

@cython.final
cdef class ModSigmoidal(Func):
    cdef public double a

@cython.final
cdef class Softplus(Func):
    cdef public double a
    cdef double log_a

@cython.final
cdef class Threshold(Func):
    cdef public double theta

@cython.final
cdef class Sign(Func):
    cdef public double theta

@cython.final
cdef class Quantile(Func):
    #
    cdef public double alpha
    #

@cython.final
cdef class QuantileFunc(Func):
    cdef public double alpha
    cdef public Func f
    
@cython.final
cdef class Expectile(Func):
    #
    cdef public double alpha
    #

@cython.final
cdef class Power(Func):
    #
    cdef public double alpha, p, alpha_p
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
    cdef public double alpha
    cdef double alpha2, q
    #
    
@cython.final
cdef class SoftAbs(Func):
    #
    cdef double eps
    
@cython.final
cdef class Sqrt(Func):
    #
    cdef double eps
    cdef double eps2
    cdef double alpha
    cdef double zero
    #
    
@cython.final
cdef class Quantile_Sqrt(Func):
    #
    cdef double eps
    cdef double eps2
    cdef double alpha
    #

@cython.final
cdef class Logistic(Func):
    #
    cdef public double p
    #

@cython.final
cdef class Huber(Func):
    #
    cdef public double C
    #

@cython.final
cdef class TM(Func):
    #
    cdef public double a

@cython.final
cdef class LogSquare(Func):
    #
    cdef public double a
    cdef public double a2
    
@cython.final
cdef class Tukey(Func):
    #
    cdef public double C
    cdef double C2
    #

@cython.final
cdef class Hinge(Func):
    #
    cdef public double C
    #

@cython.final
cdef class HingeSqrt(Func):
    #
    cdef public double alpha
    cdef double alpha2
    #

@cython.final
cdef class  Exp(Func):
    #
    cdef public double alpha

@cython.final
cdef class  Log(Func):
    #
    cdef public double alpha
    
@cython.final
cdef class KMinSquare(Func):
    #
    cdef double[::1] c
    cdef int n_dim, j_min
    
cdef class ParameterizedFunc:
    #
    cdef double evaluate(self, double x, double u) nogil
    #
    cdef double derivative(self, double x, double u) nogil
    #
    cdef double derivative_u(self, double x, double u) nogil

@cython.final
cdef class WinsorizedFunc(ParameterizedFunc):
    pass

@cython.final
cdef class WinsorizedSmoothFunc(ParameterizedFunc):
    cdef Func f

cdef class SoftMinFunc(ParameterizedFunc):
    cdef double a

    