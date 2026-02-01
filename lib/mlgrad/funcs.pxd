
# cython: language_level=3

cimport cython

from libc.math cimport pow, sqrt, exp, log, atan, fma, sinh, asinh, cosh, tanh, pi
from libc.math cimport isnan, isinf
from libc.stdlib cimport strtod

cdef inline double fabs(double x) noexcept nogil:
    if x >= 0:
        return x
    else:
        return -x

cdef inline double fmax(double x, double y) noexcept nogil:
    if x > y:
        return x
    else:
        return y


cdef class Func(object):
    cdef public unicode label
    #
    cdef double _evaluate(self, const double x) noexcept nogil
    cdef double _inverse(self, const double x) noexcept nogil
    cdef double _derivative(self, const double x) noexcept nogil
    cdef double _derivative2(self, const double x) noexcept nogil
    cdef double _derivative_div(self, const double x) noexcept nogil
    cdef double _value(self, const double x) noexcept nogil

    cdef void _evaluate_array(self, const double *x, double *y, const Py_ssize_t n) noexcept nogil
    cdef double _evaluate_sum(self, const double *x, const Py_ssize_t n) noexcept nogil    
    cdef double _evaluate_weighted_sum(self, const double *x, const double *w, const Py_ssize_t n) noexcept nogil    
    cdef void _derivative_array(self, const double *x, double *y, const Py_ssize_t n) noexcept nogil
    cdef void _derivative_weighted_sum(self, const double *x, double *y, const double *w, const Py_ssize_t n) noexcept nogil
    cdef void _derivative2_array(self, const double *x, double *y, const Py_ssize_t n) noexcept nogil
    cdef void _derivative_div_array(self, const double *x, double *y, const Py_ssize_t n) noexcept nogil
    cdef void _value_array(self, const double *x, double *y, const Py_ssize_t n) noexcept nogil
    cdef void _inverse_array(self, double *x, double *y, Py_ssize_t n) noexcept nogil
    #
    cpdef set_param(self, name, val)
    cpdef get_param(self, name)
    #

cdef class Func_sigma(Func):
    cdef public sigma
    #

cdef class PyFunc(Func):
    cdef public object pyfunc
    #

cdef class ParameterizedFunc:
    #
    cdef double _evaluate(self, double x, double u) noexcept nogil
    #
    cdef double _derivative(self, double x, double u) noexcept nogil
    #
    cdef double derivative_u(self, double x, double u) noexcept nogil
    

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
cdef class Gauss(Func):
    cdef public double scale, scale2

@cython.final
cdef class GaussSuppl(Func):
    cdef public double scale

@cython.final
cdef class DArctg(Func):
    cdef public double a

@cython.final
cdef class Linear(Func):
    cdef public double a, b

@cython.final
cdef class LogGauss2(Func):
    cdef public double w, c, scale

@cython.final
cdef class ZeroOnPositive(Func):
    #
    cdef public Func f

@cython.final
cdef class ZeroOnNegative(Func):
    #
    cdef public Func f

cdef class TruncAbs(Func):
    #
    cdef public double c
    
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
cdef class PlusId(Func):
    #
    pass

@cython.final
cdef class SquarePlus(Func):
    pass


@cython.final
cdef class Arctang(Func):
    cdef public double a

@cython.final
cdef class Sigmoidal(Func):
    cdef public double p

@cython.final
cdef class DOOM(Func):
    cdef public double p

@cython.final
cdef class ModSigmoidal(Func):
    cdef public double a

@cython.final
cdef class SoftPlus(Func):
    cdef public double a
    cdef double log_a

@cython.final
cdef class Threshold(Func):
    cdef public double theta
    cdef public double eps

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
    cdef public double p, p1, alpha, alpha_p
    #

@cython.final
cdef class Square(Func):
    pass

@cython.final
cdef class SquareSigned(Func):
    pass

@cython.final
cdef class Abs(Func):
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
    cdef public double eps

@cython.final
cdef class SoftAbs_Sqrt(Func):
    #
    cdef public double eps
    cdef double eps2
    #

@cython.final
cdef class SoftAbs_Exp(Func):
    #
    cdef public double eps
    cdef double eps1
    #

@cython.final
cdef class SoftAbs_Power(Func):
    #
    cdef public double eps
    cdef double eps2
    cdef double q, q1
    #

@cython.final
cdef class Quantile_Sqrt(Func):
    #
    cdef double eps
    cdef double eps2
    cdef double alpha

@cython.final
cdef class Expit(Func):
    #
    cdef public double p, x0

@cython.final
cdef class Logistic(Func):
    #
    cdef public double p

@cython.final
cdef class Huber(Func):
    #
    cdef public double C

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
cdef class Step(Func):
    #
    cdef public double C
    cdef public double eps
    #

@cython.final
cdef class StepRight(Func):
    #
    cdef public double C
    cdef public double eps
    #

@cython.final
cdef class Step_Sqrt(Func):
    #
    cdef public double eps
    #

@cython.final
cdef class Step_Exp(Func):
    cdef public double p

@cython.final
cdef class RectExp(Func):
    cdef public double w
    cdef public double p
        
@cython.final
cdef class Hinge(Func):
    #
    cdef public double C
    #

@cython.final
cdef class Hinge2(Func):
    #
    cdef public double C
    #

@cython.final
cdef class Square2Linear(Func):
    #
    cdef public double C

@cython.final
cdef class RELU(Func):
    #
    pass
    
@cython.final
cdef class HSquare(Func):
    #
    cdef public double C
    #

@cython.final
cdef class IntSoftHinge_Atan(Func):
    #
    cdef public double alpha
    cdef public double x0
    #
    
@cython.final
cdef class SoftHinge_Sqrt(Func):
    #
    cdef public double alpha
    cdef double alpha2
    cdef public double x0
    #

@cython.final
cdef class SoftHinge_Exp(Func):
    cdef public double alpha
    cdef public double x0


@cython.final
cdef class IntSoftHinge_Sqrt(Func):
    #
    cdef public double alpha
    cdef double alpha2
    cdef public double x0
    #
    
@cython.final
cdef class Softplus_Sqrt(Func):
    #
    cdef public double alpha
    cdef double alpha2
    cdef double x0
    #
    
@cython.final
cdef class  Exp(Func):
    #
    cdef public double p

@cython.final
cdef class  Log(Func):
    #
    cdef public double alpha
    
@cython.final
cdef class KMinSquare(Func):
    #
    cdef double[::1] c
    cdef int n_dim, j_min
    
@cython.final
cdef class WinsorizedFunc(ParameterizedFunc):
    pass

@cython.final
cdef class WinsorizedSmoothFunc(ParameterizedFunc):
    cdef Func f

cdef class SoftMinFunc(ParameterizedFunc):
    cdef double a

cdef class RelativeAbsMax(Func):
    pass
