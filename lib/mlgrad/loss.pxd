# cython: language_level=3

cimport cython

# from libc.math cimport fabs, pow, sqrt, fmax

from mlgrad.func cimport Func, ParameterizedFunc, Square
from mlgrad.model cimport Model, MLModel

cdef extern from "Python.h":
    double PyFloat_GetMax()
    double PyFloat_GetMin()
    
cdef double double_max
cdef double double_min

cdef class Loss(object):
    #
    cdef double _evaluate(self, const double x, const double y) nogil        
    #
    cdef double _derivative(self, const double x, const double y) nogil
    #
    cdef double _derivative_div(self, const double x, const double y) nogil
    #
    cdef double _residual(self, const double x, const double y) nogil

# cdef class WinsorizedLoss(Loss):
#     cdef public ParameterizedFunc wins_func

cdef class LossFunc(Loss):
    cdef Func func

@cython.final
cdef class SquareErrorLoss(LossFunc):
    pass
    #

@cython.final
cdef class ErrorLoss(LossFunc):
    pass
    #

@cython.final
cdef class IdErrorLoss(Loss):
    pass
    #
    
@cython.final
cdef class RelativeErrorLoss(LossFunc):
    pass
    #

@cython.final
cdef class MarginLoss(LossFunc):
    pass
    #

@cython.final
cdef class MLoss(Loss):
   cdef public Func rho
   cdef public Loss loss


# cdef class MultLoss2(object):
#     cdef double _evaluate(self, double[::1] y, double[::1] yk) nogil
#     cdef void _gradient(self, double[::1] y, double[::1] yk, double[::1] grad) nogil

# @cython.final
# cdef class ErrorMultLoss2(MultLoss2):
#     cdef public Func func

# @cython.final
# cdef class MarginMultLoss2(MultLoss2):
#     cdef public Func func
    
cdef class MultLoss:
    #
    cdef double _evaluate(self, double[::1] y, double yk) nogil
    cdef void _gradient(self, double[::1] y, double yk, double[::1] grad) nogil
    

@cython.final
cdef class SoftMinLoss(MultLoss):
    cdef public Loss lossfunc
    cdef public double[::1] vals
    cdef Py_ssize_t q
    cdef double a


cdef class SquareErrorMultiLoss(MultLoss):
    pass
    
cdef class ErrorMultiLoss(MultLoss):
    cdef Func func

cdef class MarginMultiLoss(MultLoss):
    cdef Func func
