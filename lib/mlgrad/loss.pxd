# cython: language_level=3

cimport cython

#from libc.math cimport fabs, pow, sqrt, fmax

from mlgrad.func cimport Func, ParameterizedFunc

cdef extern from "Python.h":
    double PyFloat_GetMax()
    double PyFloat_GetMin()
    
cdef double float_min = PyFloat_GetMax()

cdef class Loss(object):
    #
    cdef double evaluate(self, const double x, const double y) nogil        
    #
    cdef double derivative(self, const double x, const double y) nogil
    #
    cdef double difference(self, const double x, const double y) nogil

# cdef class WinsorizedLoss(Loss):
#     cdef public ParameterizedFunc wins_func

@cython.final
cdef class SquareErrorLoss(Loss):
    pass
    #

@cython.final
cdef class ErrorLoss(Loss):
    cdef public Func func
    #

@cython.final
cdef class RelativeErrorLoss(Loss):
    cdef public Func func
    cdef double eps
    #

@cython.final
cdef class MarginLoss(Loss):
    cdef public Func func
    #

@cython.final
cdef class MLoss(Loss):
   cdef public Func rho
   cdef public Loss loss


cdef class MultLoss(object):
    cdef double evaluate(self, double[::1] y, double[::1] yk) nogil
    cdef void gradient(self, double[::1] y, double[::1] yk, double[::1] grad) nogil

@cython.final
cdef class ErrorMultLoss(MultLoss):
    cdef public Func func

@cython.final
cdef class MarginMultLoss(MultLoss):
    cdef public Func func

cdef class MinLoss:
    cdef public Loss loss
    cdef double val_min

    cdef double evaluate(self, double[::1] y, double yk) nogil
    cdef void gradient(self, double[::1] y, double yk, double[::1] grad) nogil
