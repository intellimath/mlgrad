# cython: language_level=3

cimport cython

# from libc.math cimport fabs, pow, sqrt, fmax

from mlgrad.func cimport Func, ParameterizedFunc

cdef extern from "Python.h":
    double PyFloat_GetMax()
    double PyFloat_GetMin()
    
cdef double double_max
cdef double double_min

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
cdef class IdErrorLoss(Loss):
    pass
    #
    
@cython.final
cdef class RelativeErrorLoss(Loss):
    cdef public Func func
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
    
cdef class MultLoss2:
    #
    cdef double evaluate(self, double[::1] y, double yk) nogil
    cdef void gradient(self, double[::1] y, double yk, double[::1] grad) nogil
    

@cython.final
cdef class SoftMinLoss2(MultLoss2):
    cdef public Loss lossfunc
#     cdef public double val_min
    cdef public double[::1] vals
    cdef Py_ssize_t q
    cdef double a

#     cdef double evaluate(self, double[::1] y, double yk) nogil
#     cdef void gradient(self, double[::1] y, double yk, double[::1] grad) nogil
