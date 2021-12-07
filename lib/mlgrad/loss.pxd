# cython: language_level=3

cimport cython

from libc.math cimport fabsf, powf, sqrtf, fmaxf

from mlgrad.func cimport Func, ParameterizedFunc

cdef extern from "Python.h":
    float PyFloat_GetMax()
    float PyFloat_GetMin()
    
cdef float float_max
cdef float float_min

cdef class Loss(object):
    #
    cdef float evaluate(self, const float x, const float y) nogil        
    #
    cdef float derivative(self, const float x, const float y) nogil
    #
    cdef float difference(self, const float x, const float y) nogil

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
    cdef float evaluate(self, float[::1] y, float[::1] yk) nogil
    cdef void gradient(self, float[::1] y, float[::1] yk, float[::1] grad) nogil

@cython.final
cdef class ErrorMultLoss(MultLoss):
    cdef public Func func

@cython.final
cdef class MarginMultLoss(MultLoss):
    cdef public Func func
    
cdef class MultLoss2:
    #
    cdef float evaluate(self, float[::1] y, float yk) nogil
    cdef void gradient(self, float[::1] y, float yk, float[::1] grad) nogil
    

@cython.final
cdef class SoftMinLoss2(MultLoss2):
    cdef public Loss lossfunc
#     cdef public float val_min
    cdef public float[::1] vals
    cdef Py_ssize_t q
    cdef float a

#     cdef float evaluate(self, float[::1] y, float yk) nogil
#     cdef void gradient(self, float[::1] y, float yk, float[::1] grad) nogil
