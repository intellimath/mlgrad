cimport cython

from mlgrad.gd cimport GD

from mlgrad.risk cimport Risk, Functional

from libc.math cimport fabs, pow, sqrt, fmax
from libc.string cimport memcpy, memset

cdef extern from "Python.h":
    double PyFloat_GetMax()
    double PyFloat_GetMin()

cdef inline double min3(double v1, double v2, double v3):
    cdef double vmin = v1
    if v2 < vmin:
        vmin = v2
    if v3 < vmin:
        vmin = v3
    return vmin

cpdef StopCondition get_stop_condition(object)

cdef class StopCondition:
    #
    cdef init(self)
    #
    cdef bint verify(self)
    #
    cdef void finalize(self)

@cython.final
cdef class DiffL1StopCondition(StopCondition):
    cdef GD gd
    cdef double lval1
    
@cython.final
cdef class DiffL2StopCondition(StopCondition):
    cdef GD gd
    cdef double lval1, lval2

@cython.final
cdef class DiffG1StopCondition(StopCondition):
    cdef GD gd
    cdef double[::1] grad

@cython.final
cdef class DiffP1StopCondition(StopCondition):
    cdef GD gd
    cdef double[::1] param
