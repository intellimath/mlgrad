cimport cython

from mlgrad.gd cimport GD

from mlgrad.risk cimport Risk, Functional

from libc.math cimport fabsf, pow, sqrt, fmax
# from libc.math cimport fabsff, powf, sqrtf, fmaxf
from libc.string cimport memcpy, memset

cdef extern from "Python.h":
    float PyFloat_GetMax()
    float PyFloat_GetMin()

cdef inline float min3(const float v1, const float v2, const float v3):
    cdef float vmin = v1
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
#     cdef void finalize(self)

@cython.final
cdef class DiffL1StopCondition(StopCondition):
    cdef GD gd
#     cdef float lval
    cdef float lval_min
    
@cython.final
cdef class DiffL2StopCondition(StopCondition):
    cdef GD gd
    cdef float lval1, lval2

@cython.final
cdef class DiffG1StopCondition(StopCondition):
    cdef GD gd
    cdef float[::1] grad

@cython.final
cdef class DiffP1StopCondition(StopCondition):
    cdef GD gd
    cdef float[::1] param
