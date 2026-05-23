cython: language_level=3

cimport cython

from mlgrad.models cimport Model, Parameterized
from mlgrad.funcs cimport Func

from numpy cimport npy_uint8 as uint8

cdef class Projector:
    cdef _project(self, Parameterized mod)

cdef class Func2Projector(Projector):
    cdef public double C

@cython.final
cdef class LinearModelProjector(Projector):
    cdef Py_ssize_t offset

@cython.final
cdef class LinearModelPositive(Projector):
    cdef Py_ssize_t offset

@cython.final
cdef class Masked(Projector):
    cdef public Parameterized mod
    cdef public double tol
    cdef public uint8[::1] mask