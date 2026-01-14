cython: language_level=3

cimport cython

from mlgrad.models cimport Model

cdef class Normalizer:
    cdef normalize(self, double[::1] param)

@cython.final
cdef class LinearModelNormalizer(Normalizer):
    cdef Py_ssize_t offset

@cython.final
cdef class LinearModelPositive(Normalizer):
    cdef Py_ssize_t offset

@cython.final
cdef class Masked(Normalizer):
    cdef double tol
    cdef uint8[::1] mask