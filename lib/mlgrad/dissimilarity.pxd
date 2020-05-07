# cython: language_level=3

from mlgrad.func cimport Func

cdef class Dissimilarity(object):
    #
    cdef Func func
    cdef bint with_scale
    #
    cdef double evaluate(self, double z, double u)
    #
    cdef double derivative_u(self, double z, double u)
    #
    cdef double derivative_z(self, double z, double u)
    #
    cdef double derivative_uz(self, double z, double u)
    #
    cdef double derivative_uu(self, double z, double u)

