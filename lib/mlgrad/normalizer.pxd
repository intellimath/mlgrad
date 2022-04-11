# cython: language_level=3

# cimport cython

cdef class Normalizer:
    cdef normalize(self, double[::1] param)

@cython.final
cdef class LinearModelNormalizer(Normalizer):
    pass
