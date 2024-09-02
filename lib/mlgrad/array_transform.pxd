
from libc.math cimport fabs, pow, sqrt, fmax
cimport mlgrad.inventory as inventory
cimport numpy
cimport cython

cdef void _array_zscore(double *a, double *b, Py_ssize_t n)
cdef void _array_modified_zscore(double *a, double *b, Py_ssize_t n)


# cdef class ArrayTransformer:
#     cdef transform(self, double[::1] a)


# cdef class ArrayNormalizer2(ArrayTransformer):
#     cdef Py_ssize_t i0
#     cdef transform(self, double[::1] a)
