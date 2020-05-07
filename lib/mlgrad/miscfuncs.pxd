# cython: language_level=3

cdef void init_rand()
cdef long rand(long N)

cdef extern from "pymath.h" nogil:
    bint Py_IS_FINITE(double x)
    bint Py_IS_INFINITY(double x)
    bint Py_IS_NAN(double x)
    bint copysign(double x, double x)

cdef double absmax_1d(double[:] X)

#cdef double[:] as_memoryview_1d(object X)

cdef void multiply_scalar(double[:] X, double c)

cdef void fill(double[:] X, double c)
