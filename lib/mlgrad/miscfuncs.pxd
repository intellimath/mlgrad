# cython: language_level=3

cdef void init_rand()
cdef long rand(long N)

cdef extern from "pymath.h" nogil:
    bint Py_IS_FINITE(float x)
    bint Py_IS_INFINITY(float x)
    bint Py_IS_NAN(float x)
    bint copysign(float x, float x)

cdef float absmax_1d(float[::1] X)

#cdef float[:] as_memoryview_1d(object X)

cdef void multiply_scalar(float[::1] X, float c)

cdef void fill(float[::1] X, float c)
