# cython: language_level=3

cimport cython

cdef class Distance:    
    cdef float _evaluate(self, float *x, float *y, Py_ssize_t n) nogil
    cdef float evaluate(self, float[::1] x, float[::1] y) nogil
    cdef void gradient(self, float[::1] x, float[::1] y, float[::1]) nogil
    cdef set_param(self, name, val)

cdef class DistanceWithScale(Distance):
    cdef public float[:,::1] S
    cdef public float sigma
    
@cython.final
cdef class AbsoluteDistance(Distance):
    pass

@cython.final
cdef class EuclidDistance(Distance):
    pass

@cython.final
cdef class MahalanobisDistance(DistanceWithScale):
    pass

@cython.final
cdef class PowerDistance(Distance):
    cdef float p

