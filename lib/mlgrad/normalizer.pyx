from libc.math cimport fabs, pow, sqrt, fmax

cdef class Normalizer:
    cdef normalize(self, double[::1] param):
        pass

cdef class LinearModelNormalizer(Normalizer):

    cdef normalize(self, double[::1] param):
        cdef Py_ssize_t i, n = param.shape[0]
        cdef double v, s

        s = 0
        for i in range(1, n):
            v = param[i]
            s += v*v
        s = sqrt(s)

        for i in range(n):
            param[i] /= s

            