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

cdef class Masked(Normalizer):
    def __init__(self, Model mod, tol=1.0e-8):
        self.tol = tol
        self.model = mod
        mod.mask = np.zeros(mod.n_param, np.uint8)
    #
    cdef normalize(self, double[::1] param):
        cdef Py_ssize_t i
        cdef uint8[::1] mask = self.model.mask
        cdef double v, tol = self.tol
        cdef uint8 mm

        for i in range(param.shape[0]):
            mm = mask[i]
            if mm == 1:
                param[i] = 0
                continue

            v = param[i]
            if fabs(v) < tol:
                param[i] = 0
                mask[i] = 1
                
        
            