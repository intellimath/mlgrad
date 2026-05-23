from libc.math cimport fabs, pow, sqrt, fmax

cdef class Projector:
    cdef _project(self, Parameterized mod):
        pass
    #
    def project(self, param):
        self._project(param)

cdef class Func2Projector(Projector):
    #
    def __init__(self, Func2 func, double C):
        self.func = func
        self.C = C
    #
    cdef _project(self, Parameterized mod):
        self.func._normalize(mod.param)
        self.func._scale(self.C)

cdef class LinearModelProjector(Projector):
    def __init__(self, offset=0):
        self.offset = offset

    cdef _project(self, Parameterized mod):
        cdef double[::1] param = mod.param
        cdef Py_ssize_t i, n = param.shape[0]
        cdef double v, s

        s = 0
        for i in range(self.offset, n):
            v = param[i]
            s += v*v
        s = sqrt(s)

        for i in range(n):
            param[i] /= s

cdef class LinearModelPositive(Projector):
    #
    def __init__(self, offset=0):
        self.offset = offset

    cdef _project(self, Parameterized mod):
        cdef double[::1] param = mod.param
        cdef Py_ssize_t i, n = param.shape[0]
        cdef double v

        for i in range(self.offset, param.shape[0]):
            v = param[i]
            if v < 0:
                param[i] = 0

cdef class Masked(Projector):
    #
    def __init__(self, Parameterized mod, tol=1.0e-8):
        self.tol = tol
        self.mod = mod
        self.mask = np.zeros(mod.n_param, np.uint8)
    #
    cdef _project(self, Parameterized mod):
        cdef Py_ssize_t i
        cdef double[::1] param = mod.param
        cdef uint8[::1] mask = self.mask
        cdef double v, tol = self.tol

        for i in range(param.shape[0]):
            if mask[i]:
                param[i] = 0
                continue

            v = param[i]
            if fabs(v) < tol:
                param[i] = 0
                mask[i] = 1
