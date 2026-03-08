
cdef class Logistic(Func):

    def __init__(self, p=1.0):
        self.p = p
    #
    @cython.final
    @cython.cdivision(True)
    cdef double _evaluate(self, const double x) noexcept nogil:
        cdef double v =  x / self.p
        if v >= 0:
            return 1 / (1 + exp(-v))
        else:
            return 1 - 1 / (1 + exp(v))
    #
    @cython.final
    @cython.cdivision(True)
    cdef double _derivative(self, const double x) noexcept nogil:
        cdef double v =  x / self.p, v1, v2
        if v >= 0:
            v1 = exp(-v)
        else:
            v1 = exp(v)
        v2 = v1 + 1
        return v1 / (v2 * v2) / self.p
    #
    cpdef set_param(self, name, val):
        if name == "sigma":
            self.p = val
        else:
            raise NameError(name)

    cpdef get_param(self, name):
        if name == "sigma":
            return self.p
        else:
            raise NameError(name)
