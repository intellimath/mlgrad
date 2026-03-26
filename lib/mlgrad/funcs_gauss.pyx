
@cython.final
cdef class GaussSuppl(Func):
    #
    def __init__(self, scale=1.0):
        self.scale = scale
    #
    @cython.final
    cdef double _evaluate(self, const double x) noexcept nogil:
        cdef double a = self.scale
        cdef double v = x / a
        return 1 - exp(-0.5*v*v)
    #
    @cython.final
    cdef double _derivative(self, const double x) noexcept nogil:
        cdef double a = self.scale
        cdef double v = x / a
        return x * exp(-0.5*v*v) / (a*a)
    #
    @cython.final
    cdef double _derivative_div(self, const double x) noexcept nogil:
        cdef double a = self.scale
        cdef double v = x / self.scale
        return exp(-0.5*v*v) / (a*a)
    #
    # @cython.final
    # cdef double _derivative2(self, const double x) noexcept nogil:
    #     cdef double a = self.scale
    #     cdef double v = x / a
    #     cdef double v2 = v * v
    #     return exp(-0.5*v2) * (1 - v2)

@cython.final
cdef class Gauss(Func):
    #
    def __init__(self, scale=1.0):
        self.scale = scale
        self.scale2 = scale * scale
    #
    @cython.final
    cdef double _evaluate(self, const double x) noexcept nogil:
        cdef double v = x / self.scale
        return exp(-0.5*v*v)
    #
    @cython.final
    cdef double _derivative(self, const double x) noexcept nogil:
        cdef double v = x / self.scale
        return -x * exp(-0.5*v*v) / self.scale
    #
    @cython.final
    cdef double _derivative_div(self, const double x) noexcept nogil:
        cdef double v = x / self.scale
        return -exp(-0.5*v*v) /self.scale
    #
    @cython.final
    cdef double _derivative2(self, const double x) noexcept nogil:
        cdef double v = x / self.scale
        cdef double v2 = v * v
        return -exp(-0.5*v2) * (1 - v2) / self.scale2
    #
