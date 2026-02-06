
@cython.final
cdef class Step(Func):
    #
    def __init__(self, C=0, eps=0):
        self.C = C
        self.eps = eps
    #
    @cython.final
    cdef double _evaluate(self, const double x) noexcept nogil:
        cdef double C = self.C
        if x >= C:
            return self.eps
        elif x < -C:
            return 1 + self.eps
        else:
            return (1 - x/C)/2 + self.eps
    #
    @cython.final
    cdef double _derivative(self, const double x) noexcept nogil:
        if x >= self.C or x <= -self.C:
            return 0
        else:
            return -0.5/self.C
    #
    cpdef set_param(self, name, val):
        if name == "sigma":
            self.C = val
        else:
            raise NameError(name)

    cpdef get_param(self, name):
        if name == "sigma":
            return self.C
        else:
            raise NameError(name)

@cython.final
cdef class StepRight(Func):
    #
    def __init__(self, C=0, eps=0):
        self.C = C
        self.eps =  eps
    #
    @cython.final
    cdef double _evaluate(self, const double x) noexcept nogil:
        cdef double C = self.C
        if x >= C:
            return 1 + self.eps
        elif x < -C:
            return self.eps
        else:
            return (x/C - 1)/2 + self.eps
    #
    @cython.final
    cdef double _derivative(self, const double x) noexcept nogil:
        if x >= self.C or x <= -self.C:
            return 0
        else:
            return 0.5/self.C
    #
    cpdef set_param(self, name, val):
        if name == "sigma":
            self.C = val
        else:
            raise NameError(name)

    cpdef get_param(self, name):
        if name == "sigma":
            return self.C
        else:
            raise NameError(name)

@cython.final
cdef class Step_Sqrt(Func):
    #
    def __init__(self, eps=1.0e-3):
        self.eps = eps
    #
    @cython.final
    cdef double _evaluate(self, const double x) noexcept nogil:
        cdef double eps = self.eps
        return 0.5 * (1 - x / sqrt(eps*eps + x*x))
    #
    @cython.final
    cdef double _derivative(self, const double x) noexcept nogil:
        cdef double eps = self.eps
        cdef double v = eps*eps + x*x
        return -0.5 * eps*eps / (v * sqrt(v))
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

@cython.final
cdef class Step_Exp(Func):
    #
    def __init__(self, p=1.0):
        self.p = p
    #
    @cython.final
    cdef double _evaluate(self, const double x) noexcept nogil:
        if x >= 0:
            return exp(-x / self.p) / self.p
        else:
            return 1
    #
    @cython.final
    cdef double _derivative(self, const double x) noexcept nogil:
        if x >= 0:
            return -exp(-x / self.p)
        else:
            return 0
    #
