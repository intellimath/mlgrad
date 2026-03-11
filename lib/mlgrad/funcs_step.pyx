
@cython.final
cdef class RStep(Func):
    #
    def __init__(self, delta=0, eps=0):
        self.delta = delta
        self.eps = eps
    #
    @cython.final
    cdef double _evaluate(self, const double x) noexcept nogil:
        cdef double delta = self.delta
        if x > delta:
            return self.eps
        elif x < -delta:
            return 1 + self.eps
        elif delta == 0:
            return 0.5 + self.eps
        else:
            return (1 - x/delta)/2 + self.eps
    #
    @cython.final
    cdef double _derivative(self, const double x) noexcept nogil:
        if x >= self.delta or x <= -self.delta:
            return 0
        else:
            return -0.5/self.delta
    #
    cpdef set_param(self, name, val):
        if name == "sigma":
            self.delta = val
        elif name == "eps":
            self.eps = val
        else:
            raise NameError(name)

    cpdef get_param(self, name):
        if name == "delta":
            return self.delta
        elif name == "eps":
            return self.eps
        else:
            raise NameError(name)

@cython.final
cdef class Step(Func):
    #
    def __init__(self, delta=0, eps=0):
        self.delta = delta
        self.eps =  eps
    #
    @cython.final
    cdef double _evaluate(self, const double x) noexcept nogil:
        cdef double delta = self.delta
        if x >= delta:
            return 1 + self.eps
        elif x < -delta:
            return self.eps
        elif delta == 0:
            return 0.5 + self.eps
        else:
            return (1 + x/delta)/2 + self.eps
    #
    @cython.final
    cdef double _derivative(self, const double x) noexcept nogil:
        if x >= self.delta or x <= -self.delta:
            return 0
        else:
            return 0.5/self.delta
    #
    cpdef set_param(self, name, val):
        if name == "delta":
            self.delta = val
        elif name == "eps":
            self.eps = val
        else:
            raise NameError(name)

    cpdef get_param(self, name):
        if name == "delta":
            return self.delta
        elif name == "eps":
            return self.eps
        else:
            raise NameError(name)

@cython.final
cdef class RStep_Sqrt(Func):
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
cdef class RStep_Exp(Func):
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

@cython.final
cdef class RStep_Gauss(Func):
    #
    def __init__(self, sigma=1.0):
        self.sigma = sigma
        self.sigma2 = sigma*sigma
    #
    @cython.final
    cdef double _evaluate(self, const double x) noexcept nogil:
        cdef double v = x / self.sigma
        cdef double vv = 0.5*exp(v)
        if x >= 0:
            return 0.5*exp(v)
        else:
            return 1 - 0.5*exp(-v)
    #
