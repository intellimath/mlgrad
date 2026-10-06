
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
            return 1
        elif delta == 0:
            return 0.5 + self.eps
        else:
            c = (1 - self.eps) / 2
            return c * (1 - x / delta) + self.eps
    #
    @cython.final
    cdef double _derivative(self, const double x) noexcept nogil:
        if x >= self.delta or x <= -self.delta:
            return 0
        else:
            return -(1 - self.eps)/2 / self.delta
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
    def __init__(self, alpha_l=1.0, alpha_r=0.0, delta=0.0):
        self.delta = delta
        self.alpha_l = alpha_l
        self.alpha_r = alpha_r
    #
    @cython.final
    cdef double _evaluate(self, const double x) noexcept nogil:
        cdef double delta = self.delta
        if x >= delta:
            return self.alpha_r
        elif x < -delta:
            return self.alpha_l
        elif delta == 0:
            return 0.5 * (self.alpha_l + self.alpha_r)
        else:
            return 0.5 * (self.alpha_l + self.alpha_r) + 0.5 * (self.alpha_r - self.alpha_l) * (x / delta)
    #
    @cython.final
    cdef double _derivative(self, const double x) noexcept nogil:
        cdef double delta = self.delta
        if x > delta or x < -delta:
            return 0
        elif delta == 0:
            return c_inf
        else:
            return 0.5 * (self.alpha_r - self.alpha_l) / delta
    #
    cpdef set_param(self, name, val):
        if name == "delta":
            self.delta = val
        elif name == "alpha_l":
            self.alpha_l = val
        elif name == "alpha_r":
            self.alpha_r = val
        else:
            raise NameError(name)

    cpdef get_param(self, name):
        if name == "delta":
            return self.delta
        elif name == "alpha_l":
            return self.alpha_l
        elif name == "alpha_r":
            return self.alpha_r
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

@cython.final
cdef class RQStep(Func):
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
            return 1 - self.eps
        elif delta == 0:
            return 0.5 + self.eps
        else:
            return -(0.5 - self.eps) / delta * x + 0.5
    #
    @cython.final
    cdef double _derivative(self, const double x) noexcept nogil:
        if x >= self.delta or x <= -self.delta:
            return 0
        else:
            return -(0.5 - self.eps) / self.delta
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
