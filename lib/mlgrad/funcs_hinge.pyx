
@cython.final
cdef class Hinge(Func):
    #
    def __init__(self, C=1.0):
        self.C = C
    #
    @cython.final
    cdef double _evaluate(self, const double x) noexcept nogil:
        if x >= self.C:
            return 0
        else:
            return self.C - x
    #
    @cython.final
    cdef double _derivative(self, const double x) noexcept nogil:
        if x > self.C:
            return 0
        elif x == self.C:
            return -0.5
        else:
            return -1
    #
    @cython.final
    cdef double _derivative_div(self, const double x) noexcept nogil:
        if x >= self.C:
            return 0
        else:
            return -1 / (x - self.C)
    #
    @cython.final
    cdef double _derivative2(self, const double x) noexcept nogil:
        return 0
    #
    @cython.final
    cdef double _value(self, const double x) noexcept nogil:
        return self.C - x
    #
    def _repr_latex_(self):
        return r"$ρ(x)=(c-x)_{+}$"

    def to_dict(self):
        return { 'name':'hinge',
                 'args': (self.C,) }

@cython.final
cdef class IntSoftHinge_Atan(Func):
    #
    def __init__(self, alpha=1.0, x0=0):
        self.alpha = alpha
        self.x0 = x0
    #
    @cython.final
    cdef double _evaluate(self, const double x) noexcept nogil:
        cdef double a = self.alpha
        cdef double x1 = x - self.x0
        return (x1 * (pi * x1 + 2*a) - 2 * (x1 * x1 + a*a) * atan(x1/a)) / (4 * pi)
    #
    @cython.final
    @cython.cdivision(True)
    cdef double _derivative(self, const double x) noexcept nogil:
        cdef double a = self.alpha
        cdef double x1 = x - self.x0
        return x1 * (0.5 + (1/pi)*atan(-x1/a))
    #
    @cython.final
    @cython.cdivision(True)
    cdef double _derivative_div(self, const double x) noexcept nogil:
        cdef double a = self.alpha
        cdef double x1 = x - self.x0
        return (0.5 + (1/pi)*atan(-x1/a))
    #
    def _repr_latex_(self):
        return r"$ρ(x)=\frac{1}{4\pi} (x(2a+\pi x) - 2(a^2+x^2)\atan(x/a))$"

    def to_dict(self):
        return { 'name':'softhinge_sqrt',
                 'args': (self.alpha,) }


@cython.final
cdef class IntSoftHinge_Sqrt(Func):
    #
    def __init__(self, alpha = 1.0, x0=0):
        self.alpha = alpha
        self.alpha2 = alpha*alpha
        self.x0 = x0
    #
    @cython.final
    cdef double _evaluate(self, const double x) noexcept nogil:
        cdef double x1 = x - self.x0
        cdef double alpha = self.alpha
        cdef double alpha2 = self.alpha2

        return 0.25 * (alpha2*asinh(x1/alpha) - x*sqrt(alpha2+x1*x1) + 2*alpha*x1 + x1*x1)
    #
    @cython.final
    cdef double _derivative(self, const double x) noexcept nogil:
        cdef double x1 = x - self.x0
        return 0.5 * (-x1 + sqrt(self.alpha2 + x1*x1))
    #
    @cython.final
    @cython.cdivision(True)
    cdef double _derivative_div(self, const double x) noexcept nogil:
        cdef double x1 = x - self.x0
            
        return 0.5 * (-1 + x1/sqrt(self.alpha2 + x1*x1))
    #
    @cython.final
    @cython.cdivision(True)
    cdef double _derivative2(self, const double x) noexcept nogil:
        cdef double x1 = x - self.x0
        return 0.5 * (-1 + x1 / sqrt(self.alpha2 + x1*x1))
    #
    def _repr_latex_(self):
        return r"$ρ(x)=\int\frac{1}{2}(-x + \sqrt{\alpha^2+x^2})\,dx$"

    def to_dict(self):
        return { 'name':'softhinge_sqrt',
                 'args': (self.alpha,) }

@cython.final
cdef class SoftHinge_Sqrt(Func):
    #
    def __init__(self, alpha = 1.0, x0=0):
        self.alpha = alpha
        self.alpha2 = alpha*alpha
        self.x0 = x0
    #
    @cython.final
    cdef double _evaluate(self, const double x) noexcept nogil:
        cdef double x1 = x - self.x0
        return 0.5 * (-x1 + sqrt(self.alpha2 + x1*x1))
    #
    @cython.final
    @cython.cdivision(True)
    cdef double _derivative(self, const double x) noexcept nogil:
        cdef double x1 = x - self.x0
        return 0.5 * (-1 + x1/sqrt(self.alpha2 + x1*x1))
    #
    @cython.final
    @cython.cdivision(True)
    cdef double _derivative2(self, const double x) noexcept nogil:
        cdef double x1 = x - self.x0
        return 0.5 * self.alpha2 / sqrt(self.alpha2 + x1*x1)
    #
    def _repr_latex_(self):
        return r"$ρ(x)=\frac{1}{2}(-x + \sqrt{c^2+x^2})$"

    def to_dict(self):
        return { 'name':'softhinge_sqrt',
                 'args': (self.alpha, self.x0) }

@cython.final
cdef class SoftHinge_Exp(Func):
    #
    def __init__(self, alpha = 1.0, x0=0):
        self.alpha = alpha
        self.x0 = x0
    #
    @cython.final
    cdef double _evaluate(self, const double x) noexcept nogil:
        cdef double x1 = self.alpha * (x - self.x0)
        if x1 >= 0:
            return log(1 + exp(-x1))
        else:
            return -x1 + log(1 + exp(x1))
    #
    @cython.final
    @cython.cdivision(True)
    cdef double _derivative(self, const double x) noexcept nogil:
        cdef double alpha = self.alpha
        cdef double x1 = alpha * (x - self.x0)
        cdef double v
        if x1 >= 0:
            v = exp(-x1)
            return -alpha * v / (1 + v)
        else:
            return -alpha / (1 + exp(x1))
    #
    @cython.final
    @cython.cdivision(True)
    cdef double _derivative2(self, const double x) noexcept nogil:
        cdef double x1 = x - self.x0
        cdef double alpha = self.alpha
        cdef double v = exp(-alpha*x1)
        cdef double v1 = 1 / (1 + v)
        return alpha*alpha * v1 * (1 - v1)
    #
    def _repr_latex_(self):
        return r"$ρ(x)=\frac{1}{2}(-x + \sqrt{c^2+x^2})$"

    def to_dict(self):
        return { 'name':'softhinge_exp',
                 'args': (self.alpha, self.x0) }

@cython.final
cdef class Hinge2(Func):
    #
    def __init__(self, C=1.0):
        self.C = C
    #
    @cython.final
    cdef double _evaluate(self, const double x) noexcept nogil:
        cdef double v = self.C - x
        if v < 0:
            return 0
        else:
            return 0.5 * v * v
    #
    @cython.final
    cdef double _derivative(self, const double x) noexcept nogil:
        cdef double v = self.C - x
        if v < 0:
            return 0
        else:
            return -v
    #
    @cython.final
    cdef double _derivative_div(self, const double x) noexcept nogil:
        cdef double C = self.C
        cdef double v = C - x
        if v < 0:
            return 0
        else:
            return 1
    #
    @cython.final
    cdef double _derivative2(self, const double x) noexcept nogil:
        cdef double v = self.C - x
        if v < 0:
            return 0
        else:
            return 1
    #
    @cython.final
    cdef double _value(self, const double x) noexcept nogil:
        return self.C - x
    #
    def _repr_latex_(self):
        return r"$ρ(x)=(c-x)_{+}$"

    def to_dict(self):
        return { 'name':'hinge',
                 'args': (self.C,) }
