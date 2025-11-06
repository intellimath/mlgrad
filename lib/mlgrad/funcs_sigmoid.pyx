
cdef class ModSigmoidal(Func):
    #
    def __init__(self, a=1):
        self.label = u'σ'
        self.a = a
    #
    @cython.final
    cdef double _evaluate(self, const double x) noexcept nogil:
        return x / (self.a + fabs(x))
    #
    @cython.final
    cdef double _derivative(self, const double x) noexcept nogil:
        cdef double v = (self.a + fabs(x))
        return self.a / (v*v)
    #
    @cython.final
    cdef double _derivative2(self, const double x) noexcept nogil:
        cdef double v = (self.a + fabs(x))
        if x > 0:
            return -2.0 * self.a / v*v*v
        elif x < 0:
            return 2.0 * self.a / v*v*v
        else:
            return 0
    #
    def _repr_latex_(self):
        return '$%s(x, a)=\dfrac{x}{a+|x|}$' % self.label

@cython.final
cdef class Sigmoidal(Func):
    #
    def __init__(self, p=1):
        self.label = u'σ'
        self.p = p
    #
    @cython.final
    @cython.cdivision(True)
    cdef double _evaluate(self, const double x) noexcept nogil:
        return tanh(self.p * x)
    #
    @cython.final
    @cython.cdivision(True)
    cdef double _derivative(self, const double x) noexcept nogil:
        cdef double p = self.p
        cdef double v = cosh(p * x)
        return p / (v * v)
    #
    @cython.final
    @cython.cdivision(True)
    cdef double _derivative2(self, const double x) noexcept nogil:
        cdef double p = self.p
        cdef double v = cosh(p * x)
        return -2 * p * p * sinh(p * x) / (v * v * v)
    #
    def _repr_latex_(self):
        return '$%s(x, p)=th(px)$' % self.label

    def to_dict(self):
        return { 'name':'sigmoidal',
                 'args': (self.p,) }

@cython.final
cdef class DOOM(Func):
    #
    def __init__(self, p=1):
        self.label = u'σ'
        self.p = p
    #
    @cython.final
    @cython.cdivision(True)
    cdef double _evaluate(self, const double x) noexcept nogil:
        return 1-tanh(self.p * x)
    #
    @cython.final
    @cython.cdivision(True)
    cdef double _derivative(self, const double x) noexcept nogil:
        cdef double p = self.p
        cdef double v = cosh(p * x)
        return -p / (v * v)
    #
    @cython.final
    @cython.cdivision(True)
    cdef double _derivative2(self, const double x) noexcept nogil:
        cdef double p = self.p
        cdef double v = cosh(p * x)
        return 2 * p * p * sinh(p * x) / (v * v * v)
    #
    def _repr_latex_(self):
        return '$%s(x, p)=1-th(px)$' % self.label

    def to_dict(self):
        return { 'name':'doom',
                 'args': (self.p,) }

@cython.final
cdef class Arctang(Func):
    #
    def __init__(self, a=1):
        self.label = u'σ'
        self.a = a
    #
    @cython.final
    @cython.cdivision(True)
    cdef double _evaluate(self, const double x) noexcept nogil:
        return atan(x/self.a)
    #
    @cython.final
    @cython.cdivision(True)
    cdef double _derivative(self, const double x) noexcept nogil:
        cdef double v = x/self.a
        return 1 / (self.a * (1 + v*v))
    #
    @cython.final
    @cython.cdivision(True)
    cdef double _derivative2(self, const double x) noexcept nogil:
        cdef double v = x /self.a
        cdef double a2 = self.a * self.a
        cdef double u = 1 + v*v
        return -2*v / (a2 * u*u)
    #
    def _repr_latex_(self):
        return '$%s(x, p)=\dfrac{1}{1+e^{-px}}$' % self.label

    def to_dict(self):
        return { 'name':'arctg',
                 'args': (self.a,) }
