def soft_quantile_func(alpha, func):
    if type(func) is SoftAbs_Sqrt:
        return Quantile_Sqrt(alpha, func.eps)
    elif type(func) is SoftAbs_Exp:
        return Quantile_Sqrt(alpha, func.eps)
    else:
        return QuantileFunc(alpha, func)

@cython.final
cdef class QuantileFunc(Func):
    #
    def __init__(self, alpha, Func func):
        self.alpha = alpha
        self.f = func
    #
    @cython.final
    cdef double _evaluate(self, const double x) noexcept nogil:
        if x > 0:
            return self.alpha * self.f._evaluate(x)
        elif x < 0:
            return (1-self.alpha) * self.f._evaluate(x)
        else:
            return 0.5 * self.f._evaluate(0)
    #
    @cython.final
    cdef double _derivative(self, const double x) noexcept nogil:
        if x > 0:
            return self.alpha * self.f._derivative(x)
        elif x < 0:
            return (1-self.alpha) * self.f._derivative(x)
        else:
            return 0.5 * self.f._derivative(0)
    #
    @cython.final
    cdef double _derivative2(self, const double x) noexcept nogil:
        if x > 0:
            return self.alpha * self.f._derivative2(x)
        elif x < 0:
            return (1-self.alpha) * self.f._derivative2(x)
        else:
            return 0.5 * self.f._derivative2(0)
    #
    @cython.final
    cdef double _derivative_div(self, const double x) noexcept nogil:
        if x > 0:
            return self.alpha * self.f._derivative_div(x)
        elif x < 0:
            return (1-self.alpha) * self.f._derivative_div(x)
        else:
            return 0.5 * self.f._derivative_div(0)
    #
    def _repr_latex_(self):
        return '$\mathrm{id}(x)=x$'

    def to_dict(self):
        return { 'name':'quantile_func',
                'args': (self.alpha, self.f.to_dict() )
               }

@cython.final
cdef class Quantile(Func):
    #
    def __init__(self, alpha=0.5):
        self.alpha = alpha
    #
    @cython.final
    cdef double _evaluate(self, const double x) noexcept nogil:
        if x < 0:
            return (self.alpha - 1) * x
        elif x > 0:
            return self.alpha * x
        else:
            return 0
    #
    @cython.final
    cdef double _derivative(self, const double x) noexcept nogil:
        if x < 0:
            return self.alpha - 1.0
        elif x > 0:
            return self.alpha
        else:
            return self.alpha - 0.5
    #
    @cython.final
    cdef double _derivative2(self, const double x) noexcept nogil:
        if x == 0:
            return c_inf
        else:
            return 0
    #
    def _repr_latex_(self):
        return r"$ρ(x)=(\alpha - [x < 0])x$"

    def to_dict(self):
        return { 'name':'quantile',
                 'args': (self.alpha,) }

@cython.final
cdef class Quantile_Sqrt(Func):
    #
    def __init__(self, alpha=0.5, eps=1.0):
        self.alpha = alpha
        self.eps = eps
        self.eps2 = eps*eps
    #
    @cython.final
    cdef double _evaluate(self, const double x) noexcept nogil:
        cdef double v = self.eps2 + x*x
        if x >= 0:
            return (sqrt(v) - self.eps) * self.alpha
        else:
            return (sqrt(v) - self.eps) * (1-self.alpha)
    #
    @cython.final
    @cython.cdivision(True)
    cdef double _derivative(self, const double x) noexcept nogil:
        cdef double v = self.eps2 + x*x
        if x >= 0:
            return self.alpha * x / sqrt(v)
        else:
            return (1.-self.alpha) * x / sqrt(v)
    #
    @cython.final
    @cython.cdivision(True)
    cdef double _derivative2(self, const double x) noexcept nogil:
        cdef double v = self.eps2 + x*x
        if x >= 0:
            return self.alpha * self.eps2 / (v * sqrt(v))
        else:
            return (1.-self.alpha) * self.eps2 / (v * sqrt(v))
    #
    @cython.final
    @cython.cdivision(True)
    cdef double _derivative_div(self, const double x) noexcept nogil:
        cdef double v = self.eps2 + x*x
        if x >= 0:
            return self.alpha / sqrt(v)
        else:
            return (1.-self.alpha) / sqrt(v)
    #
    @cython.final
    @cython.cdivision(True)
    cdef void _evaluate_array(self, const double *x, double *y, const Py_ssize_t n) noexcept nogil:
        cdef Py_ssize_t i
        cdef double u, v
        for i in range(n):
            v = x[i]
            u = self.eps2 + v*v
            if v >= 0:
                y[i] = (sqrt(u) - self.eps) * self.alpha
            else:
                y[i] = (sqrt(u) - self.eps) * (1-self.alpha)
    #
    @cython.final
    @cython.cdivision(True)
    cdef void _derivative_array(self, const double *x, double *y, const Py_ssize_t n) noexcept nogil:
        cdef Py_ssize_t i
        cdef double u, v
        for i in range(n):
            v = x[i]
            u = self.eps2 + v*v
            if v >= 0:
                y[i] = self.alpha * v / sqrt(u)
            else:
                y[i] = (1.-self.alpha) * v / sqrt(u)
    #
    @cython.final
    @cython.cdivision(True)
    cdef void _derivative2_array(self, const double *x, double *y, const Py_ssize_t n) noexcept nogil:
        cdef Py_ssize_t i
        cdef double u, v
        for i in range(n):
            v = x[i]
            u = self.eps2 + v*v
            if v >= 0:
                y[i] = self.alpha * self.eps2 / (u * sqrt(u))
            else:
                y[i] = (1.-self.alpha) * self.eps2 / (u * sqrt(u))
    #
    @cython.final
    @cython.cdivision(True)
    cdef void _derivative_div_array(self, const double *x, double *y, const Py_ssize_t n) noexcept nogil:
        cdef Py_ssize_t i
        cdef double u, v
        for i in range(n):
            v = x[i]
            u = self.eps2 + v*v
            if v >= 0:
                y[i] = self.alpha / sqrt(u)
            else:
                y[i] = (1.-self.alpha) / sqrt(u)
    #
    def _repr_latex_(self):
        return r"$p(x)=(\sqrt{\varepsilon^2+x^2}-\varepsilon)_\alpha$"

    def to_dict(self):
        return { 'name':'quantile_sqrt',
                 'args': (self.alpha, self.eps) }

@cython.final
cdef class Quantile_AlphaLog(Func):
    #
    def __init__(self, alpha=1.0, q=0.5):
        assert alpha > 0
        self.alpha = alpha
        self.q = q
        if alpha == 0:
            self.alpha2 = 0.
        else:
            self.alpha2 = self.alpha*log(self.alpha)
    #
    @cython.final
    cdef double _evaluate(self, const double x) noexcept nogil:
        cdef double val
        if x < 0:
            val = -x - self.alpha*log(self.alpha - x) + self.alpha2
            return (1.0-self.q) * val
        elif x > 0:
            val = x - self.alpha*log(self.alpha + x) + self.alpha2
            return self.q * val
        else:
            return 0
    #
    @cython.final
    @cython.cdivision(True)
    cdef double _derivative(self, const double x) noexcept nogil:
        cdef double val
        if x < 0:
            val = x / (self.alpha - x)
            return (1-self.q) * val
        elif x > 0:
            val = x / (self.alpha + x)
            return self.q * val
        else:
            return self.q - 0.5
    #
    @cython.final
    @cython.cdivision(True)
    cdef double _derivative2(self, const double x) noexcept nogil:
        cdef double v
        if x < 0:
            v = self.alpha - x
            return (1-self.q)*self.alpha / (v*v)
        elif x > 0:
            v = self.alpha + x
            return self.q*self.alpha / (v*v)
        else:
            return 0.5 / self.alpha
    #
    @cython.final
    @cython.cdivision(True)
    cdef double _derivative_div(self, const double x) noexcept nogil:
        cdef double val
        if x < 0:
            val = 1 / (self.alpha - x)
            return (1-self.q) * val
        elif x > 0:
            val = 1 / (self.alpha + x)
            return self.q * val
        else:
            return (self.q - 0.5) / self.alpha
    #
    def _repr_latex_(self):
        return r"$ρ_q(x)=\mathrm{sign}_q(x)(|x| - \alpha\ln(\alpha+|x|)+\alpha\ln\alpha)$"

    def to_dict(self):
        return { 'name':'quantile_alpha_log',
                 'args': (self.alpha, self.q) }
