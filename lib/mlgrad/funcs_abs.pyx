
@cython.final
cdef class Abs(Func):
    #
    @cython.final
    cdef double _evaluate(self, const double x) noexcept nogil:
        return fabs(x)
    #
    @cython.final
    cdef double _derivative(self, const double x) noexcept nogil:
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            0
    #
    @cython.final
    cdef double _derivative2(self, const double x) noexcept nogil:
        if x == 0:
            return c_inf
        else:
            return 0
    #
    def _repr_latex_(self):
        return r"$ρ(x)=|x|$"

    def to_dict(self):
        return { 'name':'absolute',
                 'args': () }

@cython.final
cdef class SoftAbs(Func):
    #
    def __init__(self, eps=1.0):
        self.eps = eps
    #
    @cython.final
    @cython.cdivision(True)
    cdef double _evaluate(self, const double x) noexcept nogil:
        return x * x / (self.eps + fabs(x))
    #
    @cython.final
    @cython.cdivision(True)
    cdef double _derivative(self, const double x) noexcept nogil:
        cdef double v = self.eps + fabs(x)
        return x * (self.eps + v) / (v * v)
    #
    @cython.final
    @cython.cdivision(True)
    cdef double _derivative2(self, const double x) noexcept nogil:
        cdef double eps = self.eps
        cdef double v = eps + fabs(x)
        return 2 * eps * eps / (v * v * v)
    #
    @cython.final
    @cython.cdivision(True)
    cdef double _derivative_div(self, const double x) noexcept nogil:
        cdef double v = self.eps + fabs(x)
        return (self.eps + v) / (v * v)
    #
    def _repr_latex_(self):
        return r"$p(x)=\frac{x^2}{\varepsilon+|x|}$"

    def to_dict(self):
        return { 'name':'softabs',
                 'args': (self.eps,) }


@cython.final
cdef class SoftAbs_Sqrt(Func):
    #
    def __init__(self, eps=1.0):
        self.eps = eps
        self.eps2 = eps*eps
    #
    @cython.final
    cdef double _evaluate(self, const double x) noexcept nogil:
        return sqrt(self.eps2 + x*x) - self.eps
    #
    @cython.final
    cdef void _evaluate_array(self, const double *x, double *y, const Py_ssize_t n) noexcept nogil:
        cdef Py_ssize_t i
        cdef double v, eps = self.eps, eps2 = self.eps2

        # for i in prange(n, nogil=True, schedule='static', num_threads=num_procs):
        for i in range(n):
            v = x[i]
            y[i] = sqrt(eps2 + v*v) - eps
    #
    @cython.cdivision(True)
    @cython.final
    cdef double _derivative(self, const double x) noexcept nogil:
        # cdef double v = self.eps2 + x*x
        return x / sqrt(self.eps2 + x*x)
    #
    @cython.cdivision(True)
    @cython.final
    cdef void _derivative_array(self, const double *x, double *y, const Py_ssize_t n) noexcept nogil:
        cdef Py_ssize_t i
        cdef double v, eps = self.eps, eps2 = self.eps2

        # for i in prange(n, nogil=True, schedule='static', num_threads=num_procs):
        for i in range(n):
            v = x[i]
            y[i] = v / sqrt(eps2 + v*v)
    #
    @cython.cdivision(True)
    @cython.final
    cdef double _derivative2(self, const double x) noexcept nogil:
        cdef double v = self.eps2 + x*x
        return self.eps2 / (v * sqrt(v))
    #
    @cython.cdivision(True)
    @cython.final
    cdef void _derivative2_array(self, const double *x, double *y, const Py_ssize_t n) noexcept nogil:
        cdef Py_ssize_t i
        cdef double v, v2, eps = self.eps, eps2 = self.eps2

        # for i in prange(n, nogil=True, schedule='static', num_threads=num_procs):
        for i in range(n):
            v = x[i]
            v2 = eps2 + v*v
            y[i] = eps2 / (v2 * sqrt(v2))
    #
    @cython.cdivision(True)
    @cython.final
    cdef double _derivative_div(self, const double x) noexcept nogil:
        return 1. / sqrt(self.eps2 + x*x)
    #
    @cython.cdivision(True)
    @cython.final
    cdef void _derivative_div_array(self, const double *x, double *y, const Py_ssize_t n) noexcept nogil:
        cdef Py_ssize_t i
        cdef double v, v2, eps = self.eps, eps2 = self.eps2

        # for i in prange(n, nogil=True, schedule='static', num_threads=num_procs):
        for i in range(n):
            v = x[i]
            y[i] = 1. / sqrt(eps2 + v*v)
    #
    @cython.final
    cdef double _inverse(self, const double y) noexcept nogil:
        cdef double v = y + self.eps
        cdef double s = v*v - self.eps2
        if s < 0:
            s = 0
        return sqrt(s)
    #
    def _repr_latex_(self):
        return r"$\rho(x)=\sqrt{\varepsilon^2+x^2}$ - \varepsilon"

    def to_dict(self):
        return { 'name':'sqrt',
                 'args': (self.eps) }

@cython.final
cdef class SoftAbs_FSqrt(Func):
    #
    def __init__(self, eps=1.0, q=0.5):
        self.eps = eps
        self.eps2 = eps*eps
        self.eps3 = pow(eps, 2*q)
        self.q = q
    #
    @cython.final
    @cython.final
    cdef double _evaluate(self, const double x) noexcept nogil:
        return pow(self.eps2 + x*x, self.q) - self.eps3
    #
    # @cython.final
    # cdef void _evaluate_array(self, const double *x, double *y, const Py_ssize_t n) noexcept nogil:
    #     cdef Py_ssize_t i
    #     cdef double v, eps = self.eps, eps2 = self.eps2, eps3 = self.eps3, q=self.q

    #     for i in prange(n, nogil=True, schedule='static', num_threads=num_procs):
    #         v = x[i]
    #         y[i] = pow(eps2 + v*v, q) - eps3
    #
    @cython.final
    cdef double _derivative(self, const double x) noexcept nogil:
        cdef double q=self.q
        return 2 * q * x * pow(self.eps2 + x*x, q-1)
    #
    # @cython.final
    # cdef void _derivative_array(self, const double *x, double *y, const Py_ssize_t n) noexcept nogil:
    #     cdef Py_ssize_t i
    #     cdef double v, eps = self.eps, eps2 = self.eps2

    #     for i in prange(n, nogil=True, schedule='static', num_threads=num_procs):
    #         v = x[i]
    #         y[i] = v / sqrt(eps2 + v*v)
    #
    @cython.final
    cdef double _derivative2(self, const double x) noexcept nogil:
        cdef double q=self.q
        cdef double x2 = x * x
        cdef double v = self.eps2 + x2
        # return 2 * q * pow(v, q-1) + 4 * q * q * x2 * pow(v, q-2)
        return 2 * q * pow(v, q-1) * (1 + 2 * q * x2 / v)
    #
    # @cython.cdivision(True)
    # @cython.final
    # cdef void _derivative2_array(self, const double *x, double *y, const Py_ssize_t n) noexcept nogil:
    #     cdef Py_ssize_t i
    #     cdef double v, v2, eps = self.eps, eps2 = self.eps2

    #     for i in prange(n, nogil=True, schedule='static', num_threads=num_procs):
    #         v = x[i]
    #         v2 = eps2 + v*v
    #         y[i] = eps2 / (v2 * sqrt(v2))
    #
    @cython.final
    cdef double _derivative_div(self, const double x) noexcept nogil:
        cdef double q=self.q
        return 2 * q * pow(self.eps2 + x*x, q-1)
    #
    # @cython.cdivision(True)
    # @cython.final
    # cdef void _derivative_div_array(self, const double *x, double *y, const Py_ssize_t n) noexcept nogil:
    #     cdef Py_ssize_t i
    #     cdef double v, v2, eps = self.eps, eps2 = self.eps2

    #     for i in prange(n, nogil=True, schedule='static', num_threads=num_procs):
    #         v = x[i]
    #         y[i] = 1. / sqrt(eps2 + v*v)
    #
    # def _repr_latex_(self):
    #     return r"$p(x)=\sqrt{\varepsilon^2+x^2}$"

    # def to_dict(self):
    #     return { 'name':'sqrt',
    #              'args': (self.eps) }

cdef double ln2 = log(2)

@cython.final
cdef class SoftAbs_Exp(Func):
    #
    def __init__(self, eps=1.0):
        self.eps = eps
        self.eps1 = 1/eps
    #
    @cython.final
    cdef double _evaluate(self, const double x) noexcept nogil:
        if x > 0:
            return  x + (log(1 + exp(-2*self.eps1*x)) - ln2) * self.eps
        elif x < 0:
            return -x + (log(1 + exp( 2*self.eps1*x)) - ln2) * self.eps
        else:
            return 0
    #
    @cython.cdivision(True)
    @cython.final
    cdef double _derivative(self, const double x) noexcept nogil:
        cdef double eps1 = self.eps1

        if x > 0:
            return (1 - exp(-2*eps1*x)) / (1 + exp(-2*eps1*x))
        elif x < 0:
            return (exp(2*eps1*x) - 1) / (1 + exp(2*eps1*x))
        else:
            return 0
    #
    @cython.cdivision(True)
    @cython.final
    cdef double _derivative2(self, const double x) noexcept nogil:
        cdef double v
        cdef double eps1 = self.eps1

        if x > 0:
            v = (1 - exp(-2*eps1*x)) / (1 + exp(-2*eps1*x))
        elif x < 0:
            v = (exp(2*eps1*x - 1)) / (1 + exp(2*eps1*x))
        else:
            v = 0

        return eps1 * (1 - v*v)
    #
    # @cython.cdivision(True)
    # @cython.final
    # cdef void _derivative2_array(self, const double *x, double *y, const Py_ssize_t n) noexcept nogil:
    #     cdef Py_ssize_t i
    #     cdef double v, v2, eps = self.eps, eps2 = self.eps2

    #     for i in prange(n, nogil=True, schedule='static', num_threads=num_procs):
    #         v = x[i]
    #         v2 = eps2 + v*v
    #         y[i] = eps2 / (v2 * sqrt(v2))
    #
    # @cython.cdivision(True)
    @cython.final
    cdef double _derivative_div(self, const double x) noexcept nogil:
        cdef double eps = self.eps
        cdef double eps1 = self.eps1

        if x > 0:
            return (1 - exp(-2*self.eps1*x)) / (1 + exp(-2*self.eps1*x)) / x
        elif x < 0:
            return (1 - exp(2*self.eps1*x)) / (1 + exp(2*self.eps1*x)) / x
        else:
            return 0
    #
    # @cython.cdivision(True)
    # @cython.final
    # cdef void _derivative_div_array(self, const double *x, double *y, const Py_ssize_t n) noexcept nogil:
    #     cdef Py_ssize_t i
    #     cdef double v, v2, eps = self.eps, eps2 = self.eps2

    #     for i in prange(n, nogil=True, schedule='static', num_threads=num_procs):
    #         v = x[i]
    #         y[i] = 1. / sqrt(eps2 + v*v)
    #
    def _repr_latex_(self):
        return r"$p(x)=\log(\exp(x/\epsilon) + \exp(-x/\epsilon)) - \log 2$"

    def to_dict(self):
        return { 'name':'sqrt',
                 'args': (self.eps) }
