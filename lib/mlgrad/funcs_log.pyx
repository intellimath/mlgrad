
cdef class Log(Func):
    #
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    #
    cdef double _evaluate(self, const double x) noexcept nogil:
        return log(self.alpha+x)
    #
    cdef double _derivative(self, const double x) noexcept nogil:
        return 1 / (self.alpha+x)
    #
    cdef double _derivative2(self, const double x) noexcept nogil:
        cdef double x2 = self.alpha+x
        return -1 / (x2*x2)
    #
    def _repr_latex_(self):
        return r"$\rho(x)=\ln{\alpha+x}$"

    def to_dict(self):
        return { 'name':'log',
                 'args': (self.alpha,) }


@cython.final
cdef class LogSquare(Func):

    def __init__(self, a=1.0):
        self.a = a
        self.a2 = a * a
    #
    @cython.final
    cdef double _evaluate(self, const double x) noexcept nogil:
        cdef double v = x / self.a
        return self.a2 * log(1 + 0.5*v*v)
    #
    @cython.final
    @cython.cdivision(True)
    cdef double _derivative(self, const double x) noexcept nogil:
        cdef double v = x / self.a
        return x / (1 + 0.5*v*v)
    #
    @cython.final
    @cython.cdivision(True)
    cdef double _derivative_div(self, const double x) noexcept nogil:
        cdef double v = x / self.a
        return 1 / (1 + 0.5*v*v)
    #
    @cython.final
    @cython.cdivision(True)
    cdef double _derivative2(self, const double x) noexcept nogil:
        cdef double v = x / self.a
        cdef double v2 = v*v
        cdef double vv = 1 + 0.5 * v2
        return (1 - 0.5 * v2) / vv * vv

    def _repr_latex_(self):
        return r'$a^2\ln(1 + \frac{1}{2}(x/a)^2)$'

    def to_dict(self):
        return { 'name':'log_square',
                 'args': (self.a,) }

@cython.final
cdef class SoftLog(Func):
    #
    def __init__(self, eps=1.0):
        self.eps = eps
        self.log_eps = log(eps)
    #
    @cython.final
    cdef double _evaluate(self, const double x) noexcept nogil:
        return log(x + self.eps) - self.log_eps
    #
    @cython.final
    cdef double _derivative(self, const double x) noexcept nogil:
        return 1 / (x + self.eps)
    #
    def _repr_latex_(self):
        return r"$ρ(x)=log(x + \epsilon) - log(\epsilon)$"

    def to_dict(self):
        return { 'name':'softlog',
                 'args': (self.eps,) }

def softlog_div(double[::1] X, double eps=1.0e-9):
    cdef Py_ssize_t i, n=X.shape[0]
    cdef double x, s, log_eps = log(eps)
    cdef double[::1] YY

    Y = np.empty(n)
    YY = Y
    for i in range(n):
        x = fabs(X[i])
        if x == 0:
            YY[i] = 1.0 / eps
        else:
            YY[i] = (log(x + eps) - log_eps) / x
    s = 0
    for i in range(n):
        s += YY[i]
    s /= n
    for i in range(n):
        YY[i] /= s
    return Y