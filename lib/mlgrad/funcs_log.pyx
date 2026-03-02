
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