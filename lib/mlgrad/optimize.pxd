
cdef class Func:
    cdef double _evaluate(self, double[::1] x)
    cdef void _gradient(self, double[::1] x, double[::1] grad)


cdef double backtracking_line_search(Func f, double[::1] x, double[::1] p, double alpha0=*, double rho=*, double c=*, Py_ssize_t n_iter=*)
