
cimport mlgrad.inventory as inventory

cdef class Func:
    #
    cdef double _evaluate(self, double[::1] x):
        return  0
    #
    cdef void _gradient(self, double[::1] x, double[::1] grad):
        pass


cdef double backtracking_line_search(Func f, double[::1] x, double[::1] p,
                                     double alpha0=1.0, double rho=1.5, double c=1.0e-4, Py_ssize_t n_iter=50):
    """
    Parameters:
        f        : callable, objective function f(x)
        x        : numpy array, current point
        p        : numpy array, search direction
        alpha0   : float, initial step size
        rho      : float, reduction factor (should be in (0,1))
        c        : float, Armijo constant (small positive number)
        max_iter : int, maximum number of iterations
    Returns:
        alpha    : float, step size satisfying the Armijo condition
    """
    cdef Py_ssize_t n = x.shape[0]
    cdef double[::1] gx = inventory.empty_array(n)
    cdef double[::1] temp = inventory.empty_array(n)
    cdef double alpha = alpha0
    cdef double fx = f._evaluate(x), gxp

    f._gradient(x, gx)
    gxp = inventory._dot(&p[0], &gx[0], n)
    # gxp = grad(x).dot(p)
    for _ in range(n_iter):
        inventory._mul_const(&temp[0], &p[0], alpha, n)
        inventory._iadd(&temp[0], &x[0], n)
        # if f._evaluate(x + alpha * p) <= fx + c * alpha * gx:
        if f._evaluate(temp) <= fx + c * alpha * gxp:
            break
        alpha *= rho
    return alpha
