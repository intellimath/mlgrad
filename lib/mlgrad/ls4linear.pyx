#
# Least Squares for Simple Linear Models
#

from mlgrad cimport LinearModel

def linear_ls(X, Y, mod=None):
    if mod is None:
        mod = LinearModel(X.shape[1])
        mod.allocate()
    _linear_ls(X, Y, mod)
    return mod

cdef void _linear_ls(double[::1] X, double[::1] Y, LinearModel mod):
    cdef double sx = 0, sy = 0, sx2 = 0, sxy = 0
    cdef Py_ssize_t i, n = X.shape[0]
    cdef double *XX = &X[0], *YY = &Y[0]
    cdef double xi, yi, nn = n
    cdef double a, b
    cdef double *param = &mod.param[0]

    for i in range(n):
        xi = XX[i]
        yi = YY[i]
        sx += xi
        sy += yi
        sxy += xi * yi
        sx2 += xi * xi

    a = (nn * sxy - sx * sy) / (nn * sx2 - sx * sx)
    b = (sy - a * sx) / nn

    param[0] = b
    param[1] = a

# cdef double _linear_ls_slope(double[::1] X, double[::1] Y):
#     cdef double sx = 0, sy = 0, sx2 = 0, sxy = 0
#     cdef Py_ssize_t i, n = X.shape[0]
#     cdef double *XX = &X[0], *YY = &Y[0]
#     cdef double xi, yi, nn = n

#     for i in range(n):
#         xi = XX[i]
#         yi = YY[i]
#         sx += xi
#         sy += yi
#         sxy += xi * yi
#         sx2 += xi * xi

#     return (nn * sxy - sx * sy) / (nn * sx2 - sx * sx)

# cdef double linear_ls_slope2(double[::1] Y):
#     cdef Py_ssize_t i, n = Y.shape[0]
#     cdef double xi, yi, nn = n
#     cdef double nn = n
#     # cdef double sx = nn * (nn + 1) / 2
#     cdef double sx_n = (nn + 1) / 2
#     cdef double sx2 = nn * (nn + 1) * (2 * nn + 1) / 6
#     cdef double sy = 0, sxy = 0
#     cdef double *YY = &Y[0]

#     for i in range(n):
#         # xi = i+1
#         yi = YY[i]
#         # sx += xi 
#         sy += yi
#         sxy += (i+1) * yi
#         # sxy += xi * yi
#         # sx2 += xi * xi

#     return (sxy - sx_n * sy) / (sx2 - sx_n * nn)

def linear_wls(X, Y, W, mod=None):
    if mod is None:
        mod = LinearModel(X.shape[1])
        mod.allocate()
    _linear_wls(X, Y, W, mod)
    return mod

cdef void linear_wls(double[::1] X, double[::1] Y, double[::1] W, LinearModel mod):
    cdef double sx = 0, sy = 0, sx2 = 0, sxy = 0, sw = 0
    cdef Py_ssize_t i, n = X.shape[0]
    cdef double *XX = &X[0], *YY = &Y[0], *WW = &W[0]
    cdef double xi, yi, wi
    cdef double a, b
    cdef double *param = &mod.param[0]

    for i in range(n):
        xi = XX[i]
        yi = YY[i]
        wi = WW[i]
        sx += wi * xi
        sy += wi * yi
        sxy += wi * xi * yi
        sx2 += wi * xi * xi
        sw += wi

    a = (sw * sxy - sx * sy) / (sw * sx2 - sx * sx)
    b = (sy - a * sx) / sw

    param[0] = b
    param[1] = a


def linear_irls(X, Y, Func func, tol=1.0e-6, n_iter=100):
    """
    Iteratively Reweighted Least Squares for linear regression  

    X:
        input array
    Y: 
        output array
    func:
        Func object
    tol:
        tolerance
    n_iter:
        Maximal number of iteration
    """
    # X = np.asarray(X)
    Xs = X[:,None]
    # Y = np.asarray(Y)

    mod = linear_ls(X, Y, mod)
    E = mod.evaluate(Xs) - Y
    lval = lval_min = func.evaluate_array(E).sum()

    param_min = mod.param.copy()

    to_exit = False
    for k in range(n_iter):
        lval_prev = lval
        
        weights = func.derivative_div_array(E)
        linear_wls(X, Y, mod, weights)

        E = mod.evaluate(Xs) - Y
        lval = func.evaluate_array(E).sum()

        if abs(lval - lval_prev) / (1 + abs(lval_min)) < tol:
            to_exit = True

        if lval < lval_min:
            lval_min = lval
            param_min = mod.param.copy()

        if to_exit:
            break

    mod.param[:] = param_min
    return mod

# cdef double linear_wls_slope(double[::1] X, double[::1] Y, double[::1] W):
#     cdef double sx = 0, sy = 0, sx2 = 0, sxy = 0, sw = 0
#     cdef Py_ssize_t i, n = X.shape[0]
#     cdef double *XX = &X[0], *YY = &Y[0], *WW = &W[0]
#     cdef double xi, yi, wi

#     for i in range(n):
#         xi = XX[i]
#         yi = YY[i]
#         wi = WW[i]
#         sx += wi * xi
#         sy += wi * yi
#         sxy += wi * xi * yi
#         sx2 += wi * xi * xi
#         sw += wi

#     return (sw * sxy - sx * sy) / (sw * sx2 - sx * sx)

# cdef double linear_wls_slope2(double[::1] Y, double[::1] W):
#     cdef double sx = 0, sy = 0, sx2 = 0, sxy = 0, sw = 0
#     cdef Py_ssize_t i, n = Y.shape[0]
#     cdef double *YY = &Y[0], *WW = &W[0]
#     cdef double xi, yi, wi

#     for i in range(n):
#         xi = i+1
#         yi = YY[i]
#         wi = WW[i]
#         sx += wi * xi
#         sy += wi * yi
#         sxy += wi * xi * yi
#         sx2 += wi * xi * xi
#         sw += wi

#     return (sw * sxy - sx * sy) / (sw * sx2 - sx * sx)

