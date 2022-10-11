#
# Least Squares for Simple Linear Models
#

from mlgrad cimport LinearModel


cdef void linear_ls(double[::1] X, double[::1] Y, LinearModel mod):
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

cdef double linear_ls_slope(double[::1] X, double[::1] Y):
    cdef double sx = 0, sy = 0, sx2 = 0, sxy = 0
    cdef Py_ssize_t i, n = X.shape[0]
    cdef double *XX = &X[0], *YY = &Y[0]
    cdef double xi, yi, nn = n

    for i in range(n):
        xi = XX[i]
        yi = YY[i]
        sx += xi
        sy += yi
        sxy += xi * yi
        sx2 += xi * xi

    return (nn * sxy - sx * sy) / (nn * sx2 - sx * sx)

cdef double linear_ls_slope2(double[::1] Y):
    cdef Py_ssize_t i, n = Y.shape[0]
    cdef double xi, yi, nn = n
    cdef double nn = n
    # cdef double sx = nn * (nn + 1) / 2
    cdef double sx_n = (nn + 1) / 2
    cdef double sx2 = nn * (nn + 1) * (2 * nn + 1) / 6
    cdef double sy = 0, sxy = 0
    cdef double *YY = &Y[0]

    for i in range(n):
        # xi = i+1
        yi = YY[i]
        # sx += xi 
        sy += yi
        sxy += (i+1) * yi
        # sxy += xi * yi
        # sx2 += xi * xi

    return (sxy - sx_n * sy) / (sx2 - sx_n * nn)


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

cdef double linear_wls_slope(double[::1] X, double[::1] Y, double[::1] W):
    cdef double sx = 0, sy = 0, sx2 = 0, sxy = 0, sw = 0
    cdef Py_ssize_t i, n = X.shape[0]
    cdef double *XX = &X[0], *YY = &Y[0], *WW = &W[0]
    cdef double xi, yi, wi

    for i in range(n):
        xi = XX[i]
        yi = YY[i]
        wi = WW[i]
        sx += wi * xi
        sy += wi * yi
        sxy += wi * xi * yi
        sx2 += wi * xi * xi
        sw += wi

    return (sw * sxy - sx * sy) / (sw * sx2 - sx * sx)

cdef double linear_wls_slope2(double[::1] Y, double[::1] W):
    cdef double sx = 0, sy = 0, sx2 = 0, sxy = 0, sw = 0
    cdef Py_ssize_t i, n = Y.shape[0]
    cdef double *YY = &Y[0], *WW = &W[0]
    cdef double xi, yi, wi

    for i in range(n):
        xi = i+1
        yi = YY[i]
        wi = WW[i]
        sx += wi * xi
        sy += wi * yi
        sxy += wi * xi * yi
        sx2 += wi * xi * xi
        sw += wi

    return (sw * sxy - sx * sy) / (sw * sx2 - sx * sx)

