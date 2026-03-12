
cdef bint _iscontiguousarray(object ob):
    return numpy.PyArray_IS_C_CONTIGUOUS(ob)

cdef bint _isnumpyarray(object ob):
    return numpy.PyArray_CheckExact(ob)

cdef object _asarray(object ob):
    cdef int tp

    if not numpy.PyArray_CheckExact(ob):
        ob = np.array(ob, "d")

    if not numpy.PyArray_IS_C_CONTIGUOUS(ob):
        ob = np.ascontiguousarray(ob)

    tp = numpy.PyArray_TYPE(ob)
    if tp != numpy.NPY_DOUBLE:
        ob = numpy.PyArray_Cast(<numpy.ndarray>ob, numpy.NPY_DOUBLE)

    return ob

def asarray(ob):
    return _asarray(ob)

cdef object _asarray1d(object ob):
    cdef int ndim
    cdef int tp

    if not numpy.PyArray_CheckExact(ob):
        ob = np.array(ob, "d")

    tp = numpy.PyArray_TYPE(ob)
    if tp != numpy.NPY_DOUBLE:
        ob = numpy.PyArray_Cast(<numpy.ndarray>ob, numpy.NPY_DOUBLE)

    ndim = <int>numpy.PyArray_NDIM(ob)
    if ndim == 1:
        return ob
    elif ndim == 0:
        return ob.reshape(1)
    else:
        raise TypeError('number of axes != 1!')

def asarray1d(ob):
    return _asarray1d(ob)


cdef object _asarray2d(object ob):
    cdef int ndim
    cdef int tp

    if not numpy.PyArray_CheckExact(ob):
        ob = np.array(ob, "d")

    tp = numpy.PyArray_TYPE(ob)
    if tp != numpy.NPY_DOUBLE:
        ob = numpy.PyArray_Cast(<numpy.ndarray>ob, numpy.NPY_DOUBLE)

    ndim = <int>numpy.PyArray_NDIM(ob)
    if ndim == 2:
        return ob
    elif ndim == 1:
        return ob.reshape(-1,1)
    else:
        raise TypeError('number of axes > 2!')

def asarray2d(ob):
    return _asarray2d(ob)

cdef object empty_array(Py_ssize_t size):
    cdef numpy.npy_intp n = size
    return numpy.PyArray_EMPTY(1, &n, numpy.NPY_DOUBLE, 0)

cdef object empty_array_i(Py_ssize_t size):
    cdef numpy.npy_intp n = size
    return numpy.PyArray_EMPTY(1, &n, numpy.NPY_INT, 0)

def empty_int_array_i2(size):
    return empty_array_i(size)

cdef object filled_array(Py_ssize_t size, double val):
    cdef numpy.npy_intp n = size
    cdef object ob = numpy.PyArray_EMPTY(1, &n, numpy.NPY_DOUBLE, 0)
    cdef double *data = <double*>numpy.PyArray_DATA(ob)
    cdef Py_ssize_t i

    for i in range(size):
        data[i] = val

    return ob

cdef object empty_array2(Py_ssize_t size1, Py_ssize_t size2):
    cdef numpy.npy_intp[2] n

    n[0] = size1
    n[1] = size2
    return numpy.PyArray_EMPTY(2, &n[0], numpy.NPY_DOUBLE, 0)

cdef object zeros_array2(Py_ssize_t size1, Py_ssize_t size2):
    cdef numpy.npy_intp[2] n

    n[0] = size1
    n[1] = size2
    return numpy.PyArray_ZEROS(2, &n[0], numpy.NPY_DOUBLE, 0)

cdef object diag_matrix(double[::1] V):
    cdef numpy.npy_intp[2] nn
    cdef double[:,::1] A
    cdef Py_ssize_t i, n

    n = nn[0] = nn[1] = V.shape[0]
    A = o = numpy.PyArray_ZEROS(2, &nn[0], numpy.NPY_DOUBLE, 0)
    for i in range(n):
        A[i,i] = V[i]

    return o

cdef object zeros_array(Py_ssize_t size):
    cdef numpy.npy_intp n = size
    return numpy.PyArray_ZEROS(1, &n, numpy.NPY_DOUBLE, 0)

cdef void _normalize(double[::1] a) noexcept nogil:
    cdef Py_ssize_t i, n = a.shape[0]
    cdef double S

    S = 0
    for i in range(n):
        S += fabs(a[i])

    for i in range(n):
        a[i] /= S

def normalize(a):
    _normalize(a)

cdef void _normalize2(double[::1] a) noexcept nogil:
    cdef Py_ssize_t i, n = a.shape[0]
    cdef double v, S

    S = 0
    for i in range(n):
        v = a[i]
        S += v * v

    S = sqrt(S)

    for i in range(n):
        a[i] /= S

def normalize2(a):
    _normalize2(a)

cdef void weighted_sum_rows(double[:,::1] X, double[::1] W, double[::1] Y) noexcept nogil:
    """
    Взвешенная сумма строк матрицы:
    Вход:
       X: матрица (N,n)
       W: массив весов (N)
       Y: массив (N) - результат:
          Y[i] = W[0] * X[0,:] + ... + W[N-1] * X[N-1,:]
    """
    cdef:
        Py_ssize_t N = X.shape[0]
        Py_ssize_t n = X.shape[1]
        Py_ssize_t i, k
        double *Xk
        double *yy = &Y[0]
        double wk, y

    for i in range(n):
        y = 0
        Xk = &X[0,i]
        for k in range(N):
            wk = W[k]
            y += wk * Xk[0]
            Xk += n
        yy[i] = y

cdef _sqrt_array(double *xx, double *yy, Py_ssize_t n):
    cdef Py_ssize_t i

    for i in range(n):
        yy[i] = sqrt(xx[i])

def sqrt_array(X):
    cdef double[::1] xx = X
    cdef Py_ssize_t i, n = xx.shape[0]
    cdef double[::1] yy

    Y = empty_array(n)
    yy = Y
    _sqrt_array(&xx[0], &yy[0], n)
    return Y

cdef double _norm2(double[::1] a):
    cdef Py_ssize_t i, n = a.shape[0]
    cdef double s, v

    s = 0
    for i in range(n):
        v = a[i]
        s += v*v
    return sqrt(s)

def norm2(a):
    return _norm2(a)
