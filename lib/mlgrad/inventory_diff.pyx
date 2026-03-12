cdef void _diff4(double *x, double *y, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i

    # y[0] = 0
    # y[1] = 0
    # y[n-2] = 0
    # y[n-1] = 0
    for i in range(2,n-2):
        y[i-2] = x[i-2] - 4*x[i-1] + 6*x[i] - 4*x[i+1] + x[i+2]

cdef void _diff3(double *x, double *y, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i

    # y[0] = 0
    # y[n-1] = 0
    # y[n-2] = 0
    for i in range(1,n-2):
        y[i-1] = x[i-1] - 3*x[i] + 3*x[i+1] - x[i+2]

cdef void _diff2(double *x, double *y, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i

    # y[0] = 0
    # y[n-1] = 0
    for i in range(1, n-1):
        y[i-1] = x[i-1] - 2*x[i] + x[i+1]

cdef void _diff2w2(double *x, double *w, double *y, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i

    y[0] = w[0] * x[0] - 2*w[0] * x[1] + w[0] * x[2]
    y[1] = -2*w[0] * x[0] + (4*w[0]+w[1]) * x[1] + \
           (-2*w[0]-2*w[1]) * x[2] + w[1] * x[3]

    for i in range(2,n-2):
        y[i] = w[i-2] * x[i-2] + \
               (-2*w[i-2] - 2*w[i-1]) * x[i-1] + \
               (w[i-2] + 4*w[i-1] + w[i]) * x[i] + \
               (-2*w[i-1]-2*w[i])*x[i+1] + \
               w[i] * x[i+2]

    y[n-2] = w[n-4] * x[n-4] + (-2*w[n-4] -2*w[n-3]) * x[n-3] + \
             (w[n-4] + 4*w[n-3]) * x[n-2] - 2*w[n-4] * x[n-1] 
    y[n-1] = w[n-3]*x[n-3] - 2*w[n-3]*x[n-2] + w[n-3]*x[n-1]

cdef void _diff1(double *x, double *y, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i

    # y[n-1] = 0
    for i in range(n-1):
        y[i] = x[i] - x[i+1]


def diff4(double[::1] a, double[::1] b=None):
    cdef Py_ssize_t n = a.shape[0]
    cdef bint flag = 0
    if b is None:
        b = empty_array(n-4)
        flag = 1
    _diff4(&a[0], &b[0], n)
    if flag:
        return b.base
    else:
        return np.asarray(b)

def diff3(double[::1] a, double[::1] b=None):
    cdef Py_ssize_t n = a.shape[0]
    cdef bint flag = 0
    if b is None:
        b = empty_array(n-3)
        flag = 1
    _diff3(&a[0], &b[0], n)
    if flag:
        return b.base
    else:
        return np.asarray(b)

def diff2(double[::1] a, double[::1] b=None):
    cdef Py_ssize_t n = a.shape[0]
    cdef bint flag = 0
    if b is None:
        b = empty_array(n-2)
        flag = 1
    _diff2(&a[0], &b[0], n)
    if flag:
        return b.base
    else:
        return np.asarray(b)

def diff2w2(double[::1] a, double[::1] w, double[::1] b=None):
    cdef Py_ssize_t n = a.shape[0]
    cdef bint flag = 0
    if b is None:
        b = empty_array(n)
        flag = 1
    _diff2w2(&a[0], &w[0], &b[0], n)
    if flag:
        return b.base
    else:
        return np.asarray(b)

def diff1(double[::1] a, double[::1] b=None):
    cdef Py_ssize_t n = a.shape[0] 
    cdef bint flag = 0
    if b is None:
        b = empty_array(n-1)
        flag = 1
    _diff1(&a[0], &b[0], n)
    if flag:
        return b.base
    else:
        return np.asarray(b)
