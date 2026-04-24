
cdef double _mean(double *a, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double s = 0

    for i in range(n):
        s += a[i]
    return s/n

cdef double _average(double *x, double *w, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double W = 0, wi, s = 0

    for i in range(n):
        wi = w[i]
        s += x[i] * wi
        W += wi
    return s / W

cdef void _average2(double[:,::1] x, double[::1] w, double[::1] y) noexcept nogil:
    cdef Py_ssize_t i, j
    cdef Py_ssize_t N=x.shape[0], n=x.shape[1]
    cdef double W, s = 0

    W = 0
    for j in range(N):
        W += w[j]

    for i in range(n):
        s = 0
        # k = i
        for j in range(N):
            s += x[j,i] * w[j]
        # k += n
        y[i] = s / W

# cdef void _average2_t(double *x, double *w, double *y, Py_ssize_t N, Py_ssize_t n) noexcept nogil:
#     cdef Py_ssize_t i, j, k
#     cdef double W = 0, wi, s = 0

#     W = 0
#     for j in range(N):
#         W += w[j]

#     for i in range(n):
#         s = 0
#         k = i
#         for j in range(N):
#             s += x[k] * w[j]
#             k += n
#         y[i] = s / W

def average(double[::1] x, double[::1] w):
    return _average(&x[0], &w[0], x.shape[0])

def average2(double[:,::1] x, double[::1] w):
    cdef double[::1] y

    yy = empty_array(x.shape[1])
    y = yy
    _average2(x, w, y)
    return yy

# def average2_t(double[:,::1] x, double[::1] w):
#     cdef double[::1] y

#     yy = np.empty(x.shape[0])
#     y = yy
#     _average2_t(&x[0,0], &w[0], &y[0], x.shape[0], x.shape[1])
#     return yy

cdef double _std(double *a, double mu, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double v, s = 0

    for i in range(n):
        v = a[i] - mu
        s += v*v
    return sqrt(s/n)

cdef double _mad(double *a, double mu, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double s = 0

    for i in range(n):
        s += fabs(a[i] - mu)
    return s/n

cdef double quick_select(double *a, Py_ssize_t n): # noexcept nogil:
    cdef Py_ssize_t low, high
    cdef Py_ssize_t median
    cdef Py_ssize_t middle, ll, hh
    cdef double t
    cdef bint is2 = n % 2

    low = 0
    high = n-1
    median = (low + high) // 2
    while 1:
        if high <= low: # One element only
            return a[median]

        if high == low + 1:  # Two elements only
            if a[low] > a[high]:
                t = a[low]; a[low] = a[high]; a[high] = t
            return a[median]

        # Find median of low, middle and high items; swap into position low
        middle = (low + high) // 2
        if a[middle] > a[high]:
            t = a[middle]; a[middle] = a[high]; a[high] = t
        if a[low] > a[high]:
            t = a[low]; a[low] = a[high]; a[high] = t
        if a[middle] > a[low]:
            t = a[middle]; a[middle] = a[low]; a[low] = t

        # Swap low item (now in position middle) into position (low+1)
        # swap(&a[middle], &a[low+1])
        t = a[middle]; a[middle] = a[low+1]; a[low+1] = t

        # Nibble from each end towards middle, swapping items when stuck
        ll = low + 1;
        hh = high;
        while 1:
            ll += 1
            while a[low] > a[ll]:
                ll += 1

            hh -= 1
            while a[hh]  > a[low]:
                hh -= 1

            if hh < ll:
                break

            # swap(&a[ll], &a[hh])
            t = a[ll]; a[ll] = a[hh]; a[hh] = t

        # Swap middle item (in position low) back into correct position
        t = a[low]; a[low] = a[hh]; a[hh] = t
        # swap(&a[low], &a[hh])

        # Re-set active partition
        if hh <= median:
            low = ll
        if hh >= median:
            high = hh - 1

cdef double _kth_smallest(double *a, Py_ssize_t n, Py_ssize_t k) noexcept nogil:
    cdef Py_ssize_t i, j, l, m
    cdef double x, temp
    cdef double *ai
    cdef double *aj

    l = 0
    m = n-1
    while l < m:
        x = a[k]
        i = l
        j = m
        while 1:
            while a[i] < x:
                i += 1
            while x < a[j]:
                j -= 1
            if i <= j:
                ai = &a[i]
                aj = &a[j]
                temp = ai[0];
                ai[0] = aj[0]
                aj[0] = temp
                # swap(&a[i],&a[j])
                i += 1
                j -= 1
            if i > j:
                break

        if j < k:
            l = i
        if k < i:
            m = j

    return a[k]

cdef double _median_1d(double[::1] x): # noexcept nogil:
    cdef Py_ssize_t n2, n = x.shape[0]
    cdef double mv1, mv2

    if n % 2:
        n2 = (n-1) // 2
        mv1 = _kth_smallest(&x[0], n, n2)
        return mv1
    else:
        n2 = n // 2
        mv1 = _kth_smallest(&x[0], n, n2)
        mv2 = _kth_smallest(&x[0], n, n2-1)
        return (mv1 + mv2) / 2

cdef double _quantile_1d(double[::1] x, double alpha): # noexcept nogil:
    cdef Py_ssize_t nq, n = x.shape[0]
    cdef double mv1, mv2, an

    an = alpha * n
    nq = <Py_ssize_t>floor(an)

    mv2 = _kth_smallest(&x[0], n, nq+1)
    mv1 = _kth_smallest(&x[0], n, nq)
    return mv1 + (an - nq) * (mv2 - mv1)

cdef double _iqr_1d(double[::1] x): # noexcept nogil:
    cdef Py_ssize_t nq25, nd75, n = x.shape[0]
    cdef double mv1, mv2, an, v1, v2

    an = 0.75 * n
    nq75 = <Py_ssize_t>floor(an)
    mv2 = _kth_smallest(&x[0], n, nq75+1)
    mv1 = _kth_smallest(&x[0], n, nq75)
    v1 =  mv1 + (an - nq75) * (mv2 - mv1)

    an = 0.25 * n
    nq25 = <Py_ssize_t>floor(an)
    mv2 = _kth_smallest(&x[0], nq75, nq25+1)
    mv1 = _kth_smallest(&x[0], nq75, nq25)
    v2 =  mv1 + (an - nq25) * (mv2 - mv1)

    return (v2 - v1) / 1.349

cdef double _median_absdev_1d(double[::1] x, double mu):
    cdef Py_ssize_t i, n = x.shape[0]
    cdef double[::1] temp = empty_array(n)

    for i in range(n):
        temp[i] = fabs(x[i] - mu)
    return _median_1d(temp) / 0.6748

cdef void _median_2d(double[:,::1] x, double[::1] y): # noexcept nogil:
    cdef Py_ssize_t i, N = x.shape[0], n = x.shape[1]
    cdef double[::1] temp = empty_array(n)

    for i in range(N):
        _move(&temp[0], &x[i,0], n)
        # temp = x[i].copy()
        y[i] = _median_1d(temp)

cdef void _median_2d_t(double[:,::1] x, double[::1] y): # noexcept nogil:
    cdef Py_ssize_t i, N = x.shape[0], n = x.shape[1]
    cdef double[::1] temp = empty_array(N)

    for i in range(n):
        _move_t(&temp[0], &x[0,i], N, n)
        y[i] = _median_1d(temp)

cdef void _median_absdev_2d(double[:,::1] x, double[::1] mu, double[::1] y):
    cdef Py_ssize_t i, j, n = x.shape[0], m = x.shape[1]
    cdef double[::1] temp = empty_array(m)
    cdef double mu_i

    for i in range(n):
        mu_i = mu[i]
        for j in range(m):
            temp[j] = fabs(x[i,j] - mu_i)
        y[i] = _median_1d(temp) / 0.6748

cdef void _median_absdev_2d_t(double[:,::1] x, double[::1] mu, double[::1] y):
    cdef Py_ssize_t i, j, n = x.shape[0], m = x.shape[1]
    cdef double[::1] temp = empty_array(n)
    cdef double mu_j

    for j in range(m):
        mu_j = mu[j]
        for i in range(n):
            temp[i] = fabs(x[i,j] - mu_j)
        y[j] = _median_1d(temp) / 0.6748

cdef double _robust_mean_1d(double[::1] x, double tau): #noexcept nogil:
    cdef Py_ssize_t i, j, q, n = x.shape[0]
    cdef double s, v, mu, std
    cdef double[::1] temp = empty_array(n)

    _move(&temp[0], &x[0], n)
    mu = _median_1d(temp)
    std = _median_absdev_1d(x, mu)

    s = 0
    q = 0
    # tau /= 0.6745
    for i in range(n):
        v = x[i]
        if (v <= mu + tau*std) and (v >= mu - tau*std):
            s += v
            q += 1
    return s / q

cdef void _robust_mean_2d_t(double[:,::1] x, double tau, double[::1] y):
    cdef Py_ssize_t i, j, q, n = x.shape[0], m = x.shape[1]
    cdef double[::1] mu = empty_array(m)
    cdef double[::1] std = empty_array(m)
    cdef double s, v, mu_j, std_j

    _median_2d_t(x, mu)
    _median_absdev_2d_t(x, mu, std)

    # tau /= 0.6745
    for j in range(m):
        mu_j = mu[j]
        std_j = std[j]
        s = 0
        q = 0
        for i in range(n):
            v = x[i,j]
            if (v <= mu_j + tau*std_j) and (v >= mu_j - tau*std_j):
                s += v
                q += 1
        y[j] = s / q

cdef void _robust_mean_2d(double[:,::1] x, double tau, double[::1] y):
    cdef Py_ssize_t i, j, q, n = x.shape[0], m = x.shape[1]
    cdef double[::1] mu = empty_array(n)
    cdef double[::1] std = empty_array(n)
    cdef double s, v, mu_i, std_i

    _median_2d(x, mu)
    _median_absdev_2d(x, mu, std)

    # tau /= 0.6745
    for i in range(n):
        mu_i = mu[i]
        std_i = std[i]
        s = 0
        q = 0
        for j in range(m):
            v = x[i,j]
            if (v <= mu_i + tau*std_i) and (v >= mu_i - tau*std_i):
                s += v
                q += 1
        y[i] = s / q

cdef void _zscore(double *a, double *b, Py_ssize_t n):
    cdef Py_ssize_t i
    cdef double mu = _mean(a, n)
    cdef double sigma = _std(a, mu, n)

    for i in range(n):
        b[i] = (a[i] - mu) / sigma

cdef void _modified_zscore(double *a, double *b, Py_ssize_t n):
    cdef Py_ssize_t i
    cdef double mu, sigma
    cdef double[::1] aa = empty_array(n)
    # cdef double[::1] ss = empty_array(n)

    _move(&aa[0], &a[0], n)
    mu = _median_1d(aa)
    for i in range(n):
        aa[i] = fabs(aa[i] - mu)
    sigma = _median_1d(aa)
    sigma /= 0.6748

    for i in range(n):
        b[i] = (a[i] - mu) / sigma

cdef void _modified_zscore_mu(double *a, double *b, Py_ssize_t n, double mu):
    cdef Py_ssize_t i
    cdef double sigma
    cdef double[::1] aa = empty_array(n)

    # inventory._move(&aa[0], &a[0], n)
    for i in range(n):
        aa[i] = fabs(a[i] - mu)
    sigma = _median_1d(aa)
    sigma /= 0.6748

    for i in range(n):
        b[i] = (a[i] - mu) / sigma

cdef double _max(double *x, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double v, max_val = _double_min

    for i in range(n):
        v = x[i]
        if v > max_val:
            max_val = v

    return max_val

cdef double _min(double *x, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i = 1
    cdef double v, x_min = _double_min

    for i in range(n):
        v = x[i]
        if v < x_min:
            x_min = v

    return x_min

def zscore(a, b=None):
    cdef double[::1] aa = _asarray(a)
    cdef double[::1] bb
    cdef Py_ssize_t n = a.shape[0] 
    cdef bint flag = 0

    if b is None:
        bb = b = empty_array(n)
        flag = 1
    else:
        bb = b
    _zscore(&aa[0], &bb[0], aa.shape[0])
    return b

def modified_zscore2(a, b=None):
    cdef double[:,::1] aa = _asarray(a)
    cdef double[:,::1] bb
    cdef Py_ssize_t i, n = aa.shape[0], m = aa.shape[1]
    cdef bint flag = 0
    if b is None:
        bb = b = empty_array2(n,m)
        flag = 1
    else:
        bb = b
    for i in range(n):
        _modified_zscore(&aa[i,0], &bb[i,0], m)
    if flag:
        return b
    else:
        return _asarray(bb)

def modified_zscore(a, b=None, mu=None):
    cdef double[::1] aa = _asarray(a)
    cdef double[::1] bb
    cdef Py_ssize_t n = a.shape[0]
    cdef double d_mu
    cdef bint flag = 0
    if b is None:
        bb = b = empty_array(n)
        flag = 1
    else:
        bb = b
    if mu is None:
        _modified_zscore(&aa[0], &bb[0], n)
    else:
        d_mu = mu
        _modified_zscore_mu(&aa[0], &bb[0], n, d_mu)
    if flag:
        return b
    else:
        return _asarray(bb)

def median(a, axis=None, out=None, overwrite_input=False):
    # can't be reasonably be implemented in terms of percentile as we have to
    # call mean to not break astropy
    # a = np.asanyarray(a)
    a = _asarray(a)

    # Set the partition indexes
    if axis is None:
        sz = a.size
    else:
        sz = a.shape[axis]
    if sz % 2 == 0:
        szh = sz // 2
        kth = [szh - 1, szh]
    else:
        kth = [(sz - 1) // 2]

    # We have to check for NaNs (as of writing 'M' doesn't actually work).
    # supports_nans = np.issubdtype(a.dtype, np.inexact) or a.dtype.kind in 'Mm'
    # if supports_nans:
    #     kth.append(-1)

    if not overwrite_input:
        a = a.copy()
    if axis is None:
        part = a.ravel()
        part.partition(kth)
    else:
        a.partition(kth, axis=axis)
        part = a

    if part.shape == ():
        # make 0-D arrays work
        return part.item()
    if axis is None:
        axis = 0

    indexer = [slice(None)] * part.ndim
    index = part.shape[axis] // 2
    if part.shape[axis] % 2 == 1:
        # index with slice to allow mean (below) to work
        indexer[axis] = slice(index, index + 1)
    else:
        indexer[axis] = slice(index - 1, index + 1)
    indexer = tuple(indexer)

    # Use mean in both odd and even case to coerce data type,
    # using out array if needed.
    rout = part[indexer].mean(axis=axis, out=out)
    # if supports_nans and sz > 0:
    #     # If nans are possible, warn and replace by nans like mean would.
    #     rout = np.lib._utils_impl._median_nancheck(part, rout, axis)

    return rout

def median_1d(x, copy=True):
    if copy:
        xx = x.copy()
    else:
        xx = x
    return _median_1d(_asarray(xx))

def quantile_1d(x, alpha, copy=True):
    if copy:
        xx = x.copy()
    else:
        xx = x
    return _quantile_1d(_asarray(xx), alpha)

def iqr_1d(x, copy=True):
    if copy:
        xx = x.copy()
    else:
        xx = x
    return _iqr_1d(_asarray(xx))

def median_absdev_1d(x, mu=None):
    xx = _asarray(x)
    if mu is None:
        xx = xx.copy()
        mu = _median_1d(xx)
    return _median_absdev_1d(xx, mu)

def median_2d_t(x):
    y = empty_array(x.shape[1])
    _median_2d_t(_asarray(x), y)
    return y

def median_2d(x):
    y = empty_array(x.shape[0])
    _median_2d(_asarray(x), y)
    return y

def median_absdev_2d(x, mu):
    y = empty_array(x.shape[0])
    _median_absdev_2d(_asarray(x), mu, y)
    return y

def median_absdev_2d_t(x, mu):
    y = empty_array(x.shape[1])
    _median_absdev_2d_t(_asarray(x), mu, y)
    return y

def robust_mean_1d(x, tau):
    return _robust_mean_1d(_asarray(x), tau)

# def robust_mean_1d(x, tau):
#     x = _asarray(x)

#     mu = median(x)
#     std = median(abs(x - mu))

#     tau /= 0.6745
#     tau_std = tau * std
#     return  x[(v <= mu + tau_std) & (v >= mu - tau_std)].mean()

def robust_mean_2d(x, tau):
    y = empty_array(x.shape[0])
    _robust_mean_2d(_asarray(x), tau, y)
    return y

def robust_mean_2d_t(x, tau):
    y = empty_array(x.shape[1])
    _robust_mean_2d_t(_asarray(x), tau, y)
    return y
