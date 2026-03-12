
cdef void _covariance_matrix(double[:, ::1] X, double[::1] loc, double[:,::1] S) noexcept nogil:
    cdef Py_ssize_t i, j
    cdef Py_ssize_t n = X.shape[1], N = X.shape[0]
    cdef double s, loc_i, loc_j
    #
    for i in range(n):
        loc_i = loc[i]
        for j in range(n):
            loc_j = loc[j]
            s = 0
            for k in range(N):
                s += (X[k,i] - loc_i) * (X[k,j] - loc_j)
            S[i,j] = s / N

def covariance_matrix(X, loc=None, S=None):
    X = _asarray(X)
    n = X.shape[1]
    if S is None:
        S = empty_array2(n, n)
    else:
        S = _asarray(S)
        if S.shape[0] != n and S.shape[1] != n:
            raise TypeError(f"ivalid shape of S: {S.shape}")
    if loc is None:
        loc = zeros_array(n)
    else:
        loc = _asarray(loc)
    _covariance_matrix(X, loc, S)
    return S

cdef void _covariance_matrix_weighted(double[:, ::1] X, double[::1] W, 
                                      double[::1] loc, double[:,::1] S) noexcept nogil:
    cdef Py_ssize_t i, j
    cdef Py_ssize_t n = X.shape[1], N = X.shape[0]
    cdef double s, loc_i, loc_j
    #
    for i in range(n):
        loc_i = loc[i]
        for j in range(n):
            loc_j = loc[j]
            s = 0
            for k in range(N):
                s += W[k] * (X[k,i] - loc_i) * (X[k,j] - loc_j)
            S[i,j] = s

# cdef void _covariance_matrix_weighted(
#             double *X, const double *W, const double *loc, double *S, 
#             const Py_ssize_t n, const Py_ssize_t N) noexcept nogil:

#     cdef Py_ssize_t i, j, k
#     cdef double s, loc_i, loc_j
#     cdef double *X_ki
#     cdef double *X_kj
#     cdef double *S_i
#     cdef double *S_j

#     S_i = S_j = S
#     for i in range(n):
#         loc_i = loc[i]
#         for j in range(i, n):
#             loc_j = loc[j]
#             X_kj = X + j
#             X_ki = X + i

#             s = 0
#             for k in range(N):
#                 s += W[k] * (X_ki[0] - loc_i) * (X_kj[0] - loc_j)
#                 X_ki += n
#                 X_kj += n

#             S_i[j] = S_j[i] = s
#             S_j += n
#         S_i += n

def covariance_matrix_weighted(X, W, loc=None, S=None):
    X = _asarray(X)
    n = X.shape[1]
    W = _asarray(W)
    if S is None:
        S = empty_array2(n, n)
    else:
        S = _asarray(S)
        if S.shape[0] != n and S.shape[1] != n:
            raise TypeError(f"ivalid shape of S: {S.shape}")
    if loc is None:
        loc = zeros_array(n)
    else:
        loc = _asarray(loc)
    _covariance_matrix_weighted(X, W, loc, S)
    return S

cdef _inverse_matrix(double[:,::1] AM, double[:,::1] IM):
    """
    Returns the inverse of the passed in matrix.
        :param A: The matrix to be inversed

        :return: The inverse of the matrix A
    """
    # Section 1: Make sure A can be inverted.
    # check_squareness(A)
    # check_non_singular(A)

    # Section 2: Make copies of A & I, AM & IM, to use for row operations
    cdef Py_ssize_t fd, i, j, n = len(AM)
    cdef double fdScaler, crScaler
    cdef double *AM_fd
    cdef double *AM_i
    cdef double *IM_fd
    cdef double *IM_i

    _clear(&IM[0,0], n*n)
    for i in range(n):
        IM[i,i] = 1

    # Section 3: Perform row operations
    for fd in range(n): # fd stands for focus diagonal
        # FIRST: scale fd row with fd inverse.
        AM_fd = &AM[fd,0]
        IM_fd = &IM[fd,0]
        fdScaler = 1.0 / AM[fd,fd]
        for j in range(n): # Use j to indicate column looping.
            AM_fd[j] *= fdScaler
            IM_fd[j] *= fdScaler
        # SECOND: operate on all rows except fd row as follows:
        for i in range(n): # *** skip row with fd in it.
            if i == fd:
                continue
            AM_i = &AM[i,0]
            IM_i = &IM[i,0]
            crScaler = AM[i,fd] # cr stands for "current row".
            for j in range(n): # cr - crScaler * fdRow, but one element at a time.
                AM_i[j] -= crScaler * AM_fd[j]
                IM_i[j] -= crScaler * IM_fd[j]

    # Section 4: Make sure that IM is an inverse of A within the specified tolerance
    # if check_matrix_equality(I,matrix_multiply(A,IM),tol):
    #     return IM
    # else:
    #     raise ArithmeticError("Matrix inverse out of tolerance.")

def inverse_matrix(A, copy=1):
    n, m = A.shape
    if copy:
        AM = A.copy()
    else:
        AM = A
    IM = empty_array2(n, n)
    _inverse_matrix(AM, IM)
    return IM

cdef double _mahalanobis_norm_one(double *S, const double *x, 
                                  const Py_ssize_t n) noexcept nogil:
    cdef double x1, x2
    cdef Py_ssize_t i, j
    cdef double s, x_i, s_i
    cdef double *S_i

    if n == 2:
        x1 = x[0]
        x2 = x[1]
        return S[0] * x1 * x1 + \
               S[3] * x2 * x2 + \
               2 * (S[1] * x1 * x2)

    s = 0
    S_i = S
    for i in range(n):
        x_i = x[i]
        s += S_i[i] * x_i * x_i

        s_i = 0
        for j in range(i+1, n):
            s_i += S_i[j] * x[j]

        s += 2 * x_i * s_i

        S_i += n

    return s

cdef _mahalanobis_norm(double[:,::1] S, double[:,::1] X, double[::1] Y):
    cdef Py_ssize_t n = S.shape[0]
    cdef Py_ssize_t k, N = X.shape[0]

    # if X.shape[1] != n:
    #     raise TypeError("X.shape[1] != S.shape[0]")
    # if Y.shape[0] != N:
    #     raise TypeError("Y.shape[1] != X.shape[0]")

    for k in range(N):
        Y[k] = _mahalanobis_norm_one(&S[0,0], &X[k,0], n)

def mahalanobis_norm(S, X):
    Y = empty_array(X.shape[0])
    _mahalanobis_norm(S, X, Y)
    return Y

cdef _mahalanobis_distance(double[:,::1] X, double[:,::1] S, double[::1] c, double[::1] Y):
    cdef Py_ssize_t n = S.shape[0]
    cdef Py_ssize_t k, N = X.shape[0]
    cdef double[::1] v = empty_array(n)
    cdef double *YY = &Y[0]

    for k in range(N):
        _sub(&v[0], &X[k,0], &c[0], n)
        YY[k] = _mahalanobis_norm_one(&S[0,0], &v[0], n)

def mahalanobis_distance(X, S, c, Y=None):
    # if X.shape[1] != n:
    #     raise TypeError("X.shape[1] != S.shape[0]")
    # if Y.shape[0] != N:
    #     raise TypeError("Y.shape[1] != X.shape[0]")
    # if c.shape[0] != n:
    #     raise TypeError("c.shape[0] != S.shape[0]")

    if Y is None:
        Y = empty_array(X.shape[0])
    _mahalanobis_distance(X, S, c, Y)
    return Y

cdef void _scatter_matrix_weighted(double[:,::1] X, double[::1] W, double[:,::1] S) noexcept nogil:
    """
    Вычисление взвешенной ковариационной матрицы
    Вход:
       X: матрица (N,n)
       W: массив весов (N)
    Результат:
       S: матрица (n,n):
          S = (W[0] * outer(X[0,:],X[0,:]) + ... + W[N-1] * outer(X[N-1,:],X[N-1,:]))
    """
    cdef:
        Py_ssize_t N = X.shape[0]
        Py_ssize_t n = X.shape[1]
        Py_ssize_t i, j, k
        double s, xkj
        double *Xk
        double *ss

    ss = &S[0,0]
    for i in range(n):
        for j in range(n):
            Xk = &X[0,0]
            s = 0
            for k in range(N):
                s += W[k] * Xk[i] * Xk[j]
                Xk += n
            ss[j] = s
        ss += n

cdef void _scatter_matrix(double[:,::1] X, double[:,::1] S) noexcept nogil:
    """
    Вычисление ковариационной матрицы
    Вход:
       X: матрица (N,n)
    Результат:
       S: матрица (n,n):
          S = (1/N) X.T @ X
    """
    cdef:
        Py_ssize_t N = X.shape[0]
        Py_ssize_t n = X.shape[1]
        Py_ssize_t i, j, k
        double s
        double *Xk
        double *ss

    ss = &S[0,0]
    for i in range(n):
        for j in range(n):
            Xk = &X[0,0]
            s = 0
            for k in range(N):
                s += Xk[i] * Xk[j]
                Xk += n
            ss[j] = s
        ss += n

    ss = &S[0,0]
    for i in range(n):
        for j in range(n):
            ss[j] /= N
        ss += n

def scatter_matrix(double[:,::1] X):
    cdef Py_ssize_t n = X.shape[1]
    cdef object S = zeros_array2(n, n)
    _scatter_matrix(X, S)
    return S

def scatter_matrix_weighted(double[:,::1] X, double[::1] W):
    cdef Py_ssize_t n = X.shape[1]
    cdef object S = zeros_array2(n, n)
    _scatter_matrix_weighted(X, W, S)
    return S
