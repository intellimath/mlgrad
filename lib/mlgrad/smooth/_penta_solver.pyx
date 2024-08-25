
# cdef class SDMatrix:
#     #
#     cdef __init__(self, double[::1] S):
#         self.nr, self.nc = S.shape[0], S.shape[1]
#         self.S = S
#     #
#     cdef double get(self, Py_ssize_t i, Py_ssize_t j):
#         cdef Py_ssize_t m = self.nc // 2
#         if i > j:
#             i, j = j, i
#         if j - i > m:
#             return 0
#         else:
#             return self.S[j-i,]
        

cdef _get_D2T_D2(double[::1] S):
    cdef Py_ssize_t i, j, n = S.shape[1]

    # d
    S[2,0] = 1
    S[2,1] = 5
    S[2,n-1] = 1
    S[2,n-2] = 5
    for i in range(2, n-2):
        S[2,i] = 6

    # a
    S[0,1] = -2
    S[n-2, n-1] = -2
    for i in range(1,n-2):
        S[i,i+1] = -4

    # c
    S[1,0] = -2
    S[n-1,n-2] = -2
    for i in range(2,n-1):
        S[i+1,i] = -4

    # b
    for i in range(0, n-2):
        S[i,i+2] = 1

    # e
    for i in range(2):
        S[i,i-2] = 1

cdef double penta_solver(double S[:,::1], double[::1] Y, double[::1] X):
    cdef Py_ssize_t i, j, n = Y.shape[0]
    cdef double *y = &Y[0]
    
    cdef double *e = &S[0,0]
    cdef double *c = &S[1,0]
    cdef double *d = &S[2,0]
    cdef double *a = &S[3,0]
    cdef double *b = &S[4,0]
    
    cdef double[:,::1] T = np.zeros((5,n), 'd')
    cdef double *mu =    &T[0,0]
    cdef double *alpha = &T[1,0]
    cdef double *beta =  &T[2,0]
    cdef double *gamma = &T[3,0]
    cdef double *zeta  = &T[4,0]
    
    cdef double *x = &X[0]

    mu[0] = d[0]
    alpha[0] = a[0] / mu[0]
    beta[0] = b[0] / mu[0]
    zeta[0] = y[0] / mu[0]

    gamma[1] = c[1]
    mu[1]    = d[1] - alpha[0] * gamma[1]
    alpha[1] = (a[1] - beta[0] * gamma[1]) / mu[1]
    beta[1]  = b[1] / mu[1]
    zeta[1]  = (y[1] - zeta[0] * gamma[1]) / mu[1]

    for i in range(2, n-3):
        gamma[i] = c[i] - alpha[i-2] * e[i]
        mu[i]    = d[i] - beta[i-2] * e[i] - alpha[i-1] * gamma[i]
        alpha[i] = (a[i] - beta[i-1] * gamma[i]) / mu[i]
        beta[i]  = b[i] / mu[i]
        zeta[i]  = (y[i] - zeta[i-2] * e[i] - zeta[i-1] * gamma[i]) / mu[i]

    gamma[n-2] = c[n-2] - alpha[n-4] * e[n-2]
    mu[n-2]    = d[n-2] - beta[n-4] * e[n-2] - alpha[n-3] * gamma[n-2]
    alpha[n-2] = (a[n-2] - beta[n-3] * gamma[n-2]) / mu[n-2]
    gamma[n-1] = c[n-1] - alpha[n-3] * e[n-1]
    mu[n-1]    = d[n-1] - beta[n-3] * e[n-1] - alpha[n-2] * gamma[n-1]
    zeta[n-2]  = (y[n-2] - zeta[n-3] * e[n-2] - zeta[n-3] * gamma[n-2]) / mu[n-2]
    zeta[n-1]  = (y[n-1] - zeta[n-2] * e[n-1] - zeta[n-2] * gamma[n-1]) / mu[n-1]

    x[n-1] = z[n-1]
    x[n-2] = z[n-2] - alpha[n-2] * x[n-1]
    i = n-3
    while i >= 0:
        x[i] = z[i] - alpha[i] * x[i+1] - beta[i] * x[i+2]
        i -= 1
        

        