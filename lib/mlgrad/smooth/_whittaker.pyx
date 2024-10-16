#
# _whittaker.pyx
#

import numpy as np

cdef _get_D2T_D2(double tau, double[::1] W, double[::1] W2, double[:,::1] S):
    cdef Py_ssize_t i, j, n = S.shape[1]
    cdef double *SS
    # cdef double *WW = &W[0]
    # cdef double *WW2 = &W2[0]

    cdef double *e = &S[0,0]
    cdef double *c = &S[1,0]
    cdef double *d = &S[2,0]
    cdef double *a = &S[3,0]
    cdef double *b = &S[4,0]

    # d
    # SS = &S[2,0]
    d[0] = W2[1]*tau + W[0]
    d[1] = (4*W2[1] + W2[2])*tau + W[1]
    d[n-1] = W2[n-2]*tau + W[n-1]
    d[n-2] = (4*W2[n-3] + W2[n-2])*tau + W[n-2]
    for i in range(2, n-1):
        d[i] = (W2[i-1] + 4*W2[i] + W2[i+1])*tau + W[i]

    # a
    # SS = &S[3,0]
    a[0] = -2*W2[1]*tau
    a[n-2] = -2*W2[n-2]*tau
    for i in range(1,n-2):
        a[i] = (-2*W2[i] - 2*W2[i+1])*tau

    # b
    # SS = &S[4,0]
    for i in range(n-2):
        b[i] = W2[i+1]*tau

    # c
    # SS = &S[1,0]
    c[1] = -2*W2[1]*tau
    c[n-1] = -2*W2[n-1]*tau
    for i in range(2,n-1):
        c[i] = (-2*W2[i-1] - 2*W2[i])*tau

    # e
    # SS = &S[0,0]
    for i in range(2,n):
        e[i] = W2[i-1]*tau
        
cdef _penta_solver(double[:,::1] S, double[::1] Y, double[::1] X):
    cdef Py_ssize_t i, j, n = Y.shape[0]
    cdef double *y = &Y[0]
    cdef double *x = &X[0]
    
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
    
    mu[0] = d[0]
    alpha[0] = a[0] / mu[0]
    beta[0] = b[0] / mu[0]
    zeta[0] = y[0] / mu[0]

    gamma[1] = c[1]
    mu[1]    = d[1] - alpha[0] * gamma[1]
    alpha[1] = (a[1] - beta[0] * gamma[1]) / mu[1]
    beta[1]  = b[1] / mu[1]
    zeta[1]  = (y[1] - zeta[0] * gamma[1]) / mu[1]

    for i in range(2, n-2):
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

    x[n-1] = zeta[n-1]
    x[n-2] = zeta[n-2] - alpha[n-2] * x[n-1]
    i = n-3
    while i >= 0:
        x[i] = zeta[i] - alpha[i] * x[i+1] - beta[i] * x[i+2]
        i -= 1

def penta_solver(Y, tau, W, W2, _zeros=np.zeros):
    N = len(Y)
    S = _zeros((5,N), "d")
    _get_D2T_D2(tau, W, W2, S)
    X = _zeros(N, "d")
    _penta_solver(S, Y*W, X)
    return X
    
def whittaker_smooth_penta(Y, tau=1.0, W=None, W2=None, _ones=np.ones):
    N = Y.shape[0]
    if W is None:
        W = _ones(N, "d")
    if W2 is None:
        W2 = _ones(N, "d")
    X = penta_solver(Y, tau, W, W2)
    return X

cdef _get_D1T_D1(double tau, double[::1] W, double[::1] W2, double[:,::1] S):
    cdef Py_ssize_t i, j, n = S.shape[1]
    
    cdef double *a = &S[0,0]
    cdef double *b = &S[1,0]
    cdef double *c = &S[2,0]
    # cdef double *WW = &W[0]
    # cdef double *WW2 = &W2[0]

    for i in range(1,n):
        a[i] = -tau * W2[i-1]

    for i in range(n-1):
        c[i] = -tau * W2[i]
        
    b[0] = tau * W2[0] + W[0]
    b[n-1] = tau * W2[n-2] + W[n-1]
    for i in range(1,n-1):
        b[i] = tau * (W2[i-1] + W2[i]) + W[i]

        
cdef _tria_solver(double[:,::1] S, double[::1] Y, double[::1] X):
    cdef Py_ssize_t i, n = Y.shape[0]
    
    cdef double *a = &S[0,0]
    cdef double *b = &S[1,0]
    cdef double *c = &S[2,0]
    
    cdef double[:,::1] T = np.zeros((2,n), "d")
    cdef double *c1 = &T[0,0]
    cdef double *y1 = &T[1,0]

    cdef double *x = &X[0]
    cdef double *y = &Y[0]
    

    c1[0] = c[0] / b[0]
    for i in range(1,n):
        c1[i] = c[i] / (b[i] - a[i] * c1[i-1])

    y1[0] = y[0] / b[0]
    for i in range(1,n):
        y1[i] = (y[i] - a[i] * y1[i-1]) / (b[i] - a[i] * c1[i-1])

    x[n-1] = y1[n-1]
    i = n-2
    while i >= 0:
        x[i] = y1[i] - c1[i] * x[i+1]
        i -= 1

def tria_solver(Y, tau, W, W2, _zeros=np.zeros):
    N = len(Y)
    S = _zeros((3,N), "d")
    _get_D1T_D1(tau, W, W2, S)
    X = _zeros(N, "d")
    _tria_solver(S, Y*W, X)
    return X

def whittaker_smooth_tria(Y, tau=1.0, W=None, W2=None, _ones=np.ones):
    N = Y.shape[0]
    if W is None:
        W = _ones(N, "d")
    if W2 is None:
        W2 = _ones(N, "d")
    X = tria_solver(Y, tau, W, W2)
    return X

# cdef class WhittakerSmoother:
#     #
#     def __init__(self, funcs2.Func2 func=None, funcs2.Func2 func2=None, 
#                  h=0.1, n_iter=1000, 
#                  tol=1.0e-6, tau=10.0):
#         if func is None:
#             self.func = funcs2.FuncNorm(funcs.Square())
#         else:
#             self.func = func
#         if func2 is None: 
#             self.func2 = funcs2.FuncDiff2(funcs.Square())
#         else:
#             self.func2 = func2
#         self.n_iter = n_iter
#         self.tol = tol
#         self.h = h
#         self.tau = tau
#         self.Z = None
#         self.qvals = None
#     #
#     #
#     def fit(self, double[::1] X, double[::1] W=None, double[::1] W2=None):
#         cdef double h = self.h
#         cdef double tau = self.tau
#         cdef double tol = self.tol
#         cdef funcs2.Func2 func = self.func
#         cdef funcs2.Func2 func2 = self.func2
#         cdef Py_ssize_t j, N = len(X)
#         # cdef averager.ArrayAverager avg
#         cdef double[::1] Z = np.zeros(N, 'd')
#         cdef double[::1] Z_min = np.zeros(N, 'd')
#         cdef double[::1] E = np.zeros(N, 'd')
#         cdef double[::1] G1 = np.zeros(N, 'd')
#         cdef double[::1] G2 = np.zeros(N, 'd')
#         cdef double[::1] grad = np.zeros(N, 'd')
#         cdef double qval, qval_prev, qval_min, qval_min_prev
#         cdef list qvals
#         cdef int M = 0

#         # avg = averager.ArrayAdaM2()
#         # avg.init(N)
        
#         if self.Z is None:
#             inventory.move(Z, X)
#             # Z = X.copy()
#         else:
#             inventory.move(Z, self.Z)
#             # Z = self.Z
#         inventory.move(Z_min, Z)
#         # Z_min = Z.copy()

#         inventory.sub(E, X, Z)
#         if W is None:
#             qval = func._evaluate(E) / tau
#         else:
#             qval = func._evaluate_ex(E, W) / tau

#         if W2 is None:
#             qval += func2._evaluate(Z)
#         else:
#             qval += func2._evaluate_ex(Z, W2)

#         qvals = [qval]
    
#         qval_min = qval
#         qval_min_prev = 2.0 * qval_min

#         for K in range(self.n_iter):
#             qval_prev = qval
#             # Z_prev = Z.copy()

#             if W is None:
#                 func._gradient(E, G1)
#             else:
#                 func._gradient_ex(E, G1, W)
        
#             if W2 is None:
#                 func2._gradient(Z, G2)
#             else:
#                 func2._gradient_ex(Z, G2, W2)
                
#             for j in range(N):
#                 grad[j] = -G1[j] / tau + G2[j]
#             inventory.normalize(grad)

#             # avg.update(grad, h)
            
#             for j in range(N):
#                 Z[j] -= h * grad[j] * N

#             # inventory.isub(Z, avg.array_average)

#             inventory.sub(E, X, Z)

#             if W is None:
#                 qval = func._evaluate(E) / tau
#             else:
#                 qval = func._evaluate_ex(E, W) / tau

#             if W2 is None:
#                 qval += func2._evaluate(Z)
#             else:
#                 qval += func2._evaluate_ex(Z, W2)
                
#             qvals.append(qval)

#             if qval < qval_min:
#                 qval_min_prev = qval_min
#                 qval_min = qval
#                 inventory.move(Z_min, Z)
#                 # for j in range(N):
#                 #     if Z_min[j] < 0:
#                 #         Z_min[j] = 0
                
#             if fabs(qval - qval_prev) / (1.0 + fabs(qval_min)) < tol:
#                 break

#             if fabs(qval_min - qval_min_prev) / (1.0 + fabs(qval_min)) < tol:
#                 break
                
#             if qval > qval_prev:
#                 M += 1
                
#             if M > 10:
#                 break

#         self.Z = Z_min
#         # self.Z = Z
#         self.qval = qval_min
#         self.K = K+1
#         self.delta_qval = fabs(qval_min - qval_min_prev)
#         self.qvals = qvals
