#
# _whittaker.pyx
#

import numpy as np
import scipy
import mlgrad.inventory as inventory

#
#       Banded array (2*d+1,n)
#
#       c0 c1 c2 c3 ... c_n-5 2*d (d=4)
#       c0 c1 c2 c3 ... c_n-4 
#       b0 b1 b2 b3 ... b_n-3 
#       a0 a1 a2 a3 ... a_n-2 d+1
#    d: D0 D1 D2 D3 ... D_n-1 d 
#       a1 a2 a3 a4 ... a_n-1 d-1
#       b2 b3 b4 b5 ... b_n-1 
#       c3 c4 c5 c6 ... c_n-1
#       d4 d5 d6 d7 ... d_n-1 0
#
#   Example of `ab` (shape of a is (6,6), `u` =1, `l` =2)::
#
#        *    a01  a12  a23  a34  a45
#        a00  a11  a22  a33  a44  a55
#        a10  a21  a32  a43  a54   *
#        a20  a31  a42  a53   *    *

cdef set_diagonal(double[:,::1] S, double[::1] W):
    cdef Py_ssize_t i, j, n = S.shape[1], m = S.shape[0] // 2
    cdef double *d

    d = &S[m,0]
    for i in range(n):
        d[i] = W[i]

cdef add_diagonal(double[:,::1] S, double[::1] W):
    cdef Py_ssize_t i, j, n = S.shape[1], m = S.shape[0] // 2
    cdef double *d

    d = &S[m,0]
    for i in range(n):
        d[i] += W[i]

def add_D2_W2(double[:,::1] S, double[::1] W, double tau):
    cdef Py_ssize_t i, j, n = S.shape[1]
    cdef double *d

    d = &S[2,0]
    d[0] +=   tau * W[0]
    d[1] +=   tau * (4*W[0] + W[1])
    d[n-2] += tau * (4*W[n-3] + W[n-4])
    d[n-1] += tau * W[n-3]
    for i in range(2, n-2):
        d[i] += tau * (W[i-2] + 4*W[i-1] + W[i])

    d = &S[1,0]
    d[1] += tau * -2*W[0]
    d[n-1] += tau * -2*W[n-3]
    for i in range(2,n-1):
        d[i] += tau * -2*(W[i-2] + W[i-1])

    d = &S[0,0]
    for i in range(2,n):
        d[i] += tau * W[i-2]

    d = &S[3,0]
    d[0] += tau * -2*W[0]
    d[n-2] += tau * -2*W[n-3]
    for i in range(1, n-2):
        d[i] += tau * -2*(W[i-1] + W[i])

    d = &S[4,0]
    for i in range(n-2):
        d[i] += tau * W[i]

def add_D3_W3(double[:,::1] S, double[::1] W, double tau):
    cdef Py_ssize_t i, j, n = S.shape[1]
    cdef double *d

    d = &S[0,0]
    for i in range(3,n):
        d[i] += tau * -W[i-3]

    d = &S[1,0]
    d[0] += tau * 3*W[0]
    d[n-1] += tau * 3*W[n-3]
    for i in range(1,n-1):
        d[i] += tau * 3*(W[i] + W[i-1])

    d = &S[2,0]
    d[0] +=   tau * W[0]
    d[1] +=   tau * (4*W[0] + W[1])
    d[n-2] += tau * (4*W[n-3] + W[n-4])
    d[n-1] += tau * W[n-3]
    for i in range(2, n-2):
        d[i] += tau * (W[i-2] + 4*W[i-1] + W[i])

    d = &S[3,0]
    d[0] += tau * -2*W[0]
    d[n-2] += tau * -2*W[n-3]
    for i in range(1, n-2):
        d[i] += tau * -2*(W[i-1] + W[i])

    d = &S[7,0]
    for i in range(0,n-3):
        d[i] += tau * -W[i]

def add_D4_W4(double[:,::1] S, double[::1] W, double tau):
    cdef Py_ssize_t i, j, n = S.shape[1]
    cdef double *d

    d = &S[4,0]
    d[0] +=   tau * W[0]
    d[1] +=  tau * (16*W[0] + W[1])
    d[2] +=  tau * (36*W[0] + 16*W[1] + W[2])
    d[3] +=  tau * (16*W[0] + 36*W[1] + 16*W[2] + W[3])
    d[n-4] +=  tau * (16*W[n-5] + 36*W[n-6] + 16*W[n-7] + W[n-8])
    d[n-3] +=  tau * (36*W[n-5] + 16*W[n-6] + W[n-7])
    d[n-2] +=  tau * (16*W[n-5] + W[n-6])
    d[n-1] += tau * W[n-5]
    for i in range(4, n-4):
        d[i] += tau * (W[i] + 16*W[i-1] + 36*W[i-2] + 16*W[i-3] + W[i-4])

    d = &S[3,0]
    d[1] +=   tau * -4*W[0]
    d[2] +=   tau * (-24*W[0] + -4*W[1])
    d[3] +=   tau * (-24*W[0] + -24*W[1] + -4*W[2])
    d[n-3] +=   tau * (-24*W[n-5] + -24*W[n-6] + -4*W[n-7])
    d[n-2] +=   tau * (-24*W[n-5] + -4*W[n-6])
    d[n-1] += tau * -4*W[n-5]
    for i in range(4, n-3):
        d[i] += tau * (W[i-4] + 16*W[i-3] + 36*W[i-2] + 16*W[i-1] + W[i])

    d = &S[2,0]
    d[2] +=   tau * 6*W[0]
    d[3] +=   tau * (16*W[0] + 6*W[1])
    d[n-2] += tau * (16*W[n-5] + 6*W[n-6])
    d[n-1] += tau * 6*W[n-5]
    for i in range(4, n-2):
        d[i] += tau * (6*W[i-4] + 16*W[i-3] + 6*W[i-2])

    d = &S[1,0]
    d[3] += tau * (-4*W[0])
    d[n-1] += tau * (-4*W[n-5])
    for i in range(4,n-1):
        d[i] += tau * -4 * (W[i-4] + W[i-3])

    d = &S[0,0]
    for i in range(4, n):
        d[i] += tau * W[i-4]
    
    d = &S[5,0]
    d[0] +=   tau * -4*W[0]
    d[1] +=   tau * (-24*W[0] + -4*W[1])
    d[2] +=   tau * (-24*W[0] + -24*W[1] + -4*W[2])
    d[n-4] +=   tau * (-24*W[n-5] + -24*W[n-6] + -4*W[n-7])
    d[n-3] +=   tau * (-24*W[n-5] + -4*W[n-6])
    d[n-2] += tau * -4*W[n-5]
    for i in range(3, n-4):
        d[i] += tau * (-4*W[i] + -24*W[i-1] + -24*W[i-2] + -4*W[i-3])

    d = &S[6,0]
    d[0] +=   tau * 6*W[0]
    d[1] +=   tau * (16*W[0] + 6*W[1])
    d[n-4] += tau * (16*W[n-5] + 6*W[n-6])
    d[n-3] += tau * 6*W[n-5]
    for i in range(2, n-4):
        d[i] += tau * (6*W[i] + 16*W[i-1] + 6*W[i-2])

    d = &S[7,0]
    d[0] += tau * (-4*W[0])
    d[n-4] += tau * (-4*W[n-5])
    for i in range(1,n-4):
        d[i] += tau * -4 * (W[i] + W[i-1])

    d = &S[8,0]
    for i in range(4, n):
        d[i] += tau * W[i-4]


def add_D1_W1(double[:,::1] S, double[::1] W1, double tau1):
    cdef Py_ssize_t i, j, n = S.shape[1]
    cdef double *d

    # d
    d = &S[1,0]
    d[0] +=   tau1 * W1[0]
    d[n-1] += tau1 * W1[n-1]
    for i in range(1, n-1):
        d[i] += tau1 * (W1[i-1] + W1[i])

    # a
    d = &S[2,0]
    for i in range(n-1):
        d[i] -= tau1 * W1[i]

    # c
    d= &S[0,0]
    for i in range(1,n):
        d[i] -= tau1 * W1[i-1]
        
# cdef _penta_solver(double[:,::1] S, double[::1] Y, double[::1] X):
#     cdef Py_ssize_t i, j, n = Y.shape[0]
#     cdef double *y = &Y[0]
#     cdef double *x = &X[0]
    
#     cdef double *e = &S[4,0]
#     cdef double *c = &S[3,0]
#     cdef double *d = &S[2,0]
#     cdef double *a = &S[1,0]
#     cdef double *b = &S[0,0]
    
#     cdef double[:,::1] T = np.zeros((5,n), 'd')
#     cdef double *mu =    &T[0,0]
#     cdef double *alpha = &T[1,0]
#     cdef double *beta =  &T[2,0]
#     cdef double *gamma = &T[3,0]
#     cdef double *zeta  = &T[4,0]

#     cdef double eps = 1.0e-33
    
#     mu[0] = d[0]
#     # if mu[0] == 0:
#     #     mu[0] = eps
#     alpha[0] = a[0] / mu[0]
#     beta[0] = b[0] / mu[0]
#     zeta[0] = y[0] / mu[0]

#     gamma[1] = c[1]
#     mu[1]    = d[1] - alpha[0] * gamma[1]
#     # if mu[1] == 0:
#     #     mu[1] = eps
#     alpha[1] = (a[1] - beta[0] * gamma[1]) / mu[1]
#     beta[1]  = b[1] / mu[1]
#     zeta[1]  = (y[1] - zeta[0] * gamma[1]) / mu[1]

#     for i in range(2, n-2):
#         gamma[i] = c[i] - alpha[i-2] * e[i]
#         mu[i]    = d[i] - beta[i-2] * e[i] - alpha[i-1] * gamma[i]
#         # if (mu[i]) == 0:
#         #     mu[i] = eps
#         alpha[i] = (a[i] - beta[i-1] * gamma[i]) / mu[i]
#         beta[i]  = b[i] / mu[i]
#         zeta[i]  = (y[i] - zeta[i-2] * e[i] - zeta[i-1] * gamma[i]) / mu[i]

#     gamma[n-2] = c[n-2] - alpha[n-4] * e[n-2]
#     mu[n-2]    = d[n-2] - beta[n-4] * e[n-2] - alpha[n-3] * gamma[n-2]
#     # if mu[n-2] == 0:
#     #     mu[n-2] = eps
#     alpha[n-2] = (a[n-2] - beta[n-3] * gamma[n-2]) / mu[n-2]
#     gamma[n-1] = c[n-1] - alpha[n-3] * e[n-1]
#     mu[n-1]    = d[n-1] - beta[n-3] * e[n-1] - alpha[n-2] * gamma[n-1]
#     # if mu[n-1] == 0:
#     #     mu[n-1] = eps
#     zeta[n-2]  = (y[n-2] - zeta[n-3] * e[n-2] - zeta[n-3] * gamma[n-2]) / mu[n-2]
#     zeta[n-1]  = (y[n-1] - zeta[n-2] * e[n-1] - zeta[n-2] * gamma[n-1]) / mu[n-1]

#     x[n-1] = zeta[n-1]
#     x[n-2] = zeta[n-2] - alpha[n-2] * x[n-1]
#     i = n-3
#     while i >= 0:
#         x[i] = zeta[i] - alpha[i] * x[i+1] - beta[i] * x[i+2]
#         i -= 1

def whittaker_smooth_banded_solver(Y, W, W1, W2, tau1, tau2, tau_z=0, d=2, 
                                   _zeros=np.zeros, _full=np.full):
    N = len(Y)
    S = _zeros((2*d+1, N), "d")
    set_diagonal(S, W)
    if W2 is not None and tau2 > 0:
        if d == 1:
            add_D1_W1(S, W2, tau2)
        elif d == 2:
            add_D2_W2(S, W2, tau2)
        elif d == 3:
            #add_D3_W3(S, W2, tau2)
            pass
        elif d == 4:
            add_D4_W4(S, W2, tau2)
        else:
            raise ValueError("2 <= d <= 4")

    if W1 is not None and tau1 > 0:
        add_D1_W1(S, W1, tau1)

    if tau_z > 0:
        add_diagonal(S, _full(N, -tau_z, "d"))

    # print(S)

    Yw = Y * W

    # X = _zeros(N, "d")
    X = scipy.linalg.solve_banded((d,d), S, Yw,
                                  overwrite_ab=False, overwrite_b=False, check_finite=True)
    return X

def whittaker_smooth_banded(Y, W=None, W1=None, W2=None, 
                        tau1=0, tau2=1.0, tau_z=0, d=2, _ones=np.ones):
    N = Y.shape[0]
    if W is None:
        W = _ones(N, "d")
    if W2 is None:
        W2 = _ones(N, "d")
    if W1 is None:
        W1 = _ones(N, "d")

    inventory.add_to_zeroes(W, 1.0e-9)
    inventory.add_to_zeroes(W1, 1.0e-9)
    inventory.add_to_zeroes(W2, 1.0e-9)
    X = whittaker_smooth_banded_solver(Y, W, W1, W2, tau1, tau2, tau_z=tau_z, d=d)
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

        
# cdef _tria_solver(double[:,::1] S, double[::1] Y, double[::1] X):
#     cdef Py_ssize_t i, n = Y.shape[0]
    
#     cdef double *a = &S[0,0]
#     cdef double *b = &S[1,0]
#     cdef double *c = &S[2,0]
    
#     cdef double[:,::1] T = np.zeros((2,n), "d")
#     cdef double *c1 = &T[0,0]
#     cdef double *y1 = &T[1,0]

#     cdef double *x = &X[0]
#     cdef double *y = &Y[0]

#     c1[0] = c[0] / b[0]
#     for i in range(1,n):
#         c1[i] = c[i] / (b[i] - a[i] * c1[i-1])

#     y1[0] = y[0] / b[0]
#     for i in range(1,n):
#         y1[i] = (y[i] - a[i] * y1[i-1]) / (b[i] - a[i] * c1[i-1])

#     x[n-1] = y1[n-1]
#     i = n-2
#     while i >= 0:
#         x[i] = y1[i] - c1[i] * x[i+1]
#         i -= 1

# def tria_solver(Y, tau, W, W2, _zeros=np.zeros):
#     N = len(Y)
#     S = _zeros((3,N), "d")
#     _get_D1T_D1(tau, W, W2, S)
#     X = _zeros(N, "d")
#     _tria_solver(S, Y*W, X)
#     return X

# def whittaker_smooth_tria(Y, tau=1.0, W=None, W2=None, _ones=np.ones):
#     N = Y.shape[0]
#     if W is None:
#         W = _ones(N, "d")
#     if W2 is None:
#         W2 = _ones(N, "d")
#     X = tria_solver(Y, tau, W, W2)
#     return X

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

# cdef class NDiagonalMatrix:
#     cdef Py_ssize_t N, n
#     cdef double[:,::1] data

#     def __init__(self, N, n):
#         self.data = np.zeros((N,2*n+1), "d")
#         self.N = N
#         self.n = n
#     #
#     cdef get(self, Py_ssize i, Py_ssize j):
#         cdef Py_ssize_t N = self.N, n = self.n, d

#         if i < j:
#             d = j - i
#             if d > n:
#                 return 0
#             else:
#                 return self.data[n+d,d]
#         elif j < i:
#             d = i - j
#             if d > n:
#                 return 0
#             else:
#                 return self.data[n-d,d]
        