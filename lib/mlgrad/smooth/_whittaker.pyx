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

cdef diff1_matrix(Py_ssize_t N):
    cdef double[:,::1] D
    cdef double *dd
    cdef Py_ssize_t i,j

    D = mat = inventory.zeros_array2(N-1, N)

    dd = &D[0,0]
    for i in range(N-1):
        dd[0] = 1
        dd[1] = -1
        dd += N+1

    return mat

cdef diff2_matrix(Py_ssize_t N):
    cdef double[:,::1] D
    cdef double *dd
    cdef Py_ssize_t i,j

    D = mat = inventory.zeros_array2(N-2, N)

    dd = &D[0,0]
    for i in range(N-2):
        dd[0] = 1
        dd[1] = -2
        dd[2] = 1
        dd += N+1

    return mat

cdef diff3_matrix(Py_ssize_t N):
    cdef double[:,::1] D
    cdef double *dd
    cdef Py_ssize_t i,j

    D = mat = inventory.zeros_array2(N-3, N)

    dd = &D[0,0]
    for i in range(N-3):
        dd[0] = 1
        dd[1] = -3
        dd[2] = 3
        dd[3] = -1
        dd += N+1

    return mat

cdef diff4_matrix(Py_ssize_t N):
    cdef double[:,::1] D
    cdef double *dd
    cdef Py_ssize_t i,j

    D = mat = inventory.zeros_array2(N-4, N)

    dd = &D[0,0]
    for i in range(N-4):
        dd[0] = 1
        dd[1] = -4
        dd[2] = 6
        dd[3] = -4
        dd[4] = 1
        dd += N+1

    return mat

cdef whittaker_diff(Py_ssize_t N, int d):
    if d == 1:
        D = diff1_matrix(N)
    elif d == 2:
        D = diff2_matrix(N)
    elif d == 3:
        D = diff3_matrix(N)
    elif d == 4:
        D = diff4_matrix(N)
    else:
        D = np.diff(np.eye(N), d, axis=0)
    return D

cdef whittaker_matrix(Py_ssize_t N):
    Z = inventory.zeros_array2(N, N)
    return Z

cdef whittaker_matrix_add_diagonal(double[:,::1] Z, double[::1] W):
    cdef Py_ssize_t i, N = Z.shape[0]

    for i in range(N):
        Z[i,i] += W[i]

cdef whittaker_matrix_add_DD(double[:,::1] ZZ, double tau, double[::1] W, int d):
    cdef Py_ssize_t i, j, N = ZZ.shape[0], Nd = N - d
    cdef Py_ssize_t l, mm, nn1, nn2
    cdef double[:,::1] DD, DD2
    cdef double s, w

    D = whittaker_diff(N, d)
    DD = D

    D2 = inventory.zeros_array2(Nd, N)
    DD2 = D2
    for i in range(Nd):
        w = W[i]
        for j in range(d+1):
            DD2[i,i+j] = w * DD[i,i+j]

    for i in range(Nd):
        for j in range(d+1):
            DD[i,i+j] *= tau

    # ZZ = Z
    for i in range(N):
        mm = i + d + 1
        if mm > N:
            mm = N
        for j in range(i, mm):
            nn1 = i - d
            if nn1 < 0:
                nn1 = 0
            nn2 = j + 1
            if nn2 > Nd:
                nn2 = Nd
            s = 0
            for l in range(nn1, nn2):
                s += DD[l,i] * DD2[l,j]
            if i == j:
                ZZ[i,i] += s
            else:
                ZZ[i,j] += s
                ZZ[j,i] += s
# end def

cdef whittaker_Y(double[::1] y, double[::1] W):
    cdef Py_ssize_t i, N = y.shape[0]
    cdef double[::1] YY

    Y = inventory.empty_array(N)

    YY = Y
    for i in range(N):
        YY[i] = W[i] * y[i]

    return Y

def whittaker_matrices(y, W=None, W2=None, d2=2, tau2=1.0, W1=None, tau1=0.0):
    cdef Py_ssize_t N = len(y)

    if W is None:
        W = inventory.filled_array(N, 1.0)
        # W[:] = 1.0

    Y = whittaker_Y(y, W)

    Z = whittaker_matrix(N)
    whittaker_matrix_add_diagonal(Z, W)

    if tau2 > 0:
        if W2 is None:
            W2 = inventory.filled_array(N, 1.0)
            # W2[:] = 1.0
        whittaker_matrix_add_DD(Z, tau2, W2, d2)
    if tau1 != 0:
        if W1 is None:
            W1 = inventory.filled_array(N, tau1)
            # W1[:] = tau1
        whittaker_matrix_add_diagonal(Z, W1)

    return Z, Y

        

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
