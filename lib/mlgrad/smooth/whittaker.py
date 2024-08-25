# from mlgrad.af import averaging_function
# from mlgrad.funcs import Quantile_Sqrt
# import mlgrad.funcs2 as funcs2
# import mlgrad.funcs as funcs
# import mlgrad.averager as averager

from ._whittaker import WhittakerSmoother, whittaker_smooth_penta as whittaker_smooth

import math
import numpy as np
import scipy

def whittaker_smooth_scipy(y, tau=1.0e5, W=None, W2=None, d=2, **kwargs):
    N = len(y)
    D = scipy.sparse.csc_matrix(np.diff(np.eye(N), d))
    if W is None:
        W = np.ones(N, "d")
    W = scipy.sparse.spdiags(W, 0, N, N)
    if W2 is None:
        W2 = np.ones(N, "d")
    W2 = scipy.sparse.spdiags(W2, 0, N, N)
    Z = W + tau * D.dot(D.T.dot(W2))
    z = scipy.sparse.linalg.spsolve(Z, W*y)

    return z


# def whittaker_smooth(X, *, func=None, func2=None, tau=1.0,
              # h=0.1, n_iter=1000, tol=1.0e-6):
    # alg = WhittakerSmoother(func=func, func2=func2, tau=tau,
                            # h=h, n_iter=n_iter, tol=tol)
    # # s = 1 #(abs(X)).max() / 2
    # alg.fit(X)
    # Z = np.array(alg.Z, dtype="d", order="C", copy=True)
    # # Z *= s
    # # print(alg.K, alg.delta_qval)
    # return Z, {'qval': alg.qval, 'qvals':alg.qvals}

def whittaker_agg(X, aggfunc, func=None, func2=None, tau=1.0, 
                  h=0.1, n_iter=100, tol=1.0e-6, n_iter2=1000):

    Z = whittaker_smooth_penta(X, tau=tau, W=W)
    Z_min = Z.copy()

    def weight_func(E, aggfunc=aggfunc):
        U = func2.evaluate_items(E)
        aggfunc.fit(U)
        return aggfunc.gradient(U)

    self.weight_func = weight_func
    
    E = Z - X
    W = weight_func(E)
    s = s_min = aggfunc.u

    flag = False
    for K in range(n_iter2):
        Z = whittaker_smooth_penta(X, tau=tau, W=W)

        E = Z - X
        W = weight_func(E)
        s = aggfunc.u

        if abs(s - s_min) / (1+abs(s_min)) < tol:
            flag = True
            
        if s < s_min:
            s_min = s
            Z_min = self.Z.copy()

        if flag:
            break

    return Z_min

def whittaker_weight_func(X, weight_func=None, weight_func2=None, tau=1.0, n_iter=100, tol=1.0e-4):
    from math import isclose
    
    Z = whittaker_smooth(X, tau=tau)
    
    E = X - Z
    r = max(abs(E))
    qvals = [r]
    
    W = np.ones_like(X)
    W2 = np.ones_like(X)

    if weight_func is not None:
        W = weight_func(E)
    if weight_func2 is not None:
        W2 = weight_func2(E)

    flag = False
    for K in range(n_iter):
        Z_prev = Z
        r_prev = r

        Z = whittaker_smooth(X, tau=tau, W=W, W2=W2)

        r = max(abs(Z - Z_prev))
        qvals.append(r)

        if isclose(r, r_prev, rel_tol=tol):
            break

        E = X - Z
        if weight_func is not None:
            W = weight_func(E)
        if weight_func2 is not None:
            W2 = weight_func2(E)

    return Z, {'qvals':qvals}

# class WhittakerSmoothPartition:
#     #
#     def __init__(self, n_part, func=None, func2=None, h=0.001, n_iter=1000, 
#                  tol=1.0e-9, tau2=1.0, collect_qvals=False):
#         self.n_part = n_part
#         if func is None:
#             self.func = Square()
#         else:
#             self.func = func
#         # if func1 is None: 
#         #     self.func1 = SquareDiff1()
#         # else:
#         #     self.func1 = func1
#         if func2 is None: 
#             self.func2 = SquareDiff2()
#         else:
#             self.func2 = func2
#         self.n_iter = n_iter
#         self.tol = tol
#         self.h = h
#         # self.tau1 = tau1
#         self.tau2 = tau2
#         self.Z = None
#         self.collect_qvals = collect_qvals
#         self.qvals = None
#     #
#     def fit(self, X, weights=None):
#         n_part = self.n_part
#         h = self.h
#         # tau1 = self.tau1
#         tau2 = self.tau2
#         tol = self.tol
#         func = self.func
#         # func1 = self.func1
#         func2 = self.func2
#         N = len(X)

#         # if weights is None:
#         #     W = np.ones_like(X)
#         # else:
#         #     W = np.asarray(weights)
#         W = np.ones_like(X)
        
#         if self.Z is None:
#             Z = np.zeros((n_part, len(X)), "d")
#             # Z = np.random.random((n_part, len(X)))
#             # Z /= Z.sum(axis=0)
#             for i in range(n_part):
#                 Z[i,:] *= X[:] / n_part
#         else:
#             Z = self.Z
#         # print(Z.sum(axis=0) - X)
#         Z_min = Z.copy()

#         ZX = Z.sum(axis=0) - X
#         print(ZX.shape)
#         qval = W @ func.evaluate_array(ZX)
#         qval += tau2 * ((Z @ Z.T).sum() - (Z*Z).sum()) * 0.5
#         qval /= N
#         # for i in range(n_part):
#         #     qval += tau2 * func2.evaluate(Z[i])
    
#         qval_min = qval
#         qval_min_prev = 10*qval_min

#         if self.collect_qvals:
#             qvals = [qval]

#         for K in range(self.n_iter):
#             qval_prev = qval

#             grad = np.zeros((n_part, len(X)), "d")
#             ee = W @ func.derivative_array(ZX - X)
#             for i in range(n_part):
#                 grad[i,:] = ee
#                 # grad += tau2 * func2.gradient(Z[i])
#                 grad[i,:] += tau2 * (Z.sum(axis=0) - Z[i])
#                 grad[i,:] /= N
                
#             Z -= h * grad
#             np.putmask(Z, Z<0, 0)

#             ZX = Z.sum(axis=0) - X
#             qval = W @ func.evaluate_array(ZX)
#             qval += tau2 * ((Z @ Z.T).sum() - (Z*Z).sum()) * 0.5
#             # for i in range(n_part):
#             #     qval += tau2 * func2.evaluate(Z[i])
#             qval /= N

#             if self.collect_qvals:
#                 qvals.append(qval)

#             if qval < qval_min:
#                 qval_min_prev = qval_min
#                 qval_min = qval
#                 Z_min = Z.copy()

#             if abs(qval - qval_prev) / (1.0 + abs(qval_min)) < tol:
#                 break

#             if abs(qval_min - qval_min_prev) / (1.0 + abs(qval_min)) < tol:
#                 break

#         self.Z = Z_min
#         self.K = K+1
#         if self.collect_qvals:
#             self.qvals = qvals
