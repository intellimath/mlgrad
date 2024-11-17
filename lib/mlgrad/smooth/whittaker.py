from mlgrad.af import averaging_function
import mlgrad.funcs as funcs
import mlgrad.funcs2 as funcs2
import mlgrad.inventory as inventory
import mlgrad.array_transform as array_transform
# import mlgrad.averager as averager

from ._whittaker import whittaker_smooth_penta, whittaker_smooth_tria

import math
import numpy as np
import scipy
# import matplotlib.pyplot as plt

def whittaker_smooth_scipy(y, tau=1.0, W=None, W2=None, d=2):
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

def whittaker_smooth(X, W=None, W1=None, W2=None, tau1=0, tau2=1.0, solver='scipy', d=2, **kwargs):
    if solver == 'fast' and 1 <= d <= 2:
        if W2 is not None:
            return whittaker_smooth_penta(X, W, W1, W2, tau1, tau2)
        elif W1 is not None:
            return whittaker_smooth_tria(X, tau1, W, W1)
        else:
            return whittaker_smooth_penta(X, W, W1, W2, tau1, tau2)
    elif solver == 'scipy':
        return whittaker_smooth_scipy(X, tau2, W, W2, d=d)
    else:
        raise RuntimeError(f"invalid solver '{solver} or d: {d}")
        
def whittaker_smooth_ex(X, 
              aggfunc = averaging_function("AM"), 
              aggfunc2 = averaging_function("AM"), 
              func = funcs.Square(), 
              func2 = funcs.Square(),
              solver = "penta", d=2,
              tau1=0, tau2=4.0, n_iter=100, tol=1.0e-6):

    N = len(X)

    Z = whittaker_smooth(X, tau=tau, solver=solver, d=d)
    Z_min = Z.copy()

    E = (Z - X)

    U = func.evaluate_array(E)
    aggfunc.evaluate(U)
    W = aggfunc.weights(U)

    D2 = array_transform.array_diff2(Z)
    U2 = func2.evaluate_array(D2)
    aggfunc2.evaluate(U2)
    W2 = aggfunc2.weights(U2)
    
    s = s_min = aggfunc.u + tau * aggfunc2.u

    # ring_array = inventory.RingArray(16)
    # ring_array.add(s)

    flag = False
    for K in range(n_iter):
        Z = whittaker_smooth(X, tau=tau, W=W, W2=W2, solver=solver, d=d)

        E = Z - X

        U = func.evaluate_array(E)
        aggfunc.evaluate(U)
        W = aggfunc.weights(U)
    
        D2 = array_transform.array_diff2(Z)
        U2 = func2.evaluate_array(D2)
        aggfunc2.evaluate(U2)
        W2 = aggfunc2.weights(U2)

        s = aggfunc.u + tau * aggfunc2.u
        # ring_array.add(s)

        if abs(s - s_min) / (1+abs(s_min)) < tol:
            flag = True

        # mad_val = ring_array.mad()
        # if mad_val < tol:
        #     flag = True

        if s < s_min:
            s_min = s
            Z_min = Z.copy()

        if flag:
            break

    return Z_min

def whittaker_smooth_weight_func(
            X, weight_func=None, weight_func2=None, 
            tau=1.0, solver='fast', d=2, n_iter=100, tol=1.0e-8):
    
    Z = whittaker_smooth(X, tau=tau, solver=solver, d=d)
    
    E = X - Z

    if d == 2:
        D = array_transform.array_diff2(Z)
    elif d == 1:
        D = array_transform.array_diff1(Z)
        
    N = len(X)
    
    W = np.ones(N, "d")
    W2 = np.ones(N, "d")

    if weight_func is not None:
        W = weight_func(E)
        # W /= W.sum()
    if weight_func2 is not None:
        W2 = weight_func2(E, D)
        # W2 /= W2.sum()

    # qval = (E*E*W).sum()
    # qval += tau * (D*D*W2).sum()
    
    # qval_min = qval
    Z_min = Z.copy()
    # qvals = []

    flag = False
    for K in range(n_iter):
        Z_prev = Z
        # qval_prev = qval

        Z = whittaker_smooth(X, tau=tau, W=W, W2=W2, solver=solver, d=d)

        E = X - Z

        if d == 2:
            D = array_transform.array_diff2(Z)
        elif d == 1:
            D = array_transform.array_diff1(Z)
        
        if weight_func is not None:
            W = weight_func(E)
            # W /= W.sum()
        if weight_func2 is not None:
            W2 = weight_func2(E, D)
            # W2 /= W2.sum()

        # qval = (E*E*W).sum()
        # qval += tau * (D*D*W2).sum()

        # qvals.append(qval)
        dZ = abs(Z - Z_prev)

        if dZ.max() / Z.mean() < tol:
            break
        
        # if abs(qval - qval_prev) / (1 + abs(qval_min)) < tol:
        #     flag = True

        # if qval > qval_min:
        #     Z_min = Z.copy()
        #     qval_min = qval

        # if flag:
        #     break

    Z = Z_min

    return Z, {'qvals':None}

def whittaker_smooth_weight_func2(
            X, func=None, func1=None, func2=None, 
            tau1=1.0, tau2=1.0, solver='fast', d=2, n_iter=100, tol=1.0e-8):
    
    # Z = whittaker_smooth(X, tau=tau, solver=solver, d=d)
    Z = X * 0.999
    
    E = X - Z

    if func2 is not None:
        D2 = array_transform.array_diff2(Z)
    if func1 is not None:
        D1 = array_transform.array_diff1(Z)
        
    N = len(X)

    W1 = None
    W2 = None
    
    W = np.ones(N, "d")
    # W1 = np.ones(N, "d")
    # W2 = np.ones(N, "d")

    if func is not None:
        W = func.derivative_div_array(E)
        # W /= W.sum()
    if func1 is not None:
        W1 = func1.derivative_div_array(D1)
        # W1 /= W1.sum()
    if func2 is not None:
        W2 = func2.derivative_div_array(D2)
        # W2 /= W2.sum()

    qval = func.evaluate_sum(E)
    if func1:
        qval += tau1 * func1.evaluate_sum(D1)
    if func2:
        qval += tau2 * func2.evaluate_sum(D2)
    qval_min = qval

    # Z_min = Z.copy()
    qvals = []

    flag = False
    for K in range(n_iter):
        qval_prev = qval

        Z = whittaker_smooth(X, W=W, W1=W1, W2=W2, tau1=tau1, tau2=tau2, solver=solver, d=d)

        E = X - Z

        if func1 is not None:
            D1 = array_transform.array_diff1(Z)
        if func2 is not None:
            D2 = array_transform.array_diff2(Z)
        
        if func is not None:
            W = func.derivative_div_array(E)
            # W /= W.sum()
        if func1 is not None:
            W1 = func1.derivative_div_array(D1)
            # W1 /= W1.sum()
        if func2 is not None:
            W2 = func2.derivative_div_array(D2)
            # W2 /= W2.sum()

        qval = func.evaluate_sum(E)
        if func1 is not None:
            qval += tau1 * func1.evaluate_sum(D1)
        if func2 is not None:
            qval += tau2 * func2.evaluate_sum(D2)
        qvals.append(qval)
        
        if abs(qval - qval_prev) / (1 + abs(qval_prev)) < tol:
            flag = True

        # if qval > qval_min:
        #     Z_min = Z.copy()
        #     qval_min = qval

        if flag:
            break

    # Z = Z_min

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
