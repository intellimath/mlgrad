from mlgrad.af import averaging_function
import mlgrad.funcs as funcs
import mlgrad.funcs2 as funcs2
import mlgrad.inventory as inventory
# import mlgrad.array_transform as array_transform
# import mlgrad.averager as averager
from sys import float_info

from mlgrad.smooth._whittaker import whittaker_matrices
from mlgrad.smooth.utils import create_banded

import math
import numpy as np
import scipy
from scipy.special import expit
from scipy.optimize import curve_fit
from scipy.stats import logistic as stats_logistic

# import matplotlib.pyplot as plt

def whittaker_smooth_base(y, W=None, W2=None, tau2=1.0, tau1=0, d=2):
    d2 = d
    Z, Y = whittaker_matrices(y, W=W, W2=W2, d2=d2, tau2=tau2, tau1=tau1)
    # print(Z.shape, Y.shape)
    Zb = create_banded(Z, d)
    z = scipy.linalg.solve_banded((d,d), Zb, Y, overwrite_ab=True, overwrite_b=True, check_finite=True)
    return z

def whittaker_smooth(y, W=None, W2=None, func=None, func2=None, func2_e=None, 
                     tau2=1.0, tau1=0, d=2, use_sigma=False, mode=2):
    if (func is not None) or (func2 is not None) or (func2_e is not None):
        if mode == 2:
            return whittaker_smooth_weight_func2(y,
                                             func=func,
                                             func2=func2,
                                             func2_e=func2_e,
                                             tau2=tau2, tau1=tau1, 
                                             use_sigma=use_sigma,
                                             d=d)[0]
        elif mode == 1:
            return whittaker_smooth_weight_func(y,
                                             func=func,
                                             func2=func2,
                                             func2_e=func2_e,
                                             tau2=tau2, tau1=tau1, d=d)[0]
        else:
            raise TypeError(f"Invalid mode: {mode}")
    else:
        return whittaker_smooth_base(y, W, W2, tau2, tau1=tau1, d=d)

def whittaker_smooth_scipy(y, W=None, W2=None, tau=1.0, d=2):
    N = len(y)
    D = scipy.sparse.csc_matrix(np.diff(np.eye(N), d))
    if W is None:
        W = np.ones(N, "d")
    W = scipy.sparse.spdiags(W, 0, N, N)
    if W2 is None:
        W2 = np.ones(N, "d")
    W2 = scipy.sparse.spdiags(W2, 0, N, N)
    Z = W + tau * D.dot(D.T.dot(W2))
    Y = y * W
    # Z, Y = whittaker_matrices(y, tau, W, W2, d)
    z = scipy.sparse.linalg.spsolve(Z, Y)
    return z

def whittaker_smooth_ex(X,
              aggfunc = averaging_function("AM"),
              aggfunc2 = averaging_function("AM"),
              func = funcs.Square(),
              func2 = funcs.Square(),
              func2_e = None,
              d=2,
              tau2=1.0, n_iter=100, tol=1.0e-5):

    N = len(X)

    dd = d // 2
    if d % 2 == 0:
        dd1 = dd
        dd2 = -dd
    else:
        dd1 = dd
        dd2 = -dd-1    

    Z = whittaker_smooth_base(X, tau2=tau2, d=d)
    Z_min = Z.copy()

    E = (Z - X)

    U = func.evaluate_array(E)
    aggfunc_u = aggfunc.evaluate(U)
    W = aggfunc.weights(U) * func.derivative_div_array(E)
    # if np.any(np.isnan(W)):
    #     raise TypeError("has nan")

    D2 = inventory.diff2(Z)
    U2 = func2.evaluate_array(D2)
    aggfunc2_u = aggfunc2.evaluate(U2)
    W2 = aggfunc2.weights(U2) 

    if func2 is not None and tau2 > 0:
        W2 *= func2.derivative_div_array(D2)
    if func2_e is not None and tau2 > 0:
        W2 *= func2_e(E[dd1:dd2])

    s = s_min = aggfunc_u + tau2 * aggfunc2_u
    s_min_prev = inventory.double_max / 10
    qvals = [s]

    # ring_array = inventory.RingArray(16)
    # ring_array.add(s)

    flag = False
    for K in range(n_iter):
        Z = whittaker_smooth_base(X, tau2=tau2, W=W, W2=W2, d=d)

        E = Z - X

        U = func.evaluate_array(E)
        aggfunc_u = aggfunc.evaluate(U)
        W = aggfunc.weights(U) * func.derivative_div_array(E)
        # if np.any(np.isnan(W)):
        #     raise TypeError(f"has nan: K={K}")

        D2 = inventory.diff2(Z)
        U2 = func2.evaluate_array(D2)
        aggfunc2_u = aggfunc2.evaluate(U2)
        W2 = aggfunc2.weights(U2) * func2.derivative_div_array(D2)
        if func2_e is not None and tau2 > 0:
            W2 *= func2_e(E[dd1:dd2])

        s = aggfunc_u + tau2 * aggfunc2_u
        # ring_array.add(s)
        qvals.append(s)

        # mad_val = ring_array.mad()
        # if mad_val < tol:
        #     flag = True

        if s < s_min:
            s_min_prev = s_min
            s_min = s
            Z_min = Z.copy()

        if abs(s_min_prev - s_min) < tol * abs(s_min):
            flag = True
        # if abs(s_min_prev - s_min) / (1+abs(s_min)) < tol:
        #     flag = True

        if flag:
            break

    return Z_min, {'qvals':qvals, 'K':K+1, 's':s_min}

def func_aspls(residual, asymmetric_coef=2.0):
    neg_residual = residual[residual < 0]
    if neg_residual.size < 2:
        return np.zeros_like(residual)
    # mu = np.median(neg_residual)
    # std = np.median(abs(neg_residual - mu))
    mu = neg_residual.mean()
    std = neg_residual.std()
    if std == 0:
        std = float_info.min

    # shifted_residual = residual - std
    # add a negative sign since expit performs 1/(1+exp(-input))
    weights = expit(-(asymmetric_coef / std) * (residual - std))
    return weights

def func_logistic(residual):
    X = residual
    mu, sigma = stats_logistic.fit(residual)
    weights = stats_logistic.cdf(X, mu, sigma)
    return 1-weights

def func_rstep(e=0.001):
    def _func_rstep_(residual, e=e):
        Y = (residual < 0).astype("d")
        return Y + e
    return _func_rstep_

def whittaker_smooth_weight_func(
            X, func=None, func1=None, func2=None, func2_e=None,
            tau2=1.0, tau1=0.0,
            d=2, n_iter=100, tol=1.0e-4):

    Z = whittaker_smooth_base(X, tau2=tau2, d=d)

    E = X - Z

    W2 = np.ones(N, "d")

    if func is None:
        func = funcs.Square()
    if func2 is None:
        func2 = funcs.Square()

    dd = d // 2
    if d % 2 == 0:
        dd1 = dd
        dd2 = -dd
    else:
        dd1 = dd
        dd2 = -dd-1

    N = len(X)

    W = func.derivative_div_array(E)
    # W /= W.sum()
    qval = func.evaluate_sum(E)
    if func2 is not None and tau2 > 0:
        D2 = array_transform.array_diff2(Z)
        W2 = func2.derivative_div_array(D2)
        qval += tau2 * func2.evaluate_sum(D2)
    if func2_e is not None and tau2 > 0:
        W2 *= func2_e(E[dd1:dd2])

    qval_min = qval
    Z_min = Z.copy()
    qvals = [qval]

    flag = False
    for K in range(n_iter):
        Z_prev = Z.copy()
        qval_prev = qval
        # print("W:", W)
        # print("W2:", W2)

        Z = whittaker_smooth_base(X, W=W, W1=W1, W2=W2,
                             tau1=tau1, tau2=tau2, d=d)

        E = X - Z

        W = func.derivative_div_array(E)
        # W /= W.sum()
        qval = func.evaluate_sum(E)
        if func2 is not None and tau2 > 0:
            D2 = array_transform.array_diff2(Z)
            W2 = func2.derivative_div_array(D2)
            qval += tau2 * func2.evaluate_sum(D2)
        if func2_e is not None and tau2 > 0:
            W2 *= func2_e(E[dd1:dd2])

        qvals.append(qval)

        if abs(qval - qval_prev) / (1 + abs(qval_min)) < tol:
            flag = True

        if qval < qval_min:
            Z_min = Z.copy()
            qval_min = qval

        if flag:
            break

    Z = Z_min
    dq = abs(qval - qval_prev) / (1 + abs(qval_min))

    return Z, {'qvals':qvals, 'K':K+1, 'tol':dq}

def whittaker_smooth_weight_func2(
            X, func=None, func2=None, func2_e=None, windows=None,
            tau2=1.0, tau1=0, use_sigma=False,
            d=2, n_iter=200, tol=1.0e-6):

    N = len(X)
    sigma = sigma2 = 1.0

    if d == 2:
        diff = inventory.diff2
    elif d == 1:
        diff = inventory.diff1
    elif d == 3:
        diff = inventory.diff3
    elif d == 4:
        diff = inventory.diff4
    else:
        diff_matrix = np.diff(np.eye(N), d, axis=0)
        diff = lambda X: diff_matrix @ X

    dd = d // 2
    if d % 2 == 0:
        dd1 = dd
        dd2 = -dd
    else:
        dd1 = dd
        dd2 = -dd-1

    Z = whittaker_smooth_base(X, tau2=tau2, tau1=tau1, d=d)
    # Z = X * 0.999

    E = X - Z

    if func is not None:
        if use_sigma:
            W = func(E / sigma)
        else:
            W = func(E)
    else:
        W  = np.ones(N, "d")

    if use_sigma:
        sigma2 = (W @ (E*E)) / N
        sigma = math.sqrt(sigma2)

    if func2 is not None and tau2 > 0:
        D2 = diff(Z)
        W2 = func2(D2)
    else:
        W2 = np.ones(N-d, "d")

    if func2_e is not None and tau2 > 0:
        W2 *= func2_e(E[dd1:dd2])

    # if windows:
    #     for ww in windows:
    #         i0, i1 = ww
    #         W2[i0:i1+1] = w_tau2 / tau2

    qvals = []

    flag = False
    for K in range(n_iter):
        Z_prev = Z.copy()

        Z = whittaker_smooth_base(X, W=W, W2=W2, tau2=tau2, tau1=tau1, d=d)

        dz = abs(Z - Z_prev).mean()
        zz = abs(Z_prev).mean()
        dq = dz / (1+zz)
        qvals.append(dq)

        # if abs(Z).mean() == 0:
        #     Z = Z_prev
        #     break

        if dq < tol:
            flag = True

        if flag:
            break

        E = X - Z

        if func is not None:
            if use_sigma:
                W = func(E / sigma)
            else:
                W = func(E)
        else:
            W  = np.ones(N, "d")
    
        if use_sigma:
            sigma2 = (W @ (E*E)) / N
            sigma = math.sqrt(sigma2)

        if func2 is not None and tau2 > 0:
            D2 = diff(Z)
            W2 = func2(D2)
        else:
            W2 = np.ones(N-d, "d")

        if func2_e is not None and tau2 > 0:
            W2 *= func2_e(E[dd1:dd2])

    return Z, {'qvals':qvals, 'K':K+1, 'tol':dq}

# def whittaker_smooth_weight_func3(
#             X, func=None, func2=None, func2_e=None, windows=None, 
#             tau2=1.0, w_tau2 = 1.0, 
#             d=2, n_iter=100, tol=1.0e-9):

#     Z = whittaker_smooth(X, tau2=tau2, tau1=tau1, d=d)
#     # Z = X * 0.999

#     E = X - Z

#     if func2 is not None or tau2 > 0:
#         D2 = inventory.diff2(Z)
#     # if func1 is not None or tau1 > 0:
#     #     D1 = inventory.diff1(Z)

#     N = len(X)

#     W  = np.ones(N, "d")
#     # W1 = np.ones(N, "d")
#     W2 = np.ones(N, "d")

#     if func is not None:
#         W = func(E)
#         # W /= W.mean()
#     # if func1 is not None and tau1 > 0:
#     #     W1 = func1(D1)
#     #     # W1 /= W1.sum()
#     if func2 is not None and tau2 > 0:
#         W2 = func2(D2)
#         # W2 /= W2.mean()
#     if func2_e is not None and tau2 > 0:
#         W2 *= func2_e(E)

#     tau2 = ((E * Z) @ W) / ((D2 * D2) @ W2)

#     # if windows:
#     #     for ww in windows:
#     #         i0, i1 = ww
#     #         W2[i0:i1+1] = w_tau2 / tau2

#     # qval = (E*E*W).sum()
#     # if tau2 > 0:
#     #     qval += tau2 * (D2*D2*W2).sum()
#     # if tau1 > 0:
#     #     qval += tau1 * (D1*D1*W1).sum()

#     qvals = []
#     # qval = abs(Z).sum()

#     flag = False
#     for K in range(n_iter):
#         # qval_prev = qval
#         Z_prev = Z.copy()

#         Z = whittaker_smooth(X, W=W, W2=W2, 
#                              tau2=tau2, d=d)

#         dq = abs(Z - Z_prev).mean() / abs(Z).mean()
#         if dq < tol:
#             flag = True
        
#         E = X - Z

#         # if func2 is not None or tau2 > 0:
#         D2 = inventory.diff2(Z)
#         # if func1 is not None or tau1 > 0:
#         #     D1 = inventory.diff1(Z)
        
#         if func is not None:
#             W = func(E)
#             # W /= W.sum()
#         # if func1 is not None and tau1 > 0:
#         #     W1 = func1(D1)
#         #     # W1 /= W1.sum()
#         if func2 is not None and tau2 > 0:
#             W2 = func2(D2)
#             # W2 /= W2.mean()
#         if func2_e is not None and tau2 > 0:
#             W2 *= func2_e(E)

#         tau2 = ((E * W) @ Z) / ((D2 * D2 * W2) @ Z)
        
#         # if windows:
#         #     for ww in self.windows:
#         #         i0, i1 = ww
#         #         W2[i0:i1+1] = w_tau2 / tau2

#         # qval = (E*E*W).sum()
#         # if tau2 > 0:
#         #     qval += tau2 * (D2*D2*W2).sum()
#         # if tau1 > 0:
#         #     qval += tau1 * (D1*D1*W1).sum()

#         # qvals.append(qval)

#         qvals.append(dq)
        
#         if dq < tol:
#             flag = True

#         if flag:
#             break

#         # if qval > qval_min:
#         #     Z_min = Z.copy()
#         #     qval_min = qval

#     # print("K:", K+1)

#     return Z, {'qvals':qvals, 'K':K+1}

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

def whittaker_smooth_optimize_tau2(X, W=None, W2=None, d=2, tol=1.0e-6):
    N = len(X)
    if W is None:
        W = np.ones(N, "d")
    if W2 is None:
        W2 = np.ones(N-2, "d")

    D2 = inventory.diff2(X)
    S_left = W2 @ (D2 * D2)
    tau2_left = 0

    tau2 = 1.0
    while 1:
        Z = whittaker_smooth(X, W=W, W2=W2, tau2=tau2, d=d)
        E = Z - X
        D2 = inventory.diff2(Z)
        S = W @ (E * E) + tau2 * (W2 @ (D2 * D2))
        if S < S_left:
            tau2 *= 2
        else:
            tau2_right = tau2
            S_right = S
            break

    print(tau2_left, tau2_right)

    while abs(S_left - S_right) / (S_left + S_right) > tol:
        tau2 = (tau2_left + tau2_right) / 2
        Z = whittaker_smooth(X, W=W, W2=W2, tau2=tau2, d=d)
        E = Z - X
        D2 = inventory.diff2(Z)
        S = W @ (E * E) + tau2 * (W2 @ (D2 * D2))
        if S < S_left and S < S_right:
            if S_left < S_right:
                tau2_right = tau2
                S_right = S
            elif S_left > S_right:
                tau2_left = tau2
                S_left = S
        elif S >= S_left:
            S_right = S
            tau2_right = tau2
        elif S >= S_right:
            S_left = S
            tau2_left = tau2

        print(tau2_left, tau2_right)

    tau2 = (tau2_left + tau2_right) / 2
    Z = whittaker_smooth(X, W=W, W2=W2, tau2=tau2, d=d)
    return Z, tau2
