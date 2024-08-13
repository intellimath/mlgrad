# from mlgrad.af import averaging_function
# from mlgrad.funcs import Quantile_Sqrt
# import mlgrad.funcs2 as funcs2
# import mlgrad.funcs as funcs
# import mlgrad.averager as averager

from ._whittaker import WhittakerSmoother

import math
import numpy as np

def whittaker(X, func=None, func2=None, tau=1.0,
              h=0.001, n_iter=200, tol=1.0e-6):
    alg = WhittakerSmoother(func=func, func2=func2, tau=tau,
                            h=h, n_iter=n_iter, tol=tol)
    s = np.max(abs(X)) / 2
    alg.fit(X/s)
    Z = np.array(alg.Z)
    Z *= s
    # print(alg.K, alg.delta_qval)
    return Z, alg.qvals

def whittaker_agg(X, aggfunc, func=None, func2=None, tau=1.0, 
                  h=0.001, n_iter=100, tol=1.0e-6, n_iter2=100):
    alg = WhittakerSmoother(func=func, func2=func2, tau=tau, h=h, n_iter=n_iter, tol=tol)

    alg.fit(X, W=W)
    Z_min = self.Z = Z.copy()

    def weight_func(E, aggfunc=aggfunc):
        U = func2.evaluate_items(E)
        aggfunc.fit(U)
        return aggfunc.gradient(U)

    self.weight_func = weight_func
    
    E = self.Z - X
    W = weight_func(E)
    s = s_min = aggfunc.u

    flag = False
    for K in range(n_iter2):
        alg.fit(X, W)

        E = self.Z - X
        W = weight_func(E)
        s = aggfunc.u

        if abs(s - s_min) / (1+abs(s_min)) < tol:
            flag = True
            
        if s < s_min:
            s_min = s
            Z_min = self.Z.copy()

        if flag:
            break

        W = aggfunc.gradient(U)

    return Z_min

# def whittaker_weight_func(X, weight_func, func=None, func2=None, W=None, h=0.001, 
#                      tau2=1.0, tau1=0, n_iter=1000, tol=1.0e-8, n_iter2=100, 
#                      collect_qvals=False):
#     alg = WhittakerSmoother(func=func, func2=func2, h=h, 
#                             tau2=tau2, n_iter=n_iter)
    
#     alg.fit(X, W=W)
#     Z_min = self.Z = Z.copy()
    
#     U = func2.evaluate_items(self.Z - X)
#     W = weight_func.gradient(U)
#     s = s_min = 

#     flag = False
#     for K in range(n_iter2):
#         alg.fit(X, W=W)
        
#         U = func2.evaluate_items(self.Z - X)
#         s = aggfunc.fit(U)
        
#         if abs(s - s_min) / (1+abs(s_min)) < tol:
#             flag = True
            
#         if s < s_min:
#             s_min = s
#             Z_min = self.Z.copy()
            
#         if flag:
#             break
            
#         W = aggfunc.gradient(U)

#     return Z_min

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
