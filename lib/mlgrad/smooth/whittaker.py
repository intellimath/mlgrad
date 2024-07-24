# from mlgrad.af import averaging_function
# from mlgrad.funcs import Quantile_Sqrt
import mlgrad.funcs2 as funcs2
import mlgrad.funcs as funcs
import mlgrad.averager as averager


import math
import numpy as np

class WhittakerSmoother:
    #
    def __init__(self, func=None, func1=None, func2=None, h=0.001, n_iter=1000, 
                 tol=1.0e-6, tau1=4.0, tau2=10.0, collect_qvals=False):
        if func is None:
            self.func = funcs2.SquareNorm()
        else:
            self.func = func
        if func1 is None: 
            self.func1 = funcs2.SquareDiff1()
        else:
            self.func1 = func1
        if func2 is None: 
            self.func2 = funcs2.SquareDiff2()
        else:
            self.func2 = func2
        self.n_iter = n_iter
        self.tol = tol
        self.h = h
        self.tau2 = tau2
        self.tau1 = tau1
        self.Z = None
        self.collect_qvals = collect_qvals
        self.qvals = None
    #
    def fit(self, X, W=None, W2=None, W1=None):
        h = self.h
        tau1 = self.tau1
        tau2 = self.tau2
        tol = self.tol
        func = self.func
        func1 = self.func1
        func2 = self.func2

        if W is None:
            W = np.ones_like(X)
        else:
            W = np.asarray(W)

        if W2 is None:
            W2 = np.ones_like(X)
        else:
            W2 = np.asarray(W2)

        if W1 is None:
            W1 = np.ones_like(X)
        else:
            W1 = np.asarray(W1)

        avg = averager.ArrayAdaM2()
        avg.init(len(X))
        
        if self.Z is None:
            Z = X.copy()
        else:
            Z = self.Z
        Z_min = Z.copy()

        ZX = Z - X
        qval = func.evaluate_ex(ZX, W) + \
               tau2 * func2.evaluate_ex(Z, W2)
        if tau1 > 0:
               qval += tau1 * func1.evaluate_ex(Z, W1)
    
        qval_min = qval
        qval_min_prev = 10*qval_min

        if self.collect_qvals:
            qvals = [qval]

        for K in range(self.n_iter):
            qval_prev = qval

            grad = func.gradient_ex(ZX, W) + \
                   tau2 * func2.gradient_ex(Z, W2)
            if tau1 > 0:
                   grad += tau1 * func1.gradient_ex(Z, W1)

            avg.update(grad, h)

            Z -= avg.array_average

            ZX = Z - X
            qval = func.evaluate_ex(ZX, W) + \
                   tau2 * func2.evaluate_ex(Z, W2)
            if tau1 > 0:
                   qval += tau1 * func1.evaluate_ex(Z, W1)

            if self.collect_qvals:
                qvals.append(qval)

            if qval < qval_min:
                qval_min_prev = qval_min
                qval_min = qval
                Z_min = Z.copy()

            if abs(qval - qval_prev) / (1.0 + abs(qval_min)) < tol:
                break

            if abs(qval_min - qval_min_prev) / (1.0 + abs(qval_min)) < tol:
                break

        self.Z = Z_min
        self.K = K+1
        if self.collect_qvals:
            self.qvals = qvals

def whittaker(X, func=None, func1=None, func2=None, W=None, h=0.001, 
              tau2=1.0, tau1=0, n_iter=1000, tol=1.0e-8):
    alg = WhittakerSmoother(func=func, func1=func1, func2=func2, h=h, 
                            tau2=tau2, tau1=tau1, n_iter=n_iter)
    s = np.median(abs(X))
    if s > 10:
        alg.fit(X/s, W=W)
        alg.Z *= s
    else:
        alg.fit(X, W=W)
    return alg.Z

def whittaker_agg(X, aggfunc, func=None, func2=None, W=None, h=0.001, 
                     tau2=1.0, tau1=0, n_iter=1000, tol=1.0e-9, n_iter2=100, 
                     collect_qvals=False):
    alg = WhittakerSmoother(func=func, func2=func2, h=h, 
                            tau2=tau2, n_iter=n_iter)
    
    alg.fit(X, W=W)
    Z_min = Z.copy()
    
    U = func2.evaluate_items(self.Z - X)
    s = s_min = aggfunc.fit(U)
    W = aggfunc.gradient(U)

    flag = False
    for K in range(n_iter2):
        alg.fit(X, W=W)
        
        U = func2.evaluate_items(self.Z - X)
        s = aggfunc.fit(U)
        
        if abs(s - s_min) / (1+abs(s_min)) < tol:
            flag = True
            
        if s < s_min:
            s_min = s
            Z_min = self.Z.copy()
            
        if flag:
            break
            
        W = aggfunc.gradient(U)

    return Z_min

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
