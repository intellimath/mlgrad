import numpy as np
# import matplotlib.pyplot as plt
import numpy.linalg as linalg
from math import sqrt
import sys

# import mltools.aggfuncs as aggfuncs

det = linalg.det
inv = linalg.inv

def gauss_density(x, S1, detS1):
    return np.exp(-0.5*(S1 @ x) @ x) / detS1

class GaussianMixture:

    def __init__(self, q, n_iter=500, n_iter_c=100, n_iter_s=22, tol=1.0e-9):
        self.q = q
        self.n_iter = n_iter
        self.n_iter_c = n_iter_c
        self.n_iter_s = n_iter_s
        self.tol = tol
    #
    def find_VW(self, X, c, S1):
        array = np.array
        exp = np.exp

        V = np.array((self.q, len(X)))
        for j in range(q):
            Xc = X - c[j]
            V[j,:] = exp(-0.5*(Xc.T @ S1[j]) @ Xc) * sqrt(det(S1[j])) # q x N

        W = V.mean(axis=1)
        for j in range(q):
            V[j,:] = V[j].sum()

        return V, W
    #
    def find_c(self, X, V):
        c = V @ X.T
        return c
    #
    def find_s(self, X, c, S1, V):
        new_S1 = []
        VD = np.diag(V)
        for j in range(self.q):
            Xc = X - c[j]
            S = (Xc.T @ VD) @ Xc / det(S1[j])
            new_S1.append(inv(S))
        return new_S1
    #
    def eval_qval(self, X, c, S1, W):
        Q = np.empty((q,len(X)))
        for j in range(self.q):
            Xc = X - c[j]
            Q[j,:] =  exp(-0.5*(Xc.T @ S1[j]) @ Xc) * sqrt(det(S1[j]))
        qval = np.log(Q.T @ W).sum()
        return qval
    #
    def fit_c(self, X):
        c = self.c
        S1 = self.S1
        V, W = self.find_VW(X, c, S1)
        qval = qval_min = self.eval_qval(X, c, S1, W)
        c_min = c.copy()
        qvals = [qval]
        for K in range(self.n_iter_c):
            qval_prev = qval

            c = self.find_c(self, X, V)

            V, W = self.find_VW(X, c, S1)
            qval = self.eval_qval(X, c, S1, W)
            qvals = [qval]

            if qval < qval_min:
                c_min = c.copy()
                qval_min = qval

            if abs(qval - qval_prev) / (1.0 + abs(qval)):
                break

        self.c = c_min
    #
    def fit_s(self, X):
        c = self.c
        S1 = self.S1
        V, W = self.find_VW(X, c, S1)
        qval = qval_min = self.eval_qval(X, c, S1, W)
        S1_min = [S.copy() for S in S1]
        qvals = [qval]
        for K in range(self.n_iter_s):
            qval_prev = qval

            S1 = self.find_s(self, X, c, S1, V)

            V, W = self.find_VW(X, c, S1)
            qval = self.eval_qval(X, c, S1, W)
            qvals = [qval]

            if qval < qval_min:
                S1_min = [S.copy() for S in S1]
                qval_min = qval

            if abs(qval - qval_prev) / (1.0 + abs(qval)):
                break

        self.S1 = S1_min
    #    