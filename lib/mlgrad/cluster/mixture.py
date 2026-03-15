import numpy as np
# import matplotlib.pyplot as plt
import numpy.linalg as linalg
from math import sqrt
import sys

from sklearn.cluster import kmeans_plusplus

import mltools.inventory as inventory

det = linalg.det
inv = linalg.inv
np_log = np.log
np_exp = np.exp
np_array = np.array
np_empty = np.empty

def gauss_density(x, S1, detS1):
    return np_exp(-0.5*(S1 @ x) @ x) / detS1

class GaussianMixture:

    def __init__(self, q, n_iter=500, n_iter_c=100, n_iter_s=22, tol=1.0e-9):
        self.q = q
        self.n_iter = n_iter
        self.n_iter_c = n_iter_c
        self.n_iter_s = n_iter_s
        self.tol = tol
        self.L = np.ones(q) / q
    #
    def find_VL(self, X, c, S):
        V = np_empty((self.q, len(X))) # q x N
        for j in range(q):
            S_j = S[j]
            Y_j = inventory.mahalanobis_distance(X, S_j, c[j])
            V[j,:] = np_exp(-0.5*Y_j) * sqrt(det(S_j))

        L = L * V.mean(axis=1)
        L /= L.sum()

        for j in range(q):
            V[j,:] /= V[j].sum()

        return V, L
    #
    def find_c(self, X, V):
        c = V @ X
        return c
    #
    def find_S(self, X, c, S, V):
        new_S = []
        for j in range(self.q):
            S1 = inventory.covariance_matrix_weighted(X, V[j], c[j])
            S1 /= det(S[j])
            S = inv(S1)
            new_S.append(S)
        return new_S
    #
    def eval_qval(self, X, c, S, L):
        Q = np_empty((q,len(X)))
        for j in range(self.q):
            Y_j = inventory.mahalanobis_distance(X, S_j, c[j])
            Q[j,:] =  np_exp(-0.5*Y_j) * sqrt(det(S[j]))
        qval = np_log(L @ Q).mean()
        return qval
    #
    def fit_c(self, X):
        c = self.c
        S = self.S
        V, L = self.find_VL(X, c, S)
        qval = qval_min = qval_prevmin = self.eval_qval(X, c, S, L)
        c_min = c
        for K in range(self.n_iter_c):
            c = self.find_c(self, X, V)

            V, L = self.find_VL(X, c, S)
            qval = self.eval_qval(X, c, S, L)
            self.qvals.append(qval)

            if qval < qval_min:
                qval_prevmin = qval_min
                qval_min = qval
                c_min = c

            if abs(qval_min - qval_prevmin) < self.tol * abs(qval_min):
                break

        self.c = c_min
        self.L = L
    #
    def fit_S(self, X):
        c = self.c
        S = self.S
        V, L = self.find_VL(X, c, S)
        qval = qval_min = qval_prevmin = self.eval_qval(X, c, S, L)
        S_min = S
        for K in range(self.n_iter_s):
            S = self.find_s(self, X, c, S, V)

            V, L = self.find_VL(X, c, S)
            qval = self.eval_qval(X, c, S, L)
            self.qvals.append(qval)

            if qval <= qval_min:
                qval_prevmin = qval_min
                qval_min = qval
                S_min = S

            if abs(qval_min - qval_prevmin) < self.tol * abs(qval_min):
                break

        self.S = S_min
        self.L = L
    #
    def fit(self, X):
        n = X.shape[1]
        self.c = self.c_min = self.initial_locations(X)
        self.S = self.S_min = [np.identity(n) for j in range(self.q)]
        self.qvals = []
        qval = qval_min = self.eval_qval(X)
        for K in range(self.n_iter):
            qval_prev2 = qval2
            self.fit_c(X)
            self.fit_S(X)
            qval = self.eval_qval(X, self.S, self.c, self.L)
            if qval < qval_min:
                qval_prevmin = qval_min
                qval_min = qval
                S_min = [SS.copy() for SS in self.S]
                c_min = [cc.copy() for cc in self.c]

            if abs(qval_min - qval_prevmin) < self.tol * abs(qval_min):
                break

        self.K = K + 1
    #
    def predict(self, X):
        pass