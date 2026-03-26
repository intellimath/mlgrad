import numpy as np
# import matplotlib.pyplot as plt
import numpy.linalg as linalg
from math import sqrt
import sys

from sklearn.cluster import kmeans_plusplus

import mlgrad.inventory as inventory

det = linalg.det
inv = linalg.inv
pinv = linalg.pinv
np_log = np.log
np_exp = np.exp
np_array = np.array
np_empty = np.empty

def gauss_density(x, S, detS):
    return np_exp(-0.5*(S @ x) @ x) * detS1

class GaussianMixture:

    def __init__(self, q, n_iter=500, n_iter_c=100, n_iter_s=22, tol=1.0e-9):
        self.q = q
        self.n_iter = n_iter
        self.n_iter_c = n_iter_c
        self.n_iter_s = n_iter_s
        self.tol = tol
        self.G = np.ones(q) / q
    #
    def initial_locations(self, X):
        return kmeans_plusplus(X, self.q)[0]
    #
    def calculate_probs(self, X):
        N = len(X)
        Pjk = np_empty((self.q, N))
        for j in range(self.q):
            det_j = det(self.S[j])
            Y_j = inventory.mahalanobis_distance(X, self.S[j], self.c[j])
            Pjk[j,:] = np_exp(-0.5*Y_j) * sqrt(det_j) # p_jk
        return Pjk
    #
    def find_VG(self, X):
        N = len(X)
        Pjk = self.calculate_probs(X) # p_jk
        Pk = self.G @ Pjk # p_k

        v = Pjk / Pk
        V = v.sum(axis=1)
        for j in range(self.q):
            v[j,:] = v[j] / V[j]

        G = V / V.sum()
        # G /= G.sum()

        return v, G
    #
    def find_c(self, X):
        c = self.V @ X
        return c
    #
    def find_S(self, X):
        new_S = []
        for j in range(self.q):
            S1 = inventory.covariance_matrix_weighted(X, self.V[j], self.c[j])
            S = pinv(S1)
            new_S.append(S)
        return new_S
    #
    def eval_qval(self, X):
        P = self.calculate_probs(X)
        qval = -np.log(self.G @ P).mean()
        return qval
    #
    def eval_qvals(self, X):
        P = self.calculate_probs(X)
        return P.max(axis=0)
    #
    def fit_c(self, X):
        self.V, self.G = self.find_VG(X)
        qval = self.qval_min = self.eval_qval(X)
        self.qval_prevmin = 100*self.qval_min
        c_min = self.c
        for K in range(self.n_iter_c):
            self.c = self.find_c(X)

            self.V, self.G = self.find_VG(X)
            qval = self.eval_qval(X)
            self.qvals.append(qval)

            if qval < self.qval_min:
                self.qval_prevmin = self.qval_min
                self.qval_min = qval
                c_min = self.c

            if abs(qval - self.qval_min) < self.tol * (1 + abs(self.qval_min)):
                break

        self.c = c_min
        # print(K)
    #
    def fit_S(self, X):
        self.V, self.G = self.find_VG(X)
        qval = self.qval_min = self.eval_qval(X)
        self.qval_prevmin = 100*self.qval_min
        S_min = self.S
        for K in range(self.n_iter_s):
            self.S = self.find_S(X)
            # for j in range(self.q): print(self.S[j])

            self.V, self.G = self.find_VG(X)
            qval = self.eval_qval(X)
            self.qvals.append(qval)

            if qval <= self.qval_min:
                self.qval_prevmin = self.qval_min
                self.qval_min = qval
                S_min = self.S

            if abs(qval - self.qval_min) < self.tol * (1 + abs(self.qval_min)):
                break

        self.S = S_min
        # print(K)
    #
    def fit(self, X):
        n = X.shape[1]
        self.c = c_min = self.initial_locations(X)
        print("Initial centers:")
        for c in self.c:
            print(c)
        self.S = S_min = [np.identity(n) for j in range(self.q)]
        self.qvals = []
        self.V, self.G = self.find_VG(X)

        qval = self.qval_min = self.qval_prevmin = self.eval_qval(X)
        for K in range(self.n_iter):
            # print(K)
            self.fit_c(X)
            # print(self.c)
            self.fit_S(X)
            # print(self.S)
            qval = self.eval_qval(X)
            if qval < self.qval_min:
                self.qval_prevmin = self.qval_min
                self.qval_min = qval
                S_min = self.S
                c_min = self.c

            if abs(qval - self.qval_min) < self.tol * (1 + abs(self.qval_min)):
                break

        self.c = c_min
        self.S = S_min
        self.K = K + 1
        print(f"{self.K} iterations")
    #
    def predict(self, X):
        pass