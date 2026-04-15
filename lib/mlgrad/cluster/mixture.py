import numpy as np
# import matplotlib.pyplot as plt
import numpy.linalg as linalg
from math import sqrt
import sys

from sklearn.cluster import kmeans_plusplus

import mlgrad.inventory as inventory
from scipy.special import logsumexp

solve = linalg.solve
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
    def calculate_ds_dets(self, X):
        N = len(X)
        Ds = np_empty((self.q, N))
        dets = np_empty(self.q)
        for j in range(self.q):
            dets[j] = det(self.S[j])
            Ds[j] = inventory.mahalanobis_distance(X, self.S[j], self.c[j])
        return Ds, dets
    #
    def find_VG(self, X):
        N = len(X)
        ds, dets = self.calculate_ds_dets(X)
        Pjk = 0.5*ds - np.log(dets[:,None])
        Pjk_min = np.min(Pjk, axis=0)
        Pjk2 = Pjk - Pjk_min
        Pjk2 = np.exp(-Pjk2)
        Pk2 = self.G @ Pjk2

        v = np.exp(-Pjk_min) * (Pjk2 / Pk2)
        for j in range(self.q):
            vj = v[j]
            v[j,:] = vj / vj.sum()

        # G = V / V.sum()
        # G /= G.sum()

        return v, self.G
    #
    def find_c(self, X):
        c = self.V @ X
        return c
    #
    def find_S(self, X):
        n = X.shape[1]
        E = np.eye(n)

        new_S = []
        for j in range(self.q):
            S1 = inventory.covariance_matrix_weighted(X, self.V[j], self.c[j])
            print(S1)
            S = solve(S1, E)
            S = (S.T + S) / 2
            # S = pinv(S1)
            new_S.append(S)
        return new_S
    #
    def eval_qval(self, X):
        ds, dets = self.calculate_ds_dets(X)
        dets_g = self.G * dets
        ds2 =  -0.5 * ds
        qvals = -logsumexp(ds2, b=dets_g[:,None], axis=0)
        return qvals.sum()
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
        # for c in self.c:
        #     print(c)
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
