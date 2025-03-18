import numpy as np
# import matplotlib.pyplot as plt
import numpy.linalg as linalg
from math import sqrt
import sys

# import mltools.aggfuncs as aggfuncs

from sklearn.cluster import kmeans_plusplus
import mlgrad.inventory as inventory
from mlgrad.af import averaging_function

det = linalg.det
einsum = np.einsum
einsum_path = np.einsum_path
nonzero = np.nonzero
arange = np.arange
empty = np.empty
average = np.average

def norm2(x):
    return (x @ x)

def mnorm2(x, S1):
    return (S1 @ x) @ x

class KMeansBase:
    #
    def predict(self, X):
        Y = np.empty(len(X), 'i')
        Is = self.find_clusters(X)
        for j in range(self.q):
            Ij = Is[j]
            Y[Ij] = j
        return Y
    #
    def initial_locations(self, X):
        return kmeans_plusplus(X, self.q)[0]
    #
    def _eval_dists(self, X):
        c = self.c
        DD = np.empty((self.q, len(X)), "d")
        path, _ = einsum_path("ni,ni->n", X, X, optimize='optimal')
        for j in range(self.q):
            Xj = X - c[j]
            einsum("ni,ni->n", Xj, Xj, optimize=path, out=DD[j])
        return DD
    #
    def eval_dists(self, X):
        # c = self.c
        # DD = np.empty((self.q, len(X)), "d")
        # for j in range(self.q):
        #     Z = X - c[j]
        #     einsum("ni,ni->n", Z, Z, optimize=True, out=DD[j])
        DD = self._eval_dists(X)
        D = DD.min(axis=0)
        return D
    #
    def eval_qval(self, X):
        DD = self._eval_dists(X)
        D = DD.min(axis=0)
        return sqrt(D.sum())
    #
    def find_clusters(self, X):
        DD = self._eval_dists(X)

        D = DD.min(axis=0)
        qval = D.sum()
        self.qvals.append(sqrt(qval))
        
        BB = (DD == D)
        II = arange(len(X))
        Is = [ II[BB[j]] for j in range(self.q)]
        # for k, xk in enumerate(X):
        #     U = xk - self.c
        #     # D = (U * U).sum(axis=1)
        #     D = einsum("ni,ni->n", U, U, optimize=True)
        #     dmin = D.min()
        #     qval += dmin
        #     for j, dj in enumerate(D):
        #         if dj == dmin:
        #             Is[j].append(k)
        # Is = [array(Ij) for Ij in Is]
        return Is
    #


class KMeansMahalanobisBase(KMeansBase):
    #
    def _eval_dists(self, X):
        S1 = self.S1
        c = self.c
        DD = np.empty((self.q, len(X)), "d")
        path, _ = einsum_path("ni,ij,nj->n", X, S1[0], X, optimize='optimal')
        for j in range(self.q):
            Xj = X - c[j]
            einsum("ni,ij,nj->n", Xj, S1[j], Xj, optimize=path, out=DD[j])
        return DD
    #
    def set_weights(self, weights):
        self.weights = np.asarray(weights)
    #

class KMeans(KMeansBase):
    #
    def __init__(self, q, tol=1.0e-8, n_iter=1000, verbose=False):
        self.q = q
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose
        self.S1 = None
    #
    def find_locations(self, X, Is):
        mean = np.mean
        c = np.empty((self.q, X.shape[1]), 'd')
        for j in range(self.q):
            Ij = Is[j]
            Xj = X[Ij]
            c[j,:] = Xj.mean(axis=0)
        return c
    #
    def stop_condition(self, c, c_prev):
        dc = np.abs(c - c_prev)
        dc2 = dc * dc
        dmax2 = max(dc2.sum(axis=1))
        dmax = sqrt(dmax2)
        # print(dmax)
        self.qvals.append(dmax)
        if dmax < self.tol:
            # print(c, c_prev)
            return True
        
        return False
    #
    def fit(self, X):
        self.c = self.initial_locations(X)
        self.qvals = []
        self.qvals = []
        for K in range(self.n_iter):
            c_prev = self.c.copy()
            self.Is = self.find_clusters(X)
            self.c = self.find_locations(X, self.Is)
            if self.stop_condition(self.c, c_prev):
                break
        self.K = K + 1

class RKMeans(KMeansBase):
    #
    def __init__(self, q, avrfunc=None, tol=1.0e-8, n_iter_c=100, n_iter=500, verbose=False):
        self.q = q
        self.n_iter = n_iter
        self.n_iter_c = n_iter_c
        self.tol = tol
        if avrfunc is None:
            self.avrfunc = aggfuncs.ArithMean()
        else:
            self.avrfunc = avrfunc
        self.verbose = verbose
        self.S1 = None
    #
    # def dist(self, x):
    #     U = x - self.c
    #     # D = (U * U).sum(axis=1)
    #     D = einsum("ni,ni->n", U, U, optimize=True)
    #     dmin = D.min()
    #     return sqrt(dmin)
    # #
    # def norm2(self, x):
    #     c = self.c
    #     return min((norm2(x - c[j]) for j in range(self.q)))
    #
    def find_locations(self, X, Is, G):
        n = X.shape[1]
        c = np.zeros((self.q, n), 'd')
        for j in range(self.q):
            Ij = Is[j]
            Gj = G[Ij]
            # GG = Gj.sum()
            # cj = X[Ij].T @ Gj
            # cj = sum((G[k] * X[k] for k in Ij), start=zeros(n, 'd'))
            # GG = sum(G[k] for k in Ij)
            c[j,:] = X[Ij].T @ Gj / Gj.sum()
        return c
    #
    def stop_condition(self, qval, qval_prev):
        if abs(qval - qval_prev) / (1 + self.qval_min) < self.tol:
            return True

        # if len(self.qval_mins) > 1 and \
        #     abs(qval - qval_prev) / (1 + self.qval_min) < self.tol
        
        return False
    #
    def eval_qval(self, X):
        ds = self.eval_dists(X)
        dd = self.avrfunc.evaluate(ds)
        return np.sqrt(dd)
    #
    def fit_locations(self, X):
        N = X.shape[0]
        self.c_min = self.c.copy()

        self.ds = self.eval_dists(X)
        dd = self.avrfunc.evaluate(self.ds)
        qval = self.qval_min = np.sqrt(dd)
        self.qval_min_prev = self.qval_min 
        for K in range(self.n_iter_c):
            qval_prev = qval
            self.avrfunc.evaluate(self.ds)
            G = self.avrfunc.gradient(self.ds)
            self.Is = self.find_clusters(X)
            self.c = self.find_locations(X, self.Is, G)
            
            self.ds = self.eval_dists(X)
            dd = self.avrfunc.evaluate(self.ds)
            qval = np.sqrt(dd)
            self.qvals.append(qval)
            
            if qval <= self.qval_min:
                self.qval_min_prev = self.qval_min
                self.qval_min = qval
                self.c_min = self.c.copy()
                
            if self.stop_condition(qval, qval_prev):
                break
            # if self.stop_condition(self.qval_min_prev, self.qval_min):
            #     break

        self.K = K + 1
        self.c = self.c_min        
    #
    def fit(self, X):
        q = self.q
        n = X.shape[1]
        N = X.shape[0]
        self.c = self.c_min = self.initial_locations(X)
        self.qvals = []
        # self.qval_mins = []
        self.qvals2 = []
        qval2 = self.qval_min = self.eval_qval(X)
        # self.qval_min_prev = self.qval_min
        for K in range(self.n_iter):
            qval_prev2 = qval2
            self.fit_locations(X)
            qval2 = self.eval_qval(X)
            self.qvals2.append(qval2)
            if self.stop_condition(qval2, qval_prev2):
                break
        self.K = K + 1


class KMeansMahalanobis(KMeansMahalanobisBase):
    #
    def __init__(self, q, tol=1.0e-8, n_iter_c=100, n_iter_s=22, n_iter=500, verbose=False):
        self.q = q
        self.n_iter = n_iter
        self.n_iter_c = n_iter_c
        self.n_iter_s = n_iter_s
        self.tol = tol
        self.qvals = []
        self.weights = None
        self.verbose = verbose
    #
    def find_locations(self, X, Is):
        mean = np.mean
        c = np.zeros((self.q, X.shape[1]), 'd')
        for j in range(self.q):
            Ij = Is[j]
            Xj = X[Ij]
            # Xj.mean(axis=0, out=c[j])
            if self.weights is not None:
                weights = self.weights[Ij]
                # print(weights)
                # weights /= weights.sum()
                c[j,:] = average(Xj, axis=0, weights=weights)
            else:
                c[j,:] = Xj.mean(axis=0)
        return c
    #
    def find_scatters(self, X, Is):
        outer = np.outer
        zeros = np.zeros
        inv = linalg.inv
        det = linalg.det
        # diag = np.diag
        c = self.c
        n = X.shape[1]

        S1 = []
        for j in range(self.q):
            Ij = Is[j]
            cj = c[j]
            Xj = X[Ij]
            if self.weights is not None:
                weights = np.ascontiguousarray(self.weights[Ij], dtype="d")
                Sj = inventory.covariance_matrix_weighted(Xj, weights, cj)
            else:
                Sj = inventory.covariance_matrix(Xj, cj)
            # Xj_c = X[Ij] - c[j]
            # if self.weights is not None:
            #     Sj = (Xj_c.T @ diag(self.weights[Ij]) @ Xj_c)
            # else:
            #     Sj = (Xj_c.T @ Xj_c)
            Sj /= det(Sj) ** (1.0/n)
            Sj1 = inv(Sj)
            S1.append(Sj1)

        return S1
    #
    def stop_condition(self, qval, qval_prev):
        if abs(qval - qval_prev) / (1 + self.qval_min) < self.tol:
            # print(qval, qval_prev)
            return True
        
        return False
    #
    def fit_locations(self, X):
        self.qval_min = qval = self.eval_qval(X)
        self.c_min = self.c.copy()
        for K in range(self.n_iter_c):
            qval_prev = qval
            Is = self.find_clusters(X)
            self.c = self.find_locations(X, Is)

            qval = self.eval_qval(X)
            self.qvals.append(qval)
            if qval < self.qval_min:
                self.qval_min = qval
                self.c_min = self.c.copy()

            if self.stop_condition(qval, qval_prev):
                break

        self.c = self.c_min
        self.Is = Is
        self.K_c = K+1
    #
    def fit_scatters(self, X):
        n = X.shape[1]
        self.qval_min = qval = self.eval_qval(X)
        self.S1_min = [S1.copy() for S1 in self.S1]
        for K in range(self.n_iter_s):
            qval_prev = qval
            self.Is = self.find_clusters(X)
            self.S1 = self.find_scatters(X, self.Is)

            # s = np.prod([det(S) for S in self.S1])
            # s /= s ** (1.0/n)
            # self.S1 = [S/s for S in self.S1]

            qval = self.eval_qval(X)
            self.qvals.append(qval)
            if qval < self.qval_min:
                self.qval_min = qval
                self.S1_min = [S1.copy() for S1 in self.S1]

            if self.stop_condition(qval, qval_prev):
                break

        self.S1 = [S1.copy() for S1 in self.S1_min]
        self.K_S = K+1
    #
    def fit(self, X):
        n = X.shape[1]
        self.c = self.c_min = self.initial_locations(X)
        self.S1 = self.S1_min = [np.identity(n) for j in range(self.q)]
        self.qvals = []
        self.qvals2 = []
        qval2 = self.qval_min = self.eval_qval(X)
        for K in range(self.n_iter):
            qval_prev2 = qval2
            self.fit_locations(X)
            self.fit_scatters(X)
            # print(self.K_c, self.K_S)
            qval2 = self.eval_qval(X)
            self.qvals2.append(qval2)
            if self.stop_condition(qval2, qval_prev2):
                break

        self.K = K + 1

class RKMeansMahalanobis(KMeansMahalanobis):
    #
    def __init__(self, q, avrfunc=None, tol=1.0e-8, 
                 n_iter_c=100, n_iter_s=22, n_iter=500, verbose=False):
        self.q = q
        self.n_iter = n_iter
        self.n_iter_c = n_iter_c
        self.n_iter_s = n_iter_s
        self.tol = tol
        if avrfunc is None:
            self.avrfunc = aggfuncs.ArithMean()
        else:
            self.avrfunc = avrfunc
        self.verbose = verbose
        self.weights = None
    #
    # def find_locations(self, X, Is, G):
    #     n = X.shape[1]
    #     zeros = np.zeros
    #     c = zeros((self.q, n), 'd')
    #     for j in range(self.q):
    #         Ij = Is[j]
    #         if len(Ij) == 0:
    #             raise RuntimeError(f"Empty cluster {j}")
    #         Gj = G[Ij]
    #         Xj = X[Ij]
    #         # GG = Gj.sum()
    #         # cj = X[Ij].T @ Gj
    #         # cj = sum((G[k] * X[k] for k in Ij), start=zeros(n, 'd'))
    #         # GG = sum(G[k] for k in Ij)
    #         c[j,:] = (Xj.T @ Gj) / Gj.sum()
    #     return c
    # #
    # def find_scatters(self, X, I, G):
    #     inv = linalg.pinv
    #     det = linalg.det
    #     # diag = np.diag
    #     n = X.shape[1]
    #     c = self.c
    #     S1 = []
    #     n1 = 1.0/n
    #     for j in range(self.q):
    #         # Sj = np.zeros((n,n), 'd')
    #         Ij = I[j]
    #         Gj = G[Ij]
    #         # GGj = np.diag(Gj)
    #         Xcj = X[Ij] - c[j]
    #         Sj = einsum("ni,n,nj->ij", Xcj, Gj, Xcj, optimize=True)
    #         Sj /= Gj.sum()
            
    #         # Sj = ((Xcj.T @ GGj) @ Xcj) / Gj.sum()            
            
    #         # cj = self.c[j]
    #         # g = 0
    #         # for k in Ij:
    #         #     v = X[k] - cj
    #         #     Sj += G[k] * np.outer(v, v)
    #         #     g += G[k]
    #         # Sj /= g
    #         Sj /= det(Sj) ** n1
    #         Sj1 = inv(Sj)
    #         S1.append(Sj1)
    #     return S1
    #
    def stop_condition(self, qval, qval_prev):
        if abs(qval - qval_prev) / (1 + self.qval_min) < self.tol:
            return True
        
        return False
    #
    def eval_qval(self, X):
        ds = self.eval_dists(X)
        dd = self.avrfunc.evaluate(ds)
        return np.sqrt(dd)
    #
    def fit_locations(self, X):
        N = X.shape[0]
        self.c_min = self.c.copy()

        self.ds = self.eval_dists(X)
        dd = self.avrfunc.evaluate(self.ds)
        qval = self.qval_min = np.sqrt(dd)
        self.qval_min_prev = self.qval_min 
        for K in range(self.n_iter_c):
            qval_prev = qval
            self.weights = self.avrfunc.gradient(self.ds)
            self.weights /= self.weights.sum()
            self.Is = self.find_clusters(X)
            self.c = self.find_locations(X, self.Is)
            
            self.ds = self.eval_dists(X)
            dd = self.avrfunc.evaluate(self.ds)
            qval = np.sqrt(dd)
            self.qvals.append(qval)
            
            if qval <= self.qval_min:
                self.qval_min_prev = self.qval_min
                self.qval_min = qval
                self.c_min = self.c.copy()
                
            if self.stop_condition(qval, qval_prev):
                break
            # if self.stop_condition(self.qval_min_prev, self.qval_min):
            #     break

        self.K = K + 1
        self.c = self.c_min        
    #
    def fit_scatters(self, X):
        N = X.shape[0]
        self.S1_min = [S1.copy() for S1 in self.S1]

        self.ds = self.eval_dists(X)
        dd = self.avrfunc.evaluate(self.ds)
        qval = self.qval_min = np.sqrt(dd)
        self.qval_min_prev = self.qval_min 
        for K in range(self.n_iter_s):
            qval_prev = qval
            self.weights = self.avrfunc.gradient(self.ds)
            self.weights /= self.weights.sum()
            self.Is = self.find_clusters(X)
            self.S1 = self.find_scatters(X, self.Is)
            
            self.ds = self.eval_dists(X)
            dd = self.avrfunc.evaluate(self.ds)
            qval = np.sqrt(dd)
            self.qvals.append(qval)

            if qval <= self.qval_min:
                self.qval_min_prev = self.qval_min
                self.qval_min = qval
                self.S1_min = [S1.copy() for S1 in self.S1]
            
            if self.stop_condition(qval, qval_prev):
                break
            # if self.stop_condition(self.qval_min_prev, self.qval_min):
            #     break

        self.K = K + 1
        self.S1 = [S1.copy() for S1 in self.S1_min]
    #
    def fit(self, X):
        q = self.q
        n = X.shape[1]
        N = X.shape[0]
        self.c = self.c_min = self.initial_locations(X)
        self.S1 = self.S1_min = [np.identity(n) for j in range(q)]
        self.qvals = []
        self.qvals2 = []
        qval2 = self.qval_min = self.eval_qval(X)
        for K in range(self.n_iter):
            qval_prev2 = qval2
            self.fit_locations(X)
            self.fit_scatters(X)
            qval2 = self.eval_qval(X)
            self.qvals2.append(qval2)
            if self.stop_condition(qval2, qval_prev2):
                break
        self.K = K + 1

