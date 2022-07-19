# coding: utf-8

# The MIT License (MIT)
#
# Copyright (c) <2015-2019> <Shibzukhov Zaur, szport at gmail dot com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import numpy as np
import numpy.linalg as linalg

import mlgrad.avragg as avragg
import mlgrad.distance as distance

from cython.parallel cimport parallel, prange

cdef void arithmetic_mean(double[:, ::1] X, double[::1] loc):
    cdef int i, n = X.shape[1], N = X.shape[0]
    cdef double v

    for i in range(n):
        v = 0
        for k in range(N):
            v += X[k,i]
        loc[i] = v / N

cdef multiply_matrix(double[:,::1] A, double[:,::1] B, double[:,::1] C):
    cdef Py_ssize_t n = A.shape[0]
    cdef Py_ssize_t i, j, k
    cdef double s
    cdef double *Ai, *Bk
    
    for i in range(n):
        Ai = &A[i,0]
        for j in range(n):
            s = 0
            Bk = &B[k,0]
            for k in range(n):
                s += Ai[k] * Bk[j]
            C[i,j] = s
        
cdef void covariance_matrix(double[:, ::1] X, double[::1] loc, double[:,::1] S):
    cdef Py_ssize_t i, j
    cdef Py_ssize_t n = X.shape[1], N = X.shape[0]
    cdef double s, loc_i, loc_j
    cdef double *Xk

    for i in range(n):
        loc_i = loc[i]
        for j in range(n):
            loc_j = loc[j]
            s = 0
            Xk = &X[k,0]
            for k in range(N):
                s += (Xk[i] - loc_i) * (Xk[j] - loc_j)
            S[i,j] = s / N
            
cdef double max_val = PyFloat_GetMax()

def standard_location(X):
    n = X.shape[1]
    loc = np.zeros(n, 'd')
    arithmetic_mean(X, loc)
    return loc
    
def standard_covariance(X, loc):
    n = X.shape[1]
    S = np.zeros((n,n), 'd')
    covariance_matrix(X, loc, S)
    return S

cdef void _scale_matrix(double[:,::1] S, double to=1.0):
    cdef int n = S.shape[0]
    cdef double vol
    cdef int i, j

    vol = linalg.det(S)
    vol = pow(vol/to, 1.0/n)
    vol = 1/vol
    for i in range(n):
        for j in range(n):
            S[i,j] *= vol

def scale_matrix(S, to=1.0):
    _scale_matrix(S, to=1.0)

cdef class KMeans_MLSE:
    
    cpdef calc_distances(self):
        cdef Py_ssize_t j, k, j_min
        cdef Py_ssize_t n = self.X.shape[0], N = self.X.shape[0]
        cdef int n_cluster = self.n_cluster
        cdef double d_min, d
        cdef double[:,::1] X = self.X
        cdef double[::1] Xk
        cdef double[::1] loc
        cdef double[::1] D
        cdef double[::1] D_min = self.D_min
        cdef int count[::1] = self.count

        for j in range(n_cluster):
            count[j] = 0

        for k in range(N):
            d_min = max_val
            j_min = 0
            Xk = self.X[k]
            for j in range(n_cluster):
                loc = <double[::1]>self.loc[j]
                D[j, k] = d = self.distfunc.evaluate(X[k], loc)
                if d < d_min:
                    d_min = d
                    j_min = j
            Y[k] = j_min
            D_min[k] = d_min
            count[j_min] += 1

#         self.avg.init(self.D)
#         self.avg.fit(self.D)
    
    cpdef double Q(self):
        cdef double dval
        
        self.calc_distances()

        self.avg.init(self.D_min)
        self.avg.fit(self.D_min)

        dval = self.avg.u

        return dval
    
    def update_distfunc(self, S):
        for j in range(self.n_cluster):
            S1 = np.linalg.inv(S[j])
            if self.distfunc is None:
                self.distfunc[j] = distance.MahalanobisDistance(S1)
            else:
                self.distfunc[j].S = S1
    
cdef class KMeans_MLocationEstimator(KMeans_MLSE):

    def __init__(self, int n_cluster, Average avg, double[:,:,::1] S, tol=1.0e-6, h=0.01, n_iter=1000):
        self.avg = avg
        self.n_cluster = n_cluster
        self.X = None
        self.S = S
        self.loc = None
        self.distfunc = None
        self.tol = tol
        self.n_iter = n_iter
        self.h = h
        if n_cluster != S.shape[0]:
            raise ValueError("n_cluster != S.shape[0]")
        
    cpdef double Q(self):
        cdef double dval

        self.calc_distances()
        dval = self.avg.u

        return dval

    def init(self, double[:,::1] X, double[:,::1] loc=None):
        cdef int i, j, k, N, n 
        cdef double v

        self.X = X
        N, n = self.X.shape[0], self.X.shape[1]
        n_cluster = self.n_cluster

        if loc is None:
            self.loc = np.zeros((n_cluster, n), 'd')
            indexes = np.random.randint(0, N-1, n_cluster, 'i')
            for j in range(n_cluster):
                k = indexes[j]
                for i in range(n):
                    self.loc[j,i] = X[k,i]
        else:
            self.loc = loc

        self.loc_min = np.zeros((n_cluster, n), 'd')
        copy_memoryview2(self.loc_min, self.loc)

        if self.distfunc is None:
            self.distfunc = np.array(n_cluster, 'o')
            for j in range(n_cluster):
                self.distfunc[j] = Distance(np.linalg.inv(self.S[j]))

        self.weights = np.full(N, 1./N, 'd')

        self.D  = np.zeros((n_cluster, N), 'd')
        self.D_min  = np.zeros(N, 'd')
        self.count  = np.zeros(n_cluster, 'i')
        self.dval_min = self.dval = self.calc_Q()
        self.dval_prev = PyFloat_GetMax()
        self.dvals = []

    def fit(self, double[:,::1] X, double[::1] loc=None):
        cdef int i, n, N

        self.init(X, loc)
        self.K = 1

        while self.K <= self.n_iter:
            self.fit_step()

            self.dval_prev = self.dval
            self.dval = self.Q()
            self.dvals.append(self.dval)
            if self.dval < self.dval_min:
                self.dval_min = self.dval
                copy_memoryview2(self.loc_min, self.loc)

            if self.stop_condition():
                break

            self.K += 1

        copy_memoryview2(self.loc, self.loc_min)

    def fit_step(self):
        cdef Py_ssize_t n = self.X.shape[1], N = self.X.shape[0]
        cdef Py_ssize_t i, j, k, t
        cdef double v
        cdef double[:,::1] X = self.X
        cdef double[::1] weights
        cdef double[::1] loc
        cdef double W
        cdef int[::1] K_j

        W = 0
        for k in range(N):
            W += weights[k]

        for j in range(self.n_cluster):
            N_j = self.count[j]
            D_j = np.empty(N_j, 'd')
            K_j = np.empty(N_j, 'i')

            t = 0
            for k in range(N):
                if self.Y[k] == j:
                    D_j[t] = self.D[j, k]
                    K_j[t] = k
                    t += 1
                    
            self.avg.init(D_j)
            self.avg.fit(D_j)

            weights = np.empty(N_j, 'd')
            self.avg.gradient(D_j, weights)
            
            loc = self.loc[j]
            for i in range(n):
                v = 0
                for t in range(N_j):
                    k = K_j[t]
                    v += weights[t] * X[k,i]
                loc[i] = loc[i] * (1-self.h) + self.h *  (v / W)

    def stop_condition(self):        
        if fabs(self.dval - self.dval_prev) >= self.tol:
            return 0
        return 1

cdef class KMeans_MScatterEstimator(KMeans_MLSE):

    def __init__(self, Average avg, double[:,::1] loc, tol=1.0e-6, h=0.001, n_iter=1000):
        self.avg = avg
        self.X = None
        self.S = None
        self.loc = loc
        self.distfunc = None
        self.tol = tol
        self.n_iter = n_iter
        self.h = h
    
    def init(self, double[:,::1] X, double[:,::1] S=None):
        cdef int i, j, k, N, n 
        cdef double vol

        self.X = X
        N, n = self.X.shape[0], self.X.shape[1]

        if S is None:
            self.S = np.zeros((self.n_cluster, n, n), 'd')
            covariance_matrix(X, self.loc, self.S)
            self.S = np.linalg.inv(self.S)
        else:
            self.S = S            
#         scale_matrix(self.S)

#         self.V = np.zeros((n,n), 'd')
            
        self.S_min = np.zeros((n,n), 'd')

        self.update_distfunc(self.S)

        self.weights = np.full(N, 1./N, 'd')

        self.D  = np.empty(N, 'd')
        self.dval_prev = self.dval_min = self.dval  = PyFloat_GetMax()
        self.dvals = []

    def fit(self, double[:,::1] X, double[:,::1] S):
        self.init(X, S)
        self.K = 1
        
        while self.K <= self.n_iter:
            self.fit_step()
            
            self.dval_prev = self.dval
            self.dval = self.Q()
            self.dvals.append(self.dval)
            if self.dval < self.dval_min:
                self.dval_min = self.dval
#                 copy_memoryview2(self.S_min, self.S)
            
            if self.stop_condition():
                break

            self.K += 1
        
#         copy_memoryview2(self.S, self.S_min)
#         self.update_distfunc(self.S)

    def fit_step(self):
        cdef Py_ssize_t i, j, k, n = self.X.shape[1], N = self.X.shape[0]
        cdef double v, s
        cdef double loc_i, loc_j
        cdef double vol
        cdef double[:,::1] X = self.X
        cdef double[:,::1] S = self.S
        cdef double[::1] weights = self.weights
        cdef double W
        cdef double *Xk

        self.avg.gradient(self.D, self.weights)

        W = 0
#         for k in prange(N, nogil=True, schedule='static'):
        for k in range(N):
            W += self.weights[k]

        for i in range(n):
            loc_i = self.loc[i]
            for j in range(n):
                loc_j = self.loc[j]
                s = 0
                Xk = &X[k,0]
#                 for k in prange(N, nogil=True, schedule='static'):
                for k in range(N):
                    s += weights[k] * (Xk[i] - loc_i) * (Xk[j] - loc_j)
                S[i,j] =  S[i,j] * (1 - self.h) + self.h * (s / W)
        
#         scale_matrix(S)

        self.update_distfunc(S)

    def stop_condition(self):
        if fabs(self.dval - self.dval_prev) >= self.tol:
            return 0
        return 1    

cdef class KMeans_MLocationScatterEstimator(KMeans_MLSE):
    
    def __init__(self, Average avg, n_iter=100, h=0.01):
        self.avg = avg
        self.mlocation = None
        self.mscatter = None
        self.X = None
        self.tol = 1.0e-6
        self.n_iter = n_iter
        self.h = h

    def init(self, double[:,::1] X):
        n = X.shape[1]
        N = X.shape[0]
        self.X = X

        self.loc = np.zeros(n, 'd')
        arithmetic_mean(X, self.loc)
        self.loc_min = np.zeros(n, 'd')

        self.S = np.identity(n, 'd')
#         self.S = np.zeros((n,n), 'd')
#         covariance_matrix(X, self.loc, self.S)
#         self.scale_S()

        self.S_min = np.zeros((n,n), 'd')
        
        self.mlocation = MLocationEstimator(self.avg, self.S, h=self.h)
        self.mscatter = MScatterEstimator(self.avg, self.loc, h=self.h)

        self.distfunc = self.mscatter.distfunc
        self.mlocation.distfunc = self.mscatter.distfunc

        self.D  = np.zeros(N, 'd')
        
        self.dval_prev = self.dval_min = self.dval = PyFloat_GetMax()
        self.dvals = []

    def fit_location(self, double[:,::1] X):
        self.K = 1
        self.init(X)
        self.mlocation.fit(X, self.loc)
        
    def fit(self, double[:,::1] X):
        self.K = 1
        self.init(X)
        
        while self.K <= self.n_iter:
            self.fit_step()

            self.dval_prev = self.dval
            self.dval = self.Q()
            self.dvals.append(self.dval)
            if self.dval < self.dval_min:
                self.dval_min = self.dval
#                 copy_memoryview(self.loc_min, self.loc)
#                 copy_memoryview2(self.S_min, self.S)

            if self.stop_condition():
                break

            self.K += 1
            
#         copy_memoryview(self.loc, self.loc_min)            
#         copy_memoryview2(self.S, self.S_min)
#         self.update_distfunc(self.S)
            
    def fit_step(self):
        
        self.mlocation.fit(self.X, self.loc)

        self.mscatter.fit(self.X, self.S)

        self.update_distfunc(self.S)
        self.mlocation.update_distfunc(self.S)

    def stop_condition(self):
        if fabs(self.dval - self.dval_prev) >= self.tol:
            return 0
        return 1
