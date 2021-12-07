# coding: utf-8

# cython: language_level=3
# cython: boundscheck=True
# cython: wraparound=True
# cython: nonecheck=True
# cython: embedsignature=True
# cython: initializedcheck=True
#cython: unraisable_tracebacks=True  

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
# import mlgrad.distance as distance

from mlgrad.kmeans import init_centers2

from cython.parallel cimport parallel, prange
from openmp cimport omp_get_num_procs

cdef int num_procs = omp_get_num_procs()
if num_procs > 4:
    num_procs /= 2
else:
    num_procs = 2

cdef float max_float = PyFloat_GetMax()
    
cdef void arithmetic_mean(float[:, ::1] X, float[::1] loc):
    cdef int i, n = X.shape[1], N = X.shape[0]
    cdef float v

    for i in range(n):
        v = 0
        for k in range(N):
            v += X[k,i]
        loc[i] = v / N

cdef void covariance_matrix(float[:, ::1] X, float[::1] loc, float[:,::1] S):
    cdef int i, j
    cdef int n = X.shape[1], N = X.shape[0]
    cdef float s, loc_i, loc_j
    #
    for i in range(n):
        loc_i = loc[i]
        for j in range(n): 
            loc_j = loc[j]
            s = 0
            for k in range(N):
                s += (X[k,i] - loc_i) * (X[k,j] - loc_j)
            S[i,j] = s / N

def standard_location(X):
    n = X.shape[1]
    loc = np.zeros(n, 'f')
    arithmetic_mean(X, loc)
    return loc

def standard_covariance(X, loc, normalize=False):
    n = X.shape[1]
    S = np.zeros((n,n), 'f')
    covariance_matrix(X, loc, S)
    if normalize:
        scale_matrix(S)
    return S

cdef void _scale_matrix(float[:,::1] S, float to=1.0):
    cdef int n = S.shape[0], n2 = n*n
    cdef float vol
    cdef Py_ssize_t i
    cdef float *ptr

    vol = linalg.det(np.asarray(S)) / to
    vol = pow(vol, 1.0/n)
    vol = 1/vol
    ptr = &S[0,0]
    for i in range(n2):
        ptr[i] *= vol

def scale_matrix(S, to=1.0):
    _scale_matrix(S, to=1.0)

cdef void _calc_distances(float[::1] D, Distance distfunc, float[:,::1] X, float[::1] loc, float logdet):
        cdef Py_ssize_t k, N = X.shape[0]
        cdef Py_ssize_t n = loc.shape[0]

#         for k in prange(N, nogil=True, schedule='static', num_threads=num_procs):
        for k in range(N):
            D[k] = distfunc._evaluate(&X[k,0], &loc[0], n) + logdet
    
# cdef void _update_loc(float[:,::1] X, float[::1] weights, float[::1] loc, float h):
#     cdef Py_ssize_t i, k, n=loc.shape[0], N=X.shape[0]
#     cdef float v

#     for i in range(n):
#         v = 0
#         for k in prange(N, nogil=True, schedule='static', num_threads=num_procs):
#             v += weights[k] * X[k,i]
#         loc[i] = loc[i] * (1-h) + h * v
    
# cdef float sum_array(float[::1] weights):    
#     cdef Py_ssize_t k, N = weights.shape[0]
#     cdef float W

#     W = 0
# #     for k in prange(N, nogil=True, schedule='static', num_threads=num_procs):
#     for k in range(N):
#         W += weights[k]
#     return W

# cdef void meanify_array(float[::1] weights):    
#     cdef Py_ssize_t k, N = weights.shape[0]
#     cdef float W

#     W = 0
# #     for k in prange(N, nogil=True, schedule='static', num_threads=num_procs):
#     for k in range(N):
#         W += weights[k]
#     for k in range(N):
#         weights[k] /= W

cdef class MLSE:
    #
    cpdef _calc_distances(self):
        _calc_distances(self.D, self.distfunc, self.X, self.loc, self.logdet)
        
    def calc_distances(self, float[:,::1] X):
        D = np.zeros(X.shape[0], 'f')
        _calc_distances(D, self.distfunc, X, self.loc, self.logdet)
        return D

    cpdef get_weights(self):
        cdef int k, N = self.X.shape[0]
        cdef float W

        self._calc_distances()

        self.avg.fit(self.D)

        weights = np.zeros(len(self.D), 'f')
        self.avg.gradient(self.D, weights)

        W = 0
        for k in range(N):
            W += weights[k]
        for k in range(N):
            weights[k] /= W
        
        return weights        
        
    cpdef float Q(self):
        cdef float dval, d

        self._calc_distances()
        self.avg.fit(self.D)
        dval = self.avg.u

        return dval
    
    cpdef update_distfunc(self, S):
        cdef float[:,::1] S1

        S1 = np.linalg.pinv(S)
        if self.distfunc is None:
            self.distfunc = MahalanobisDistance(S1)
        else:
            copy_memoryview2(self.distfunc.S, S1)
#             self.distfunc.S = S1
    
cdef class MLocationEstimator(MLSE):

    def __init__(self, Average avg, float[:,::1] S=None, Func reg=None, tol=1.0e-5, n_iter=1000, tau=0.001, h=0.1):
        self.avg = avg
        self.X = None
        self.S = S
        self.D = None
        self.weights = None
        self.reg = reg
        self.tau = tau
        self.loc = None
        self.loc_min = None
        self.distfunc = None
        self.tol = tol
        self.n_iter = n_iter
        self.dvals = None
        self.h = h

#     cpdef float Q(self):
#         cdef float dval

#         self._calc_distances()
# #         _calc_distances(self.D, self.distfunc, self.X, self.loc)
#         self.avg.fit(self.D)

#         dval = self.avg.u

#         return dval
        
    def init(self, float[:,::1] X, float[::1] loc=None):
        cdef int i, j, k, N, n 
        cdef float v

        self.X = X
        N, n = self.X.shape[0], self.X.shape[1]

        if loc is None:
            self.loc = np.zeros(n, 'f')
            arithmetic_mean(X, self.loc)
        else:
            self.loc = loc

        if self.loc_min is None:
            self.loc_min = np.zeros(n, 'f')
        copy_memoryview(self.loc_min, self.loc)

        if self.S is None:
            self.S = np.identity(n, 'f')
        self.logdet = np.log(np.linalg.det(self.S))

        self.update_distfunc(self.S)

        if self.weights is None:
            self.weights = np.full(N, 1./N, 'f')

        if self.D is None:
            self.D  = np.zeros(N, 'f')
        
        self._calc_distances()
        self.avg.fit(self.D)
        self.avg.gradient(self.D, self.weights)

        self.dval = self.dval_min = self.Q()
        self.dval_prev = PyFloat_GetMax()
        if self.dvals is None:
            self.dvals = []
            
        self.hh = 1

    def fit(self, float[:,::1] X, float[::1] loc=None):
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
                copy_memoryview(self.loc_min, self.loc)

            if self.stop_condition():
                break

            self.K += 1
        
        copy_memoryview(self.loc, self.loc_min)

    cpdef fit_step(self):
        cdef Py_ssize_t n = self.X.shape[1], N = self.X.shape[0]
        cdef Py_ssize_t i, k
        cdef float[:,::1] X = self.X
        cdef float[::1] weights = self.weights
        cdef float[::1] loc = self.loc
        cdef float[::1] D = self.D
        cdef float W, tau = self.tau
        cdef Func reg = self.reg
        cdef float v, z
        cdef float h = self.h

        self._calc_distances()
        self.avg.fit(D)
        self.avg.gradient(D, weights)
#         meanify_array(weights)
        
        for i in range(n):
            v = 0
#             for k in prange(N, nogil=True, schedule='static', num_threads=num_procs):
            for k in range(N):
                v += weights[k] * X[k,i]

            if reg is not None:
                z = reg.derivative(loc[i]) * tau
            else:
                z = 0

            loc[i] = (1-h) * loc[i] + h * (v + z)

    cpdef bint stop_condition(self):        
        if fabs(self.dval - self.dval_prev) / (1 + fabs(self.dval_prev)) >= self.tol:
            return 0
        return 1

cdef class MScatterEstimator(MLSE):

    def __init__(self, Average avg, float[::1] loc = None, Func reg=None, tol=1.0e-5, n_iter=1000, tau=0.001, h=0.1, normalize=0):
        self.avg = avg
        self.X = None
        self.S = None
        self.S_min = None
        self.loc = loc
        self.reg = reg
        self.distfunc = None
        self.tol = tol
        self.n_iter = n_iter
        self.tau = tau
        self.dvals = None
        self.D = None
        self.weights = None
        self.h = h
        self.normalize=normalize
    
    def init(self, float[:,::1] X, float[:,::1] S=None):
        cdef int i, j, k, N, n 
        cdef float vol

        self.X = X
        N, n = self.X.shape[0], self.X.shape[1]
        
        if self.loc is None:
            self.loc = np.zeros(n, 'f')
            arithmetic_mean(X, self.loc)        

        if S is None:
            self.S = np.identity(n, 'f')
        else:
            self.S = S 

        if self.normalize:
            _scale_matrix(self.S)

        self.update_distfunc(self.S)
        
        if self.S_min is None:
            self.S_min = np.zeros((n,n), 'f')
        copy_memoryview2(self.S_min, self.S)

        if self.weights is None:
            self.weights = np.full(N, 1./N, 'f')

        if self.D is None:
            self.D  = np.empty(N, 'f')

        self._calc_distances()
        self.avg.fit(self.D)
        self.avg.gradient(self.D, self.weights)
        
        self.dval_min = self.dval = self.Q()
        self.dval_prev = PyFloat_GetMax()
        if self.dvals is None:
            self.dvals = [self.dval]

    def fit(self, float[:,::1] X, float[:,::1] S):
        self.init(X, S)
        self.K = 1
        
        while self.K <= self.n_iter:
            self.fit_step()
            
            self.dval_prev = self.dval
            self.dval = self.Q()
            self.dvals.append(self.dval)
            if self.dval < self.dval_min:
                self.dval_min = self.dval
                copy_memoryview2(self.S_min, self.S)

            if self.stop_condition():
                break

            self.K += 1

        copy_memoryview2(self.S, self.S_min)
        self.update_distfunc(self.S)
            
    cpdef fit_step(self):
        cdef Py_ssize_t i, j, k
        cdef Py_ssize_t n = self.X.shape[1], N = self.X.shape[0]
        cdef float v, s
        cdef float loc_i, loc_j
        cdef float vol
        cdef float[:,::1] X = self.X
        cdef float[:,::1] S = self.S
        cdef float[::1] weights = self.weights
        cdef float[::1] loc = self.loc
        cdef float W
        cdef tau = self.tau
        cdef Func reg = self.reg
        cdef float h = self.h

        self._calc_distances()
        self.avg.fit(self.D)
        self.avg.gradient(self.D, weights)
#         meanify_array(weights)

        for i in range(n):
            loc_i = loc[i]
            for j in range(i, n):
                loc_j = loc[j]
                s = 0
#                 for k in prange(N, nogil=True, schedule='static', num_threads=num_procs):
                for k in range(N):
                    s += weights[k] * (X[k,i] - loc_i) * (X[k,j] - loc_j)

                if reg is not None:
                    v = self.reg.derivative(S[i,j]) * tau 
                else:
                    v = 0

                S[j,i] = S[i,j] = (1-h) * S[i,j] + h * (s + v)

        if self.normalize:
            _scale_matrix(S)

        self.update_distfunc(S)
        self.logdet = log(np.linalg.det(S))

    cpdef bint stop_condition(self):
        if fabs(self.dval - self.dval_prev) / (1 + fabs(self.dval_prev)) >= self.tol:
            return 0
        return 1    

cdef class MLocationScatterEstimator(MLSE):
    
    def __init__(self, Average avg, n_iter=1000, n_step=10, tol=1.0e-5, h=0.1, reg=None, tau=0.001, normalize_S=1):
        self.avg = avg
        self.mlocation = None
        self.mscatter = None
        self.X = None
        self.tol = tol
        self.n_iter = n_iter
        self.n_step = n_step
        self.h = h
        self.reg = reg
        self.tau = tau
        self.normalize_S = normalize_S
        
    def init(self, float[:,::1] X):
        n = X.shape[1]
        N = X.shape[0]
        self.X = X

        self.loc = np.zeros(n, 'f')
        arithmetic_mean(X, self.loc)
        self.loc_min = np.zeros(n, 'f')

        self.S = np.identity(n, 'f')

        self.S_min = np.zeros((n,n), 'f')
        
        self.mlocation = MLocationEstimator(self.avg, self.S, tol=self.tol, n_iter=self.n_iter, h=self.h, reg=self.reg)
        self.mscatter = MScatterEstimator(self.avg, self.loc, tol=self.tol, n_iter=self.n_iter, h=self.h, reg=self.reg, normalize=self.normalize_S)
        
        self.mlocation.init(X, self.loc)
        self.mscatter.init(X, self.S)

        self.distfunc = self.mscatter.distfunc
#         self.mlocation.distfunc = self.mscatter.distfunc

        self.D  = np.zeros(N, 'f')
        
        self.dval_prev = self.dval_min = self.dval = PyFloat_GetMax()
        self.dvals = []

#     def fit_location(self, float[:,::1] X):
#         self.K = 1
#         self.init(X)
#         self.mlocation.fit(X, self.loc)
        
    def fit(self, float[:,::1] X):
        self.K = 1
        self.init(X)
        
        while self.K <= self.n_step:
            self.fit_step()

            self.dval_prev = self.dval
            self.dval = self.Q()
            self.dvals.append(self.dval)
            if self.dval < self.dval_min:
                self.dval_min = self.dval

            if self.stop_condition():
                break

            self.K += 1
            
    cpdef fit_step(self):
        
        self.mlocation.fit(self.X, self.loc)

        self.mscatter.fit(self.X, self.S)

        self.update_distfunc(self.mscatter.S)
        self.mlocation.update_distfunc(self.mscatter.S)

    cpdef bint stop_condition(self):
        if fabs(self.dval - self.dval_prev) / (1 + fabs(self.dval)) >= self.tol:
            return 0
        return 1

# cdef float euclid_distance2(float[::1] x, float[::1] y):
#     cdef int i, n = x.shape[0]
#     cdef float dist, v

#     dist = 0
#     for i in range(n):
#         v = x[i] - y[i]
#         dist += v*v
#     return dist
    
# def init_locations(X, locs):
    
#     N = X.shape[0]
#     n = X.shape[1]
#     n_locs = len(locs)
    
#     k0 = rand(N)
#     indices = np.random.randint(0, N, n_locs, 'i')
    
#     for j in range(n_locs):
#         m = indices[j]
#         copy_memoryview(locs[j], X[m])

# def  init_scatters(float[:,:,::1] scatters):
#     cdef int i, n, n_locs
#     cdef float[:,::1] S

#     n_locs = scatters.shape[0]
#     n = scatters.shape[1]

#     for i in range(n_locs):
#         S = np.identity(n, 'f')
#         copy_memoryview2(scatters[i], S)
    
