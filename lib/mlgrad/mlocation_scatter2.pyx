# coding: utf-8

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: embedsignature=True
# cython: initializedcheck=False
# cython: unraisable_tracebacks=True  

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

# from cython.parallel cimport parallel, prange
# from openmp cimport omp_get_num_procs

# cdef int num_procs = omp_get_num_procs()
# if num_procs > 4:
#     num_procs /= 2
# else:
#     num_procs = 2

cdef double max_float = PyFloat_GetMax()
    
cdef void arithmetic_mean(double[:, ::1] X, double[::1] loc):
    cdef int i, n = X.shape[1], N = X.shape[0]
    cdef double v

    for i in range(n):
        v = 0
        for k in range(N):
            v += X[k,i]
        loc[i] = v / N

cdef void covariance_matrix(double[:, ::1] X, double[::1] loc, double[:,::1] S):
    cdef int i, j
    cdef int n = X.shape[1], N = X.shape[0]
    cdef double s, loc_i, loc_j
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
    loc = np.zeros(n, 'd')
    arithmetic_mean(X, loc)
    return loc

def standard_covariance(X, loc, normalize=False):
    n = X.shape[1]
    S = np.zeros((n,n), 'd')
    covariance_matrix(X, loc, S)
    if normalize:
        scale_matrix(S)
    return S

cdef void _scale_matrix(double[:,::1] S, double to=1.0):
    cdef int n = S.shape[0], n2 = n*n
    cdef double vol
    cdef Py_ssize_t i
    cdef double *ptr

    vol = linalg.det(np.asarray(S)) / to
    vol = pow(vol, 1.0/n)
    vol = 1/vol
    ptr = &S[0,0]
    for i in range(n2):
        ptr[i] *= vol

def scale_matrix(S, to=1.0):
    _scale_matrix(S, to=1.0)

# cdef void _calc_distances(double[::1] D, Distance distfunc, double[:,::1] X, double[::1] loc, double logdet):
#         cdef Py_ssize_t k, N = X.shape[0]
#         cdef Py_ssize_t n = loc.shape[0]

# #         for k in prange(N, nogil=True, schedule='static', num_threads=num_procs):
#         for k in range(N):
#             D[k] = distfunc._evaluate(&X[k,0], &loc[0], n) + logdet
    
# cdef void _update_loc(double[:,::1] X, double[::1] weights, double[::1] loc, double h):
#     cdef Py_ssize_t i, k, n=loc.shape[0], N=X.shape[0]
#     cdef double v

#     for i in range(n):
#         v = 0
#         for k in prange(N, nogil=True, schedule='static', num_threads=num_procs):
#             v += weights[k] * X[k,i]
#         loc[i] = loc[i] * (1-h) + h * v
    
# cdef double sum_array(double[::1] weights):    
#     cdef Py_ssize_t k, N = weights.shape[0]
#     cdef double W

#     W = 0
# #     for k in prange(N, nogil=True, schedule='static', num_threads=num_procs):
#     for k in range(N):
#         W += weights[k]
#     return W

# cdef void meanify_array(double[::1] weights):    
#     cdef Py_ssize_t k, N = weights.shape[0]
#     cdef double W

#     W = 0
# #     for k in prange(N, nogil=True, schedule='static', num_threads=num_procs):
#     for k in range(N):
#         W += weights[k]
#     for k in range(N):
#         weights[k] /= W

def init_locations(X, locs):
    
    N = X.shape[0]
    n = X.shape[1]
    n_locs = len(locs)
    
    k0 = rand(N)
    indices = np.random.randint(0, N, n_locs, 'i')
    
    for j in range(n_locs):
        m = indices[j]
        copy_memoryview(locs[j], X[m])

def  init_scatters(double[:,:,::1] scatters):
    cdef int i, n, n_locs
    cdef double[:,::1] S

    n_locs = scatters.shape[0]
    n = scatters.shape[1]

    for i in range(n_locs):
        S = np.identity(n, 'd')
        copy_memoryview2(scatters[i], S)
    
cdef class MLSE2:
    
    cpdef calc_distances(self):
        cdef double d, d_min
        cdef int j, j_min, k, N = self.X.shape[0], n = self.X.shape[1]
        cdef int[::1] Y = self.Y
        cdef double[::1] D = self.D

        for k in range(N):    
            d_min = max_float
            j_min = 0
            for j in range(self.n_locs):
                d = (<Distance>self.distfuncs[j])._evaluate(&self.X[k,0], &self.locs[j,0], n)
                if not self.normalize:
                    d += self.logdet[j]
                if d < d_min:
                    d_min = d
                    j_min = j
            D[k] = d_min
            Y[k] = j_min
            
        return self.D
        
    cpdef double Q(self):
        cdef double dval
        cdef int k, j, N=self.Y.shape[0]
        cdef double[:,:,::1] scatters
#         cdef int[::1] Ns

        self.calc_distances()    
        self.avg.fit(self.D)
        dval = self.avg.u

        return dval 
    
    def update_distfuncs(self, double[:,:,::1] scatters):
        cdef int i
        cdef DistanceWithScale distfunc
        cdef double[:,::1] S1

#         for i in prange(self.n_locs, nogil=True, schedule='static', num_threads=num_procs):
        for i in range(self.n_locs):
            if self.pinv:
                S1 = np.linalg.pinv(scatters[i])
            else:
                S1 = np.linalg.inv(scatters[i])
            distfunc = self.distfuncs[i]
            if distfunc is None:
                distfunc = MahalanobisDistance(S1)
                self.distfuncs[i] = distfunc
            else:
                copy_memoryview2(distfunc.S, S1)

cdef class MLocationsScattersEstimator(MLSE2):

    def __init__(self, Average avg, n_locs=2, tol=1.0e-6, h=0.1, n_iter=1000, n_step=20, normalize=1, pinv=0):
        self.avg = avg
        self.X = None
        self.scatters = None
        self.n_locs = n_locs

        self.locs = None
        self.locs_min = None
        self.locs_min = None
        self.scatters = None
        self.scatters_min = None
        
        self.Y = None
        self.logdet = None

        self.distfuncs = None
        self.tol = tol
        self.n_iter = n_iter
        self.n_step = n_step
        self.h = h
        self.normalize = normalize
        self.pinv = pinv

    def init(self, double[:,::1] X, warm=False):
        n = X.shape[1]
        N = X.shape[0]
        self.X = X
        self.distfuncs = np.full(self.n_locs, None, object)
        self.D  = np.zeros(N, 'd')
        self.weights = np.full(N, 1./N, 'd')
        self.W = np.zeros(self.n_locs, 'd')        
        self.dval_prev = self.dval_min = self.dval = PyFloat_GetMax()
        self.dval2_prev = self.dval2_min = self.dval2 = PyFloat_GetMax()
        self.dvals = []
        self.dvals2 = []
        
        if self.Y is None:
            self.Y = np.zeros(N, 'i')
        if warm:
            self.init_locations(X, self.locs)
            self.init_scatters(X, self.scatters)
        else:
            self.init_locations(X)
            self.init_scatters(X)
        
    def init_locations(self, double[:,::1] X, double[:,::1] locs=None):
        n = X.shape[1]
        N = X.shape[0]
        n_locs = self.n_locs
        
        if locs is None:
            if self.locs is None:
                self.locs = np.zeros((n_locs, n), 'd')
                init_locations(X, self.locs)
        else:
            self.locs = locs

        if self.locs_min is None:
            self.locs_min = np.zeros((self.n_locs, n), 'd')
        copy_memoryview2(self.locs_min, self.locs)
            
        if self.scatters is None:
            self.scatters = np.zeros((n_locs, n, n), 'd')
            init_scatters(self.scatters)
        
        if not self.normalize:
            if self.logdet is None:
                self.logdet = np.zeros(self.n_locs, 'd')
            for j in range(self.n_locs):
                self.logdet[j] = log(np.linalg.det(self.scatters[j]))

        self.update_distfuncs(self.scatters)
        
        self.calc_distances()
        self.avg.fit(self.D)
        self.avg.gradient(self.D, self.weights)

        self.dval = self.dval_min = self.Q()
        self.dval_prev = max_float
        if self.dvals is None:
            self.dvals = []
        
        
    def init_scatters(self, double[:,::1] X, double[:,:,::1] scatters=None):
        n = X.shape[1]
        if self.locs is None:
            self.locs = np.zeros((self.n_locs, n), 'd')
            init_locations(X, self.locs)
            
        if scatters is None:
            if self.scatters is None:
                self.scatters = np.zeros((self.n_locs, n, n), 'd')
                init_scatters(self.scatters)
        else:
            self.scatters = scatters
        
        if not self.normalize:
            if self.logdet is None:
                self.logdet = np.zeros(self.n_locs, 'd')
            for j in range(self.n_locs):
                self.logdet[j] = log(np.linalg.det(self.scatters[j]))

        if self.scatters_min is None:
            self.scatters_min = np.zeros((self.n_locs,n, n), 'd')
        copy_memoryview3(self.scatters_min, self.scatters)

        self.update_distfuncs(self.scatters)
        
#         self.dval = self.dval_min = self.Q()
#         self.dval_prev = max_float
#         if self.dvals2 is None:
#             self.dvals2 = []
        
        
    def evaluate(self, double[:,::1] X):
        cdef double d, d_min, float_max = PyFloat_GetMax()
        cdef int j, j_min
        cdef int k, N = X.shape[0], n = X.shape[1]
        cdef int[::1] Y = np.zeros(N, 'i')
        cdef DistanceWithScale distfunc
        cdef DistanceWithScale[::1] distfuncs = self.distfuncs
        cdef double[:,::1] locs = self.locs

        for k in range(N):
            d_min = float_max
            j_min = 0
            for j in range(self.n_locs):
                distfunc = distfuncs[j]
                d = distfunc._evaluate(&X[k,0], &locs[j,0], n)
                if not self.normalize:
                    d += self.logdet[j]
                if d < d_min:
                    j_min = j
                    d_min = d
            Y[k] = j_min

        return Y

    def evaluate_dist(self, double[:,::1] X):
        cdef double d, d_min, float_max = PyFloat_GetMax()
        cdef int j
        cdef int k, N = X.shape[0], n = X.shape[1]
        cdef DistanceWithScale distfunc
        cdef DistanceWithScale[::1] distfuncs = self.distfuncs
        cdef double[:,::1] locs = self.locs
        cdef double[::1] D = np.zeros(N, 'd')

        for k in range(N):
            d_min = float_max
            for j in range(self.n_locs):
                distfunc = distfuncs[j]
                d = distfunc._evaluate(&X[k,0], &locs[j,0], n)
                if not self.normalize:
                    d += self.logdet[j]
                if d < d_min:
                    d_min = d
            D[k] = d_min
            
        return D.base

    def fit_locations(self, double[:,::1] X, double[:,::1] locs=None):
        self.init_locations(X, locs)
        self.K = 1

        self.dval = self.dval_min = self.Q()
        self.dval_prev = max_float
        copy_memoryview2(self.locs_min, self.locs)
        
        while self.K <= self.n_iter:
            self.fit_step_locations()

            self.dval_prev = self.dval
            self.dval = self.Q()
            self.dvals.append(self.dval)
            if self.dval < self.dval_min:
                self.dval_min = self.dval
                copy_memoryview2(self.locs_min, self.locs)

            if self.stop_condition():
                break

            self.K += 1
            
        copy_memoryview2(self.locs, self.locs_min)

    def fit_step_locations(self):
        cdef Py_ssize_t n = self.X.shape[1], N = self.X.shape[0]
        cdef Py_ssize_t i, j, k, l
        cdef double v
        cdef double[:,::1] X = self.X
        cdef double[::1] weights = self.weights
        cdef double[:,::1] locs = self.locs
        cdef double[:,:,::1] scatters = self.scatters
        cdef int[::1] Y = self.Y
        cdef double[::1] W = self.W
        cdef double h = self.h, h1 = 1-self.h

        self.calc_distances()        
        self.avg.fit(self.D)
        self.avg.gradient(self.D, self.weights)
        
        fill_memoryview(W, 0)
        for k in range(N):
            l = Y[k]
            W[l] += weights[k]         

        for j in range(self.n_locs):
            for i in range(n):
                locs[j,i] *= h1
            
        for k in range(N):
            l = Y[k]
            v = h * weights[k] / W[l]
            for i in range(n):
                locs[l,i] += v * X[k,i]

    def fit_scatters(self, double[:,::1] X, double[:,:,::1] scatters):
        self.init_scatters(X, scatters)

        self.K = 1
        self.dval = self.dval_min = self.Q()
        self.dval_prev = max_float
        copy_memoryview3(self.scatters_min, self.scatters)

        while self.K <= self.n_iter:
            self.fit_step_scatters()
            
            self.dval_prev = self.dval
            self.dval = self.Q()
            self.dvals.append(self.dval)
            if self.dval < self.dval_min:
                self.dval_min = self.dval
                copy_memoryview3(self.scatters_min, self.scatters)
            
            if self.stop_condition():
                break

            self.K += 1

        copy_memoryview3(self.scatters, self.scatters_min)

    def fit_step_scatters(self):
        cdef Py_ssize_t i, j, k, l
        cdef int N = self.X.shape[0], n = self.X.shape[1]
        cdef double wk, xx
        cdef double[:,::1] X = self.X
        cdef double[:,:,::1] scatters = self.scatters
        cdef double[:,::1] S
        cdef double[:,::1] locs = self.locs
        cdef double *loc, *Xk, *Si
        cdef double[::1] weights = self.weights
        cdef double[::1] W = self.W
        cdef int[::1] Y = self.Y

        cdef double h = self.h, h1 = 1-self.h
        
        cdef double *ss

        self.calc_distances()
        self.avg.fit(self.D)
        self.avg.gradient(self.D, self.weights)

        fill_memoryview(W, 0)
        for k in range(N):
            l = Y[k]
            W[l] += weights[k] 

        ss = &scatters[0,0,0]
        for i in range(self.n_locs*n*n):
            ss[i] *= h1
#         for l in range(self.n_locs):
#             for i in range(n):
#                 for j in range(n):
#                     scatters[l,i,j] *= h1
            
        for k in range(N):
            l = Y[k]
            wk = h * weights[k] / W[l]
            S = scatters[l]
            loc = &locs[l,0]
            Xk = &X[k,0]
            for i in range(n):
                xx = wk * (Xk[i] - loc[i])
                Si = &S[i,0]
                for j in range(n):
                    Si[j] += xx * (Xk[j] - loc[j])
        
        if self.normalize:
            for l in range(self.n_locs):
                S = scatters[l]
                _scale_matrix(S)

        if not self.normalize:
            for j in range(self.n_locs):
                self.logdet[j] = log(np.linalg.det(scatters[j]))

        self.update_distfuncs(scatters)

    def fit(self, double[:,::1] X, only=None, warm=False):
        self.init(X, warm)

        self.dval2 = self.dval2_min = self.Q()
        self.dval2_prev = max_float
        copy_memoryview2(self.locs_min, self.locs)
        copy_memoryview3(self.scatters_min, self.scatters)
        
        
        if only == 'scatters':
            self.fit_scatters(X)
        elif only == 'locations':
            self.fit_locations(X)
        else:
            self.Ks = 1

            while self.Ks <= self.n_step:
                self.fit_locations(self.X, self.locs)
                self.fit_scatters(self.X, self.scatters)

                self.dval2_prev = self.dval2
                self.dval2 = self.Q()
                self.dvals2.append(self.dval2)
                if self.dval2 < self.dval2_min:
                    self.dval2_min = self.dval2
                    copy_memoryview2(self.locs_min, self.locs)
                    copy_memoryview3(self.scatters_min, self.scatters)

                if self.stop_condition2():
                    break

                self.Ks += 1

        copy_memoryview2(self.locs, self.locs_min)
        copy_memoryview3(self.scatters, self.scatters_min)
                
    def stop_condition(self):        
        if fabs(self.dval - self.dval_prev) / (1 + fabs(self.dval)) >= self.tol:
            return 0
        return 1

    def stop_condition2(self):        
        if fabs(self.dval2 - self.dval2_prev) / (1 + fabs(self.dval2)) >= self.tol:
            return 0
        return 1
    
#     def stop_condition(self):        
#         if fabs(self.dval - self.dval_prev) >= self.tol:
#             return 0
#         return 1
    