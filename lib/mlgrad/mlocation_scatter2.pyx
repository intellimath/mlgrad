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
# Copyright (c) <2015-2020> <Shibzukhov Zaur, szport at gmail dot com>
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

cdef double max_float = PyFloat_GetMax()
    
cdef void arithmetic_mean(double[:, ::1] X, double[::1] loc):
    cdef Py_ssize_t i, n = X.shape[1], N = X.shape[0]
    cdef double v

    for i in range(n):
        v = 0
        for k in range(N):
            v += X[k,i]
        loc[i] = v / N

cdef void covariance_matrix(double[:, ::1] X, double[::1] loc, double[:,::1] S):
    cdef Py_ssize_t i, j
    cdef Py_ssize_t n = X.shape[1], N = X.shape[0]
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

cdef det_matrix_2(double[:,::1] S):
    return S[0,0]*S[1,1] - S[1,0]*S[0,1]

cdef double[:,::1] inv_matrix_2(double[:,::1] S):
    cdef double s00 = S[0,0], s01 = S[0,1], s10 = S[1,0], s11 = S[1,1]
    cdef double d = s00*s11 - s10*s01
    
    S[0,0] = s11 / d
    S[1,1] = s00 / d
    S[1,0] = -s10 / d
    S[0,1] = -s01 / d
    return S

cdef void _scale_matrix(double[:,::1] S, double to=1.0):
    cdef Py_ssize_t n = S.shape[0], n2 = n*n
    cdef double vol
    cdef Py_ssize_t i
    cdef double *ptr

    if n == 2:
        vol = det_matrix_2(S) / to
    else:
        vol = linalg.det(np.asarray(S)) / to
    if vol <= 0:
        print('-', S)
    vol = pow(vol, 1.0/n)
    vol = 1/vol
    ptr = &S[0,0]
    for i in range(n2):
        ptr[i] *= vol

def scale_matrix(S, to=1.0):
    _scale_matrix(S, to=1.0)

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
    cdef Py_ssize_t i, n, n_locs
    cdef double[:,::1] S

    n_locs = scatters.shape[0]
    n = scatters.shape[1]

    for i in range(n_locs):
        S = np.identity(n, 'd')
        copy_memoryview2(scatters[i], S)
    
cdef class MLSE2:
    
    cpdef calc_distances(self):
        cdef Py_ssize_t j, k, N = self.X.shape[0], n = self.X.shape[1]
        cdef double[:,::1] DD = self.DD
        cdef double[::1] DD_k
        cdef double[::1] D = self.D
        cdef double[:,::1] GG = self.GG
        cdef double[:,::1] locs = self.locs
        cdef double[:,::1] X = self.X
        cdef DistanceWithScale[::1] distfuncs = self.distfuncs
        cdef int num

        for k in range(N):
            DD_k = DD[k]
            for j in range(self.n_locs):
                DD_k[j] = (<DistanceWithScale>distfuncs[j])._evaluate(&X[k,0], &locs[j,0], n)

            self.avg_min.fit(DD_k)
            D[k] = self.avg_min.u
            self.avg_min.gradient(DD_k, GG[k])

        return D
            
    cpdef calc_weights(self):

        self.avg.fit(self.D)
        self.avg.gradient(self.D, self.weights)

    cpdef calc_update_GG(self):
        cdef Py_ssize_t j, k, N = self.X.shape[0], n = self.X.shape[1]
        cdef double wk
        cdef double[:,::1] GG = self.GG
        cdef double *GG_k
        cdef double *W = &self.W[0]
        cdef double *weights = &self.weights[0]

#         for k in prange(N, nogil=True, num_threads=num_procs):
        for k in range(N):
            wk = weights[k]
            GG_k = &GG[k, 0]
            for j in range(self.n_locs):
                GG_k[j] *= wk
                
        fill_memoryview(self.W, 0)
#         num = int_min(num_procs, self.n_locs)
#         for j in prange(self.n_locs, nogil=True, num_threads=num):
        for j in range(self.n_locs):
            for k in range(N):
                W[j] += GG[k,j]        
                

#         for k in prange(N, nogil=True, num_threads=num_procs):
        for k in range(N):
            wk = weights[k]
            GG_k = &GG[k, 0]
            for j in range(self.n_locs):
                GG_k[j] /= W[j]

    cpdef double Q(self):
        self.calc_distances()    
        self.avg.fit(self.D)
        return self.avg.u
                
        
    cpdef double local_Q(self):
        cdef Py_ssize_t k, N = self.X.shape[0], n = self.X.shape[1]
        cdef double *weights = &self.weights[0]
        cdef double *D = &self.D[0]
        cdef double s, W
        
        s = 0
        for k in range(N):
            s += weights[k] * D[k]

        return s
    
    def update_distfuncs(self, double[:,:,::1] scatters):
        cdef Py_ssize_t i, j, n
        cdef DistanceWithScale distfunc
        cdef double[:,::1] S

        for i in range(self.n_locs):
            S = scatters[i]
            distfunc = self.distfuncs[i]
            if distfunc is None:
                distfunc = MahalanobisDistance(S)
                self.distfuncs[i] = distfunc
            else:
                copy_memoryview2(distfunc.S, S)

cdef class MLocationsScattersEstimator(MLSE2):

    def __init__(self, Average avg, Average avg_min, n_locs, tol=1.0e-6, n_iter=100, n_iter_s=0, n_step=100, alpha=0.9):
        self.avg = avg
        self.avg_min = avg_min
        self.X = None
        self.n_locs = n_locs

        self.locs = None
        self.locs_min = None
        self.scatters = None
        self.scatters_min = None
        self.dvals = None

        self.distfuncs = None
        self.tol = tol
        self.n_iter = n_iter
        if n_iter_s <= 0:
            self.n_iter_s = n_iter
        else:
            self.n_iter_s = n_iter_s
        self.n_step = n_step
        self.alpha = alpha

    def init(self, double[:,::1] X, warm=False):
        n = X.shape[1]
        N = X.shape[0]
        self.X = X
        self.distfuncs = np.full(self.n_locs, None, object)
        for i in range(self.n_locs):
            self.distfuncs[i] = <DistanceWithScale>MahalanobisDistance(np.identity(n, 'd'))

        self.D  = np.zeros(N, 'd')
        self.DD  = np.zeros((N, self.n_locs), 'd')
        self.GG  = np.zeros((N, self.n_locs), 'd')
        
        self.weights = np.full(N, 1./N, 'd')
        self.W = np.zeros(self.n_locs, 'd')        
        self.dval_prev = self.dval_min = self.dval = PyFloat_GetMax()
        self.dval2_prev = self.dval2_min = self.dval2 = PyFloat_GetMax()
        self.dvals = []
        self.dvals2 = [] 
        
        if warm:
            self.init_locations(X, self.locs)
            self.init_scatters(X, self.scatters)
        else: 
            self.init_locations(X)
            self.init_scatters(X)
            
        self.calc_distances()
        self.calc_weights()        
        self.calc_update_GG()                

        self.dval = self.dval_min = self.local_Q()
        self.dvals.append(self.dval)
        self.dval_prev = max_float
        
        self.dval2 = self.dval2_min = self.Q()
        self.dval2_prev = max_float            
        self.dvals2.append(self.dval2)
        
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
            self.update_distfuncs(self.scatters)        
        
    def init_scatters(self, double[:,::1] X, double[:,:,::1] scatters=None):
        n = X.shape[1]
        if self.locs is None:
            self.locs = np.zeros((self.n_locs, n), 'd')
            init_locations(X, self.locs)
            
        if scatters is None:
            if self.scatters is None:
                self.scatters = np.zeros((self.n_locs,n,n), 'd')
                init_scatters(self.scatters)
        else:
            self.scatters = scatters
            
        if self.scatters_min is None:
            self.scatters_min = np.zeros((self.n_locs,n,n), 'd')
        copy_memoryview3(self.scatters_min, self.scatters)

        self.update_distfuncs(self.scatters)        

    def evaluate(self, double[:,::1] X):
        cdef double d, d_min, float_max = PyFloat_GetMax()
        cdef Py_ssize_t j, j_min
        cdef Py_ssize_t k, N = X.shape[0], n = X.shape[1]
        cdef Py_ssize_t[::1] Y = np.zeros(N, 'l')
        cdef DistanceWithScale distfunc
        cdef DistanceWithScale[::1] distfuncs = self.distfuncs
        cdef double[:,::1] locs = self.locs

        for k in range(N):
            d_min = float_max
            j_min = 0
            for j in range(self.n_locs):
                distfunc = distfuncs[j]
                d = distfunc._evaluate(&X[k,0], &locs[j,0], n)
                if d < d_min:
                    j_min = j
                    d_min = d
            Y[k] = j_min

        return Y

    def evaluate_dist(self, double[:,::1] X):
        cdef double d, d_min, float_max = PyFloat_GetMax()
        cdef Py_ssize_t j
        cdef Py_ssize_t k, N = X.shape[0], n = X.shape[1]
        cdef DistanceWithScale distfunc
        cdef DistanceWithScale[::1] distfuncs = self.distfuncs
        cdef double[:,::1] locs = self.locs
        cdef double[::1] D = np.zeros(N, 'd')

        for k in range(N):
            d_min = float_max
            for j in range(self.n_locs):
                distfunc = distfuncs[j]
                d = distfunc._evaluate(&X[k,0], &locs[j,0], n)
                if d < d_min:
                    d_min = d
            D[k] = d_min
            
        return D.base

    def fit_locations(self, double[:,::1] X, double[:,::1] locs=None):
        self.K = 1

        self.calc_distances()
        self.calc_weights()        
        self.calc_update_GG()                
        
        while self.K <= self.n_iter:
            
            self.fit_step_locations()
            
            self.calc_distances()
            self.calc_update_GG()                    

            self.dval_prev = self.dval
            self.dval = self.local_Q()
            self.dvals.append(self.dval)

            if self.dval < self.dval_min:
                copy_memoryview2(self.locs_min, self.locs)
                if (self.dval_min - self.dval) / (1 + fabs(self.dval_min)) < self.tol:
                    self.dval_min = self.dval
                    break
                self.dval_min = self.dval
            
            if self.stop_condition():
                break

            self.K += 1

    def fit_step_locations(self):
        cdef Py_ssize_t n = self.X.shape[1], N = self.X.shape[0]
        cdef Py_ssize_t i, j, k, l
        cdef double v, wk, wkj, Wj
        cdef double[:,::1] X = self.X
        cdef double[:,::1] locs = self.locs
        cdef double *W = &self.W[0]
        cdef double *locs_j
        cdef double *Xk
        cdef double[:, ::1] GG = self.GG
        cdef double *GG_k
        cdef double alpha = self.alpha
        
        multiply_memoryview2(locs, 1-alpha)
        for k in range(N):
            GG_k = &GG[k, 0]
            Xk = &X[k, 0]
            for j in range(self.n_locs):
                locs_j = &locs[j, 0]
                gkj = alpha * GG_k[j]
                for i in range(n):
                    locs_j[i] += gkj * Xk[i]
                
    def fit_scatters(self, double[:,::1] X, double[:,:,::1] scatters=None):
        self.K = 1
        
        self.calc_distances()
        self.calc_weights()        
        self.calc_update_GG()                

        while self.K <= self.n_iter_s:

            self.fit_step_scatters()
            
            self.calc_distances()
            self.calc_update_GG()                                
            
            self.dval_prev = self.dval
            self.dval = self.local_Q()
            self.dvals.append(self.dval)
            
            if self.dval < self.dval_min:
                copy_memoryview3(self.scatters_min, self.scatters)
                if (self.dval_min - self.dval) / (1 + fabs(self.dval_min)) < self.tol:
                    self.dval_min = self.dval
                    break
                self.dval_min = self.dval

            if self.stop_condition():
                break

            self.K += 1

    def fit_step_scatters(self):
        cdef Py_ssize_t i, j, k, l
        cdef Py_ssize_t N = self.X.shape[0], n = self.X.shape[1]
        cdef double wk, vv, Wl
        cdef double[:,::1] X = self.X
#         cdef DistanceWithScale[::1] distfuncs = self.distfuncs
        cdef double[:,:,::1] scatters = self.scatters
        cdef double[:,::1] S, S1
        cdef double[:,::1] locs = self.locs
        cdef double[::1] weights = self.weights
        cdef double[::1] W = self.W
        cdef double[:, ::1] GG = self.GG
        cdef double alpha = self.alpha

        cdef double *loc
        cdef double *Xk
        cdef double *Si

        multiply_memoryview3(scatters, 1-alpha)
        for l in range(self.n_locs):
            S = scatters[l]
            loc = &locs[l, 0]
            for k in range(N):
                Xk = &X[k, 0]
                wk = GG[k,l]
                for i in range(n):
                    vv = alpha * wk * (Xk[i] - loc[i])
                    Si = &S[i, 0]
                    for j in range(i,n):
                        Si[j] += vv * (Xk[j] - loc[j])
                        if j > i:
                            S[j,i] = Si[j]
            
        for l in range(self.n_locs):
            S = scatters[l]
            if n == 2:
                S = inv_matrix_2(S)
            else:
                S1 = linalg.pinv(np.asarray(S), hermitian=True)
                copy_memoryview2(S, S1)
            _scale_matrix(S)

        self.update_distfuncs(scatters)

    def fit(self, double[:,::1] X, only=None, warm=False):
        self.init(X, warm)

        scatters_only = (only == 'scatters')
        locations_only = (only == 'locations')

        self.Ks = 1

        while self.Ks <= self.n_step:
            if not scatters_only:
                self.fit_locations(self.X, self.locs)

                self.dval2_prev = self.dval2
                self.dval2 = self.Q()
                self.dvals2.append(self.dval2)
                if self.dval2 < self.dval2_min:
                    self.dval2_min = self.dval2
                    copy_memoryview2(self.locs_min, self.locs)

            if not locations_only:
                self.fit_scatters(self.X, self.scatters)

                self.dval2_prev = self.dval2
                self.dval2 = self.Q()
                self.dvals2.append(self.dval2)
                if self.dval2 < self.dval2_min:
                    self.dval2_min = self.dval2
                    copy_memoryview3(self.scatters_min, self.scatters)
                    
#             if locations_only or scatters_only:
#                 break

            if self.stop_condition2():
                break

            self.Ks += 1

        copy_memoryview2(self.locs, self.locs_min)
        copy_memoryview3(self.scatters, self.scatters_min)
        self.update_distfuncs(self.scatters)
        
                
    cdef bint stop_condition(self):        
        if fabs(self.dval - self.dval_prev) / (1 + fabs(self.dval_min)) >= self.tol:
            return 0
        return 1

    cdef bint stop_condition2(self):        
        if fabs(self.dval2 - self.dval2_prev) / (1 + fabs(self.dval2_min)) >= self.tol:
            return 0
        return 1
    
#     def stop_condition(self):        
#         if fabs(self.dval - self.dval_prev) >= self.tol:
#             return 0
#         return 1
    