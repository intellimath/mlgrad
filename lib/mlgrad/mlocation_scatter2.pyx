# coding: utf-8

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
cimport mlgrad.inventory as inventory

from mlgrad.kmeans import init_centers2

from mlgrad.distance cimport Distance, DistanceWithScale, MahalanobisDistance

from cython.parallel cimport parallel, prange
from openmp cimport omp_get_num_procs

cdef int num_procs = omp_get_num_procs()
if num_procs > 4:
    num_procs /= 2
else:
    num_procs = 2

cdef double max_double = PyFloat_GetMax()

cdef void arithmetic_mean(double[:, ::1] X, double[::1] loc):
    cdef Py_ssize_t i, n = X.shape[1], N = X.shape[0]
    cdef double v

    for i in range(n):
        v = 0
        for k in range(N):
            v += X[k,i]
        loc[i] = v / N

cdef void _covariance_matrix(double[:, ::1] X, double[::1] loc, double[:,::1] S) noexcept nogil:
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

def covariance_matrix(double[:, ::1] X, double[::1] loc, double[:,::1] S):
        _covariance_matrix(X, loc, S)
            
cdef void _covariance_matrix_weighted(
            double *X, const double *W, const double *loc, double *S, 
            const Py_ssize_t n, const Py_ssize_t N) noexcept nogil:
    cdef Py_ssize_t i, j, k
    cdef double s, loc_i, loc_j
    cdef double *X_ki
    cdef double *X_kj

    for i in range(n):
        loc_i = loc[i]
        for j in range(i, n):
            loc_j = loc[j]
            X_kj = X + j
            X_ki = X + i

            s = 0
            for k in range(N):
                s += W[k] * (X_ki[0] - loc_i) * (X_kj[0] - loc_j)
                X_ki += n
                X_kj += n

            S[i*n+j] = S[j*n+i] = s
            
def covariance_matrix_weighted(double[:, ::1] X, double[::1] W, 
                                     double[::1] loc, double[:,::1] S):
        _covariance_matrix_weighted(&X[0,0], &W[0], &loc[0], &S[0,0], X.shape[1], X.shape[0])

cdef void _location_weighted(double *X, const double *W, double *loc, 
                             const Py_ssize_t n, const Py_ssize_t N) noexcept nogil:
    cdef Py_ssize_t i, k
    cdef double *Xi
    cdef double c

    for i in range(n):
        c = 0
        Xi = X + i
        for k in range(N):
            c += W[k] * Xi[0]
            Xi += n
        loc[i] = c
            
def location_weighted(double[:,::1] X, double[::1] W, double[::1] loc):
    _location_weighted(&X[0,0], &W[0], &loc[0], X.shape[1],  X.shape[0])

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

cdef double _scale_matrix(double[:,::1] S):
    cdef Py_ssize_t n = S.shape[0]
    cdef double vol
    cdef Py_ssize_t i
    cdef double *ptr

    if n == 2:
        vol = det_matrix_2(S)
    else:
        vol = linalg.det(np.asarray(S))
    if vol <= 0:
        print('-', S)
    vol = vol ** (1.0/n)
    vol = 1/vol
    ptr = &S[0,0]
    for i in range(n*n):
        ptr[i] *= vol
        
    return vol

def scale_matrix(S):
    _scale_matrix(S)

def init_locations(X, double[:,::1] locs):

    N = X.shape[0]
    n = X.shape[1]
    n_locs = len(locs)

    k0 = rand(N)
    indices = np.random.randint(0, N, n_locs, 'i')

    for j in range(n_locs):
        m = indices[j]
        # inventory.move(locs[j], X[m])
        copy_memoryview(locs[j], X[m])

def  init_scatters(double[:,:,::1] scatters):
    cdef Py_ssize_t i, n, n_locs
    cdef double[:,::1] S

    n_locs = scatters.shape[0]
    n = scatters.shape[1]

    for i in range(n_locs):
        S = np.identity(n, 'd')
        inventory._move(&scatters[i,0,0], &S[0,0], n*n)
        # copy_memoryview2(scatters[i], S)

cdef class MLSE2:

    cpdef evaluate_distances(self):
        cdef Py_ssize_t j, k
        cdef Py_ssize_t  N = self.N
        cdef Py_ssize_t  n_locs = self.n_locs
        cdef Py_ssize_t  n = self.n
        cdef double[::1] DM = self.DD
        cdef double[::1] D = self.D
        cdef double[:,::1] GG = self.GG
        cdef double[:,::1] locs = self.locs
        cdef double[:,::1] X = self.X
        cdef double[::1] X_k
        cdef DistanceWithScale df
        cdef DistanceWithScale[::1] distfuncs = self.distfuncs
        cdef Average avg_min = self.avg_min
        cdef double[::1] GF = self.GF
 
        for k in range(N):
            for j in range(n_locs):
                df = distfuncs[j]
                DM[j] = df._evaluate(&X[k,0], &locs[j,0], n)

            D[k] = avg_min._evaluate(DM)
            avg_min._gradient(DM, GF)
            for j in range(n_locs):
                GG[j,k] = GF[j]

        return D

    cpdef evaluate_weights(self):
        cdef double u = self.avg._evaluate(self.D)
        self.avg._gradient(self.D, self.weights)
        self.calc_update_GG()

    cdef calc_update_GG(self):
        cdef Py_ssize_t j, k
        cdef Py_ssize_t N = self.N, n = self.n, n_locs = self.n_locs
        cdef double wk, wj
        cdef double[:,::1] GG = self.GG
        cdef double[::1] GG_j
        cdef double[::1] W = self.W
        cdef double[::1] weights = self.weights

#         for k in prange(N, nogil=True, num_threads=num_procs):
        for j in range(n_locs):
            inventory.mul(GG[j], weights)
            # GG_j = GG[j]
            # for k in range(N):
            #     GG_j[k] *= weights[k]

#         num = int_min(num_procs, self.n_locs)
#         for j in prange(self.n_locs, nogil=True, num_threads=num):
        for j in range(n_locs):
            W[j] = inventory.sum(GG[j])
            # GG_j = GG[j]
            # wj = 0
            # for k in range(N):
            #     wj += GG_j[k]
            # W[j] = wj

#         for k in prange(N, nogil=True, num_threads=num_procs):
        for j in range(n_locs):
            inventory.mul_const(GG[j], 1./W[j])
            # wj = W[j]
            # GG_j = GG[j]
            # for k in range(N):
            #     GG_j[k] /= wj

    cpdef double Q(self):
        self.evaluate_distances()
        return self.avg._evaluate(self.D)

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
                inventory.move2(distfunc.S, S)
                # copy_memoryview2(distfunc.S, S)

cdef class MLocationsScattersEstimator(MLSE2):

    def __init__(self, Average avg, Average avg_min, n_locs, 
                 tol=1.0e-6, n_iter_c=20, n_iter_s=10, n_iter=100, normalize_S=1):
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
        self.n_iter_c = n_iter_c
        self.n_iter_s = n_iter_s
        self.n_iter = n_iter
        self.normalize_S = normalize_S

    def init(self, double[:,::1] X, warm=False):
        cdef DistanceWithScale df

        self.n = n = X.shape[1]
        self.N = N = X.shape[0]
        self.X = X
        self.distfuncs = np.empty(self.n_locs, object)
        for i in range(self.n_locs):
            df = <DistanceWithScale>MahalanobisDistance(np.identity(n, 'd'))
            self.distfuncs[i] = df

        self.D  = np.zeros(N, 'd')
        self.DD  = np.zeros(self.n_locs, 'd')
        self.GG  = np.zeros((self.n_locs, N), 'd')

        self.weights = np.full(N, 1./N, 'd')
        self.W = np.zeros(self.n_locs, 'd')
        self.GF = np.zeros(self.n_locs, 'd')
        self.Lambda = np.ones(self.n_locs, 'd')
        self.dval_prev = self.dval_min = self.dval = max_double/2
        self.dval2_prev = self.dval2_min = self.dval2 = max_double/2
        self.dvals = []
        self.dvals2 = []

        if warm:
            self.init_locations(X, self.locs)
            self.init_scatters(X, self.scatters)
        else:
            self.init_locations(X)
            self.init_scatters(X)

        # self.evaluate_distances()
        # self.evaluate_weights()

    def init_locations(self, double[:,::1] X, double[:,::1] locs=None):
        n = self.n
        N = self.N
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
        inventory.move3(self.scatters_min, self.scatters)

        self.update_distfuncs(self.scatters)

    def evaluate(self, double[:,::1] X):
        cdef double d, d_min
        cdef Py_ssize_t j, j_min
        cdef Py_ssize_t k, N = X.shape[0], n = X.shape[1]
        cdef Py_ssize_t[::1] Y = np.zeros(N, 'l')
        cdef DistanceWithScale distfunc
        cdef DistanceWithScale[::1] distfuncs = self.distfuncs
        cdef double[:,::1] locs = self.locs

        for k in range(N):
            d_min = max_double
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
        cdef double d, d_min, double_max = PyFloat_GetMax()
        cdef Py_ssize_t j
        cdef Py_ssize_t k, N = X.shape[0], n = X.shape[1]
        cdef DistanceWithScale distfunc
        cdef DistanceWithScale[::1] distfuncs = self.distfuncs
        cdef double[:,::1] locs = self.locs
        cdef double[::1] D = np.zeros(N, 'd')

        for k in range(N):
            d_min = double_max
            for j in range(self.n_locs):
                distfunc = distfuncs[j]
                d = distfunc._evaluate(&X[k,0], &locs[j,0], n)
                if d < d_min:
                    d_min = d
            D[k] = d_min

        return D.base

    def fit_locations(self, double[:,::1] X, double[:,::1] locs=None):
        cdef Py_ssize_t j, K = 0

        is_completed = False
        self.dval = self.dval_min = self.Q()
        self.dvals.append(self.dval)
        inventory.move2(self.locs_min, self.locs)
        # copy_memoryview2(self.locs_min, self.locs)
        for K in range(self.n_iter_c):
            self.dval_prev = self.dval
            
            self.evaluate_distances()
            self.evaluate_weights()

            # self.fit_step_locations()
            for j in range(self.n_locs):
                _location_weighted(&self.X[0,0], &self.GG[j,0], &self.locs[j,0], 
                                   self.X.shape[1], self.X.shape[0])
            
            self.dval = self.Q()
            self.dvals.append(self.dval)

            if self.stop_condition():
                is_completed = True
            
            if self.dval < self.dval_min:
                self.dval_min = self.dval
                inventory.move2(self.locs_min, self.locs)
                # copy_memoryview2(self.locs_min, self.locs)
            
            if is_completed:
                break

        # copy_memoryview2(self.locs, self.locs_min)
        inventory.move2(self.locs, self.locs_min)

#     def fit_step_locations(self):
#         cdef Py_ssize_t n = self.n, N = self.N
#         cdef Py_ssize_t i, j, k, l
#         # cdef double v, wk, wkj, Wj
#         cdef double[:,::1] X = self.X
#         cdef double[:,::1] locs = self.locs
#         # cdef double[::1] W = self.W
#         cdef double[::1] locs_j
#         # cdef double[::1] Xk
#         cdef double[:, ::1] GG = self.GG
#         # cdef double[::1] GG_k
#         # cdef double alpha = self.alpha

#         # multiply_memoryview2(locs, 0)
#         # inventory.clear2(locs)
#         for j in range(self.n_locs):
#             _location_weighted(&X[0,0], &GG[j,0], &locs[j,0], X.shape[1], X.shape[0])
#         # for k in range(N):
#         #     GG_k = GG[k]
#         #     Xk = X[k]
#         #     for j in range(self.n_locs):
#         #         locs_j = locs[j]
#         #         gkj = alpha * GG_k[j]
#         #         for i in range(n):
#         #             locs_j[i] += gkj * Xk[i]

    def fit_scatters(self, double[:,::1] X, double[:,:,::1] scatters=None):
        # self.evaluate_distances()
        # self.evaluate_weights()
        # self.fit_step_scatters()
        
        cdef Py_ssize_t K = 0

        is_completed = False
        self.dval = self.dval_min = self.Q()
        self.dvals.append(self.dval)

        # copy_memoryview3(self.scatters_min, self.scatters)
        inventory.move3(self.scatters_min, self.scatters)
        for K in range(self.n_iter_s):
            self.dval_prev = self.dval

            self.evaluate_distances()
            self.evaluate_weights()
            self.fit_step_scatters()
            
            self.dval = self.Q()
            self.dvals.append(self.dval)
            
            if self.stop_condition():
                is_completed = True

            if self.dval < self.dval_min:
                self.dval_min = self.dval
                inventory.move3(self.scatters_min, self.scatters)
                # copy_memoryview3(self.scatters_min, self.scatters)
            
            if is_completed:
                break            
        
        # copy_memoryview3(self.scatters, self.scatters_min)
        inventory.move3(self.scatters, self.scatters_min)

        self.update_distfuncs(self.scatters)

    def fit_step_scatters(self):
        cdef Py_ssize_t i, j, k, l
        cdef Py_ssize_t N = self.X.shape[0], n = self.X.shape[1]
        # cdef double wk, vv
        cdef double[:,::1] X = self.X
#         cdef DistanceWithScale[::1] distfuncs = self.distfuncs
        cdef double[:,:,::1] scatters = self.scatters
        cdef double[:,::1] S, S1
        cdef double[:,::1] locs = self.locs
        cdef double[::1] weights = self.weights
        # cdef double[::1] W = self.W
        cdef double[:, ::1] GG = self.GG
        # cdef double[::1] GG_l
        # cdef double alpha = self.alpha

        # cdef double[::1] loc
        # cdef double[::1] Xk
        # cdef double[::1] Si

        # multiply_memoryview3(scatters, 0)
        # inventory.mul_const3(scatters, 1-alpha)
        for l in range(self.n_locs):
            _covariance_matrix_weighted(&X[0,0], &GG[l,0], &locs[l,0], 
                                        &scatters[l,0,0], X.shape[1], X.shape[0])
            # S = scatters[l]
            # loc = locs[l]
            # GG_l = GG[l]
            # for k in range(N):
            #     Xk = X[k]
            #     wk = GG[k,l]
            #     for i in range(n):
            #         vv = alpha * wk * (Xk[i] - loc[i])
            #         Si = S[i]
            #         for j in range(i,n):
            #             Si[j] += vv * (Xk[j] - loc[j])
            #             if j > i:
            #                 S[j,i] = Si[j]

        for l in range(self.n_locs):
            S = scatters[l]
            # self.Lambda[l] = linalg.det(S) ** (1./n)
            if self.normalize_S:
                _scale_matrix(S)
            if n == 2:
                S = inv_matrix_2(S)
            else:
                S1 = linalg.inv(np.asarray(S))
                copy_memoryview2(S, S1)
                # inventory.move2(S, S1)

        self.update_distfuncs(scatters)

    def fit(self, double[:,::1] X, only=None, warm=False):
        cdef Py_ssize_t K = 0
        
        self.init(X, warm)

        scatters_only = (only == 'scatters')
        locations_only = (only == 'locations')

        is_completed = False

        self.dval2 = self.dval2_min = self.Q()
        self.dvals2.append(self.dval2)
        
        for K in range(self.n_iter):
            self.dval2_prev = self.dval2

            if not scatters_only:
                self.fit_locations(self.X, self.locs)

            if not locations_only:
                self.fit_scatters(self.X, self.scatters)

            self.dval2 = self.Q()
            self.dvals2.append(self.dval2)

            if self.stop_condition2():
                is_completed = True

            if self.dval2 < self.dval2_min:
                self.dval2_min = self.dval2

            if is_completed:
                break

        self.Ks = K

    cdef bint stop_condition(self):
        if fabs(self.dval - self.dval_min) / (1 + fabs(self.dval_min)) >= self.tol:
            return 0
        return 1

    cdef bint stop_condition2(self):
        if fabs(self.dval2 - self.dval2_min) / (1 + fabs(self.dval2_min)) >= self.tol:
            return 0
        return 1
