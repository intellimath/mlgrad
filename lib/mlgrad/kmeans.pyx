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

from mlgrad.abc import Fittable
#from cython.parallel import prange

# cdef class ArrayRef:
#     #
#     def __init__(self, double[::1] data):
#         self.data = data
        
# cdef object new_arrayref(double[::1] data):
#         cdef ArrayRef arr = ArrayRef.__new__(ArrayRef)
#         arr.data = data
#         return arr

cpdef find_centers(double[:, ::1] X, int[::1] Y, double[::1] weights, double[:,::1] params):
    cdef int i, j, k, N_j
    cdef int n_sample = X.shape[0]
    cdef int n_class = params.shape[0]
    cdef int n_param = params.shape[1]
    cdef double wk, wj
    cdef double[::1] W_sum
    cdef double *Xk

    fill_memoryview2(params, 0)
    W_sum = np.zeros(n_class, 'd')
        
    for k in range(n_sample):
        j = Y[k]
        Xk = &X[k,0]
        wk = weights[k]
        W_sum[j] += wk
        for i in range(n_param):
            params[j,i] += wk * Xk[i]
            
    for j in range(n_class):
        wj = W_sum[j]
        for i in range(n_param):
            params[j,i] /= wj
            
cdef inline double euclid_distance2(double[::1] x, double[::1] y) nogil:
    cdef int i, n = x.shape[0]
    cdef double dist, v

    dist = 0
    for i in range(n):
        v = x[i] - y[i]
        dist += v*v
    return dist

cpdef find_classes(double[:, ::1] X, double[:,::1] params, int[::1] Y):
    cdef int i, j, k
    cdef int n_sample = X.shape[0]
    cdef int n_class = params.shape[0]
    cdef int n_param = params.shape[1]
    cdef double fmax = PyFloat_GetMax()
    cdef double d, dist_min
    cdef int j_min
    
    for k in range(n_sample):
        dist_min = fmax
        j_min = 0
        for j in range(n_class):
            d = euclid_distance2(X[k], params[j])
            if d < dist_min:
                j_min = j
                dist_min = d

        Y[k] = j_min

    J = np.unique(Y)
    if len(J) < n_class:
        print(params.base)
        print(J)
        raise RuntimeError("One of the clusters is empty")

cpdef double[:, ::1] init_centers(double[:,::1] X, int n_class):
    cdef double[:, ::1] params
    cdef int i, k

    n_sample = X.shape[0]
    n_param = X.shape[1]

    params = np.zeros((n_class, n_param), 'd')
    indexes = np.random.randint(0, n_sample-1, n_class, 'i')
    for i in range(n_class):
        k = indexes[i]
        copy_memoryview(params[i], X[k])
    return params

cpdef double[:, ::1] init_centers2(double[:,::1] X, int n_class):
    cdef double[:, ::1] params
    cdef int[::1] indices
    cdef double d, d_max_min, d_min
    cdef double f_max = PyFloat_GetMax()
    cdef int j, k, k_max, j_min
    cdef int n_sample = X.shape[0]
    cdef int n_param = X.shape[1]
    cdef int n_center
    
    params = np.zeros((n_class, n_param), 'd')
    indices = np.zeros((n_class,), 'i')
    
    k = rand(n_sample)
    copy_memoryview(params[0], X[k])
    indices[0] = k

    n_center = 1
    while n_center < n_class:   
        k_max = 0
        d_max_min = 0
        for k in range(n_sample):
            j_min = 0
            d_min = f_max
            for j in range(n_center):
                if indices[j] == k:
                    j_min = -1
                    break
                d = euclid_distance2(params[j], X[k])
                if d < d_min:
                    d_min = d
                    j_min = j
            if j_min < 0:
                continue
            if d_min > d_max_min:
                d_max_min = d_min
                k_max = k

        copy_memoryview(params[n_center], X[k_max])
        indices[n_center] = k_max
        n_center += 1

    return params

cdef class HCD:

    def __init__(self, Func func, double[:,::1] X, int n_class, 
                 double tol=1.0e-3, int n_iter=100):
        
        self.n_class = n_class
        self.n_sample = X.shape[0]
        self.n_param = X.shape[1]
        
        self.X = X
        self.Y = None
        self.func = func
        
        self.weights = None
        self.params = None
        self.prev_params = None
        
        self.tol = tol
        self.n_iter = n_iter
        
        self.init()
    #
    cpdef init(self):
        #cdef double[::1] Xk
        #cdef int k, N = len(self.X)
        
        init_rand()
        
        if self.weights is None:
            self.weights = np.zeros(self.n_sample, 'd')

        if self.Y is None:
            self.Y = np.zeros(self.n_sample, 'i')
        
        if self.params is None:
            self.params = init_centers2(self.X, self.n_class)
        self.prev_params = np.zeros((self.n_class, self.n_param), 'd')
    #
    def fit(self):
        cdef double[:, ::1] params = self.params
        cdef double[:, ::1] prev_params = self.prev_params
        cdef double[:, ::1] X = self.X
        cdef double[::1] weights = self.weights
        cdef int[::1] Y = self.Y
        cdef int i, j, k
        cdef int n_sample = self.n_sample
        cdef int n_class = self.n_class
        cdef int n_param = self.n_param
        cdef double dist
        cdef bint is_complete = 0
        
        #print(self.params.base)
        self.K = 1
        while self.K <= self.n_iter:
            find_classes(X, params, Y)
            #print("K:", self.K)
            
            for k in range(n_sample):
                j = Y[k]
                dist = euclid_distance2(X[k], params[j])
                weights[k] = self.func.derivative(dist)
            
            copy_memoryview2(prev_params, params)
            find_centers(X, Y, weights, params)
            #print(self.params.base)
            
            if self.stop_condition():
                is_complete = 1
                break
                
            self.K += 1
    #
    cdef bint stop_condition(self):
        cdef double[:, ::1] params = self.params
        cdef double[:, ::1] prev_params = self.prev_params
        cdef double dist
        cdef int i, n_class = self.n_class
        cdef int n_param = params.shape[1]
        
        for i in range(n_class):
            dist = sqrt(euclid_distance2(prev_params[i], params[i]))
            if dist > self.tol:
                return 0
        return 1

cdef class HCD_M1:

    def __init__(self, Average avrfunc, double[:,::1] X, int n_class, 
                 double tol=1.0e-4, int n_iter=100):

        self.n_class = n_class
        self.n_sample = X.shape[0]
        self.n_param = X.shape[1]
        
        self.X = X
        self.Y = None
        self.avrfunc = avrfunc
        
        self.weights = None
        self.dist = None
        self.params = None
        self.prev_params = None
        
        self.tol = tol
        self.n_iter = n_iter
        
        self.init()
    #
    cpdef init(self):
        #cdef double[::1] Xk
        #cdef int k, N = len(self.X)

        init_rand()

        if self.weights is None:
            self.weights = np.zeros(self.n_sample, 'd')

        if self.dist is None:
            self.dist = np.zeros(self.n_sample, 'd')

        if self.Y is None:
            self.Y = np.zeros(self.n_sample, 'i')

        if self.params is None:
            self.params = init_centers2(self.X, self.n_class)
        self.prev_params = np.zeros((self.n_class, self.n_param), 'd')
    #
    def fit(self):
        cdef double[:, ::1] params = self.params
        cdef double[:, ::1] prev_params = self.prev_params
        cdef double[:, ::1] X = self.X
        cdef double[::1] Xk
        cdef double[::1] weights = self.weights
        cdef int[::1] Y = self.Y
        cdef int i, j, k
        cdef int n_sample = self.n_sample
        cdef int n_class = self.n_class
        cdef int n_param = self.n_param
        cdef double[::1] dist = self.dist
        cdef bint is_complete = 0

        #print(self.params.base)
        self.K = 1
        while self.K <= self.n_iter:
            find_classes(X, params, Y)
            #print("K:", self.K)
            #print(np.unique(self.Y.base))
            
            for k in range(n_sample):
                j = Y[k]
                dist[k] = euclid_distance2(X[k], params[j])
            
            self.avrfunc.fit(dist)
            self.avrfunc.gradient(dist, weights)

            copy_memoryview2(prev_params, params)
            find_centers(X, Y, weights, params)
                        
            if self.stop_condition():
                is_complete = 1
                break
            
            self.K += 1

#         if not is_complete:
#             for j in range(n_class):
#                 for i in range(n_param):
#                     params[j,i] = 0.5*(params[j,i] + prev_params[j,i])
    #
    cdef bint stop_condition(self):
        cdef double[:, ::1] params = self.params
        cdef double[:, ::1] prev_params = self.prev_params
        cdef double dist
        cdef int i, n_class = self.n_class
        cdef int n_param = params.shape[1]
        
        for i in range(n_class):
            dist = sqrt(euclid_distance2(prev_params[i], params[i]))
            if dist > self.tol:
                return 0
        return 1

cpdef find_centers2(double[:, ::1] X, double[:, ::1] Y, double[::1] weights, double[:,::1] params):
    cdef int i, j, k
    cdef int n_sample = X.shape[0]
    cdef int n_class = params.shape[0]
    cdef int n_param = params.shape[1]
    cdef double wk, wkj, W
#     cdef double[::1] Xk

    fill_memoryview2(params, 0)
    
    for j in range(n_class):
        W = 0
        for k in range(n_sample):
            wk = weights[k]
            wkj = wk * Y[k,j]
            W += wkj

            for i in range(n_param):
                params[j,i] += wkj * X[k,i]            
        for i in range(n_param):
            params[j,i] /= W

cpdef calc_distance_matrix(double[:, ::1] X, double[:,::1] params, double[:, ::1] D):
    cdef int k, j
    cdef int n_sample = X.shape[0]
    cdef int n_class = params.shape[0]
    cdef int n_param = params.shape[1]

    for k in range(n_sample):
        for j in range(n_class):
            D[k,j] = euclid_distance2(X[k], params[j])

cpdef calc_Y(double[:, ::1] D, double[:, ::1] Y, Average minfunc):
    cdef int n_sample = Y.shape[0]
    cdef int n_class = Y.shape[1]
    cdef double[::1] Dk
    cdef double d, dmin
    cdef int i, k

    for k in range(n_sample):
        Dk = D[k]
        dmin = Dk[0]
        for i in range(n_class):
            d = Dk[i]
            if d < dmin:
                dmin = d
        minfunc.fit(Dk)
        minfunc.gradient(Dk, Y[k])
            
cpdef calc_weights_and_Y(double[:, ::1] D, double[:, ::1] Y, double[::1] weights, Average avrfunc, Average minfunc):
    cdef int n_sample = Y.shape[0]
    cdef int n_class = Y.shape[1]
    cdef double[::1] DD = np.zeros(n_sample, 'd')
    cdef double[::1] Dk, Yk
    cdef double d, dmin
    cdef int j, k

    for k in range(n_sample):
        Dk = D[k]
        dmin = Dk[0]
        for j in range(n_class):
            d = Dk[j]
            if d < dmin:
                dmin = d
        minfunc.fit(Dk)
        DD[k] = minfunc.u
        Yk = Y[k]
        minfunc.gradient(Dk, Yk)
    
    avrfunc.fit(DD)
    avrfunc.gradient(DD, weights)
    
cdef class HCD_M2:

    def __init__(self, Average avrfunc, Average minfunc, double[:,::1] X, int n_class, 
                 double tol=1.0e-4, int n_iter=100):

        self.n_class = n_class
        self.n_sample = X.shape[0]
        self.n_param = X.shape[1]
        
        self.X = X
        self.Y = None
        self.D = None
        self.avrfunc = avrfunc
        self.minfunc = minfunc
        
        self.weights = None
        self.params = None
        self.prev_params = None
        
        self.tol = tol
        self.n_iter = n_iter
        
        self.init()
    #
    cpdef init(self):
        init_rand()

        if self.weights is None:
            self.weights = np.zeros(self.n_sample, 'd')

        if self.Y is None:
            self.Y = np.zeros((self.n_sample, self.n_class), 'd')
        if self.D is None:
            self.D = np.zeros((self.n_sample, self.n_class), 'd')

        if self.params is None:
            self.params = init_centers2(self.X, self.n_class)
        self.prev_params = np.zeros((self.n_class, self.n_param), 'd')
    #
    def fit(self):
        cdef double[:, ::1] params = self.params
        cdef double[:, ::1] prev_params = self.prev_params
        cdef double[:, ::1] X = self.X
        cdef double[::1] weights = self.weights
        cdef double[:, ::1] Y = self.Y
        cdef double[:, ::1] D = self.D
        cdef int i, j, k
        cdef int n_sample = self.n_sample
        cdef int n_class = self.n_class
        cdef int n_param = self.n_param
        cdef bint is_complete = 0

#         print(self.params.base)
        self.K = 1
        while self.K <= self.n_iter:
#             if self.K % 100 == 0:
#             print(params.base)
            copy_memoryview2(prev_params, params)

            calc_distance_matrix(X, params, D)
            calc_weights_and_Y(D, Y, weights, self.avrfunc, self.minfunc)            
            find_centers2(X, Y, weights, params)
                        
            if self.stop_condition():
                is_complete = 1
                break
            
            self.K += 1
        print(self.K, params.base)
    #
    def __call__(self, double[:,::1] X):
        cdef int n_sample = X.shape[0]
        cdef int n_param = X.shape[1]
        cdef double[:,::1] D = np.zeros((n_sample, self.n_class), 'd')
        cdef double[:,::1] Y = np.zeros((n_sample, self.n_class), 'd')

        calc_distance_matrix(X, self.params, D)
        calc_Y(D, Y, self.minfunc)
        
        return Y.base
    #
    cdef bint stop_condition(self):
        cdef double[:, ::1] params = self.params
        cdef double[:, ::1] prev_params = self.prev_params
        cdef double dist
        cdef int i, n_class = self.n_class
        cdef int n_param = params.shape[1]
        
        for i in range(n_class):
            dist = sqrt(euclid_distance2(prev_params[i], params[i]))
            if dist > self.tol:
                return 0
        return 1
    
Fittable.register(HCD)
Fittable.register(HCD_M1)
Fittable.register(HCD_M2)
