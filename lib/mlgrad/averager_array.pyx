# coding: utf-8

#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: language_level=3
#cython: embedsignature=True
#cython: initializedcheck=False

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

from cython.parallel cimport parallel, prange
#

from openmp cimport omp_get_num_procs

cdef int num_procs = omp_get_num_procs()
if num_procs >= 4:
    num_procs /= 2
else:
    num_procs = 2

cdef class ArrayAverager:
    #
    cdef init(self, ndim):
        pass
    #
    cdef void set_param1(self, double val):
        pass
    #
    cdef void set_param2(self, double val):
        pass
    #
    cdef void update(self, double[::1] x, double h):
        pass

cdef class ArraySave(ArrayAverager):

    def __init__(self, beta=Beta, normalize=1):
        self.array_average = None
    #
    cdef init(self, ndim):
        if self.array_average is None:
            self.array_average = np.zeros(ndim, dtype='d')
        else:
            fill_memoryview(self.array_average, 0)
    #
    cdef void update(self, double[::1] x, double h):
        cdef int i, m = x.shape[0]
        cdef double[::1] array_average = self.array_average 
        
        for i in prange(m, nogil=True, schedule='static', num_threads=num_procs):
            array_average[i] = h * x[i]
    
cdef class ArrayMOM(ArrayAverager):

    def __init__(self, beta=Beta, normalize=0):
        self.beta = beta
        self.mgrad = None
        self.array_average = None
        self.normalize = normalize
    #
    cdef init(self, ndim):
        self.M = 0
        
        if self.mgrad is None:
            self.mgrad = np.zeros(ndim, dtype='d')
        else:
            fill_memoryview(self.mgrad, 0)
            
        if self.array_average is None:
            self.array_average = np.zeros(ndim, dtype='d')
        else:
            fill_memoryview(self.array_average, 0)
    #
    cdef void set_param1(self, double val):
        self.beta = val
    #
    cdef void update(self, double[::1] x, double h):
        cdef int i, m = self.mgrad.shape[0]
        cdef double beta = self.beta
        cdef double[::1] mgrad = self.mgrad
        cdef double[::1] array_average = self.array_average
    
        for i in range(m):
            mgrad[i] = beta * mgrad[i] + x[i]

        if self.normalize:
            self.M *= beta
            self.M += 1
            for i in range(m):
                array_average[i] = h * mgrad[i] / self.M
        else:
            for i in range(m):
                array_average[i] = h * mgrad[i]
    
cdef class ArrayAMOM(ArrayAverager):

    def __init__(self, beta=Beta, normalize=0):
        self.beta = beta
        self.mgrad = None
        self.array_average = None
        self.normalize = normalize
    #
    cdef void set_param1(self, double val):
        self.beta = val
    #
    cdef init(self, ndim):
        self.M = 0
        
        if self.mgrad is None:
            self.mgrad = np.zeros(ndim, dtype='d')
        else:
            fill_memoryview(self.mgrad, 0)
            
        if self.array_average is None:
            self.array_average = np.zeros(ndim, dtype='d')
        else:
            fill_memoryview(self.array_average, 0)
    #
    cdef void update(self, double[::1] x, double h):
        cdef int i, m = self.mgrad.shape[0]
        cdef double v, mv, vv
        cdef double beta = self.beta
        cdef double[::1] mgrad = self.mgrad
        cdef double[::1] array_average = self.array_average

        for i in range(m):
            mgrad[i] = beta * mgrad[i] + (1-beta) * x[i]

        if self.normalize:
            self.M *= beta
            self.M += 1-beta
            for i in range(m):
                array_average[i] = h * mgrad[i] / self.M
        else:
            for i in range(m):
                array_average[i] = h * mgrad[i]

# cdef class ArrayRUD(ArrayAverager):

#     def __init__(self, beta=Beta, normalize=0):
#         self.beta = beta
#         self.mgrad = None
#         self.array_average = None
#         self.normalize = normalize
#     #
#     cdef init(self, ndim):
#         self.M = 0
        
#         if self.mgrad is None:
#             self.mgrad = np.zeros(ndim, dtype='d')
#         else:
#             fill_memoryview(self.mgrad, 0)
            
#         if self.array_average is None:
#             self.array_average = np.zeros(ndim, dtype='d')
#         else:
#             fill_memoryview(self.array_average, 0)
#     #
#     cdef void set_param1(self, double val):
#         self.beta = val
#     #
#     cdef void update(self, double[::1] x, double h):
#         cdef int i, m = self.mgrad.shape[0]
#         cdef double beta = self.beta
#         cdef double[::1] mgrad = self.mgrad
#         cdef double[::1] array_average = self.array_average
    
#         for i in range(m):
#             mgrad[i] = beta * mgrad[i] + x[i]

#         if self.normalize:
#             self.M *= beta
#             self.M += 1
#             for i in range(m):
#                 array_average[i] = h * mgrad[i] / self.M
#         else:
#             for i in range(m):
#                 array_average[i] = h * mgrad[i]

cdef class ArrayRMSProp(ArrayAverager):

    def __init__(self, beta=Beta, epsilon=1.0e-8):
        self.beta = beta
        self.epsilon = epsilon
        self.vgrad = None
        self.array_average = None
    #
    cdef void set_param1(self, double val):
        self.beta = val
    #
    cdef init(self, ndim):
        self.M = 0
                    
        if self.vgrad is None:
            self.vgrad = np.zeros(ndim, dtype='d')
        else:
            fill_memoryview(self.vgrad, 0)
            
        if self.array_average is None:
            self.array_average = np.zeros(ndim, dtype='d')
        else:
            fill_memoryview(self.array_average, 0)
    #
    cdef void update(self, double[::1] x, double h):
        cdef int i, m = self.vgrad.shape[0]
        cdef double v, mv, vv
        cdef double beta = self.beta
        cdef double[::1] vgrad = self.vgrad
        cdef double[::1] array_average = self.array_average
    
        self.M *= beta
        self.M += 1-beta
        for i in range(m):
            v = x[i]
            vgrad[i] = beta * vgrad[i] + (1-beta) * v*v
            vv = vgrad[i] / self.M
            array_average[i] = h * v / (sqrt(vv) + self.epsilon)

cdef class ArrayAdaM2(ArrayAverager):

    def __init__(self, beta1=Beta1, beta2=Beta2, epsilon=1.0e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.mgrad = None
        self.vgrad = None
        self.array_average = None
    #
    cdef void set_param1(self, double val):
        self.beta1 = val
    #
    cdef void set_param2(self, double val):
        self.beta2 = val
    #
    cdef init(self, ndim):
        self.beta1_k = 1.
        self.beta2_k = 1.
        
        if self.mgrad is None:
            self.mgrad = np.zeros(ndim, dtype='d')
        else:
            fill_memoryview(self.mgrad, 0)
            
        if self.vgrad is None:
            self.vgrad = np.zeros(ndim, dtype='d')
        else:
            fill_memoryview(self.vgrad, 0)
            
        if self.array_average is None:
            self.array_average = np.zeros(ndim, dtype='d')
        else:
            fill_memoryview(self.array_average, 0)
    #
    cdef void update(self, double[::1] x, double h):
        cdef int i, m = self.mgrad.shape[0]
        cdef double v, mv, vv
        cdef double beta1 = self.beta1, beta2 = self.beta2 
        cdef double[::1] mgrad = self.mgrad
        cdef double[::1] vgrad = self.vgrad
        cdef double[::1] array_average = self.array_average
    
        self.beta1_k *= beta1
        self.beta2_k *= beta2
        for i in prange(m, nogil=True, schedule='static', num_threads=num_procs):
            v = x[i]
            mgrad[i] = beta1 * mgrad[i] + (1.0 - beta1) * v 
            mv = mgrad[i] / (1.0 - self.beta1_k)

            vgrad[i] = beta2 * vgrad[i] + (1-beta2) * v*v
            vv = vgrad[i] / (1 - self.beta2_k)
            array_average[i] = h * (mv / (sqrt(vv) + self.epsilon))

cdef class ArrayAdaM1(ArrayAverager):

    def __init__(self, beta1=Beta1, beta2=Beta2, epsilon=1.0e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.mgrad = None
        self.vgrad = None
        self.array_average = None
    #
    cdef void set_param1(self, double val):
        self.beta1 = val
    #
    cdef void set_param2(self, double val):
        self.beta2 = val
    #
    cdef init(self, ndim):
        self.beta1_k = 1.
        self.beta2_k = 1.

        if self.mgrad is None:
            self.mgrad = np.zeros(ndim, dtype='d')
        else:
            fill_memoryview(self.mgrad, 0)
        
        if self.vgrad is None:
            self.vgrad = np.zeros(ndim, dtype='d')
        else:
            fill_memoryview(self.vgrad, 0)

        if self.array_average is None:
            self.array_average = np.zeros(ndim, dtype='d')
        else:
            fill_memoryview(self.array_average, 0)
    #
    cdef void update(self, double[::1] x, double h):
        cdef int i, m = self.mgrad.shape[0]
        cdef double v, mv, vv
        cdef double beta1 = self.beta1, beta2 = self.beta2
        cdef double[::1] mgrad = self.mgrad
        cdef double[::1] vgrad = self.vgrad
        cdef double[::1] array_average = self.array_average
    
        self.beta1_k *= beta1
        self.beta2_k *= beta2
        for i in prange(m, nogil=True, schedule='static', num_threads=num_procs):
            v = x[i]
            mgrad[i] = beta1 * mgrad[i] + (1.0 - beta1) * v 
            mv = mgrad[i] / (1.0 - self.beta1_k)

            vgrad[i] = beta2 * vgrad[i] + (1.0 - beta2) * fabs(v)
            vv = vgrad[i] / (1.0 - self.beta2_k)
            array_average[i] = h * mv / (vv + self.epsilon)

# cdef class ArrayAdaNG(ArrayAverager):

#     def __init__(self, beta=Beta, epsilon=1.0e-8):
#         self.beta = beta
#         self.epsilon = epsilon
#         self.mgrad = None
#         self.array_average = None
#     #
#     cdef init(self, ndim):
#         self.beta_k = 1.
        
#         if self.mgrad is None:
#             self.mgrad = np.zeros(ndim, dtype='d')
#         else:
#             fill_memoryview(self.mgrad, 0)

#         if self.array_average is None:
#             self.array_average = np.zeros(ndim, dtype='d')
#         else:
#             fill_memoryview(self.array_average, 0)
#     #
#     cdef double[::1] update(self, double[::1] x):
#         cdef int i, m = self.mgrad.shape[0]
#         cdef double mn, v, mv
#         cdef double beta = self.beta
#         cdef double[::1] mgrad = self.mgrad
#         #cdef double[::1] vgrad = self.vgrad
#         cdef double[::1] array_average = self.array_average
    
#         self.beta_k *= beta
#         mn = 0.
#         for i in range(m):
#             v = x[i]
#             mn += v*v
#         mn = sqrt(mn)
            
#         for i in range(m):
#             v = x[i] / mn
#             mgrad[i] = beta * mgrad[i] + (1.0 - beta) * v 
#             mv = mgrad[i] / (1.0 - self.beta_k)

#             array_average[i] = mv

#         return array_average

cdef class ArrayTAverager(ArrayAverager):
    #
    def __init__(self):
        self.array_average = None
        self.array_sum = None
    #
    cdef init(self, ndim):
        self.T = 1.
        if self.array_average is None:
            self.array_average = np.zeros(ndim, dtype='d')
        else:
            fill_memoryview(self.array_average, 0)
        if self.array_sum is None:
            self.array_sum = np.zeros(ndim, dtype='d')
        else:
            fill_memoryview(self.array_sum, 0)
    #
    cdef void update(self, double[::1] x, double h):
        cdef int i, m = self.array_sum.shape[0]
        cdef double T = self.T
        cdef double[::1] array_average = self.array_average
        cdef double[::1] array_sum = self.array_sum
    
        for i in range(m):
            array_sum[i] += h * x[i] * T
            array_average[i] = 2.0 * array_sum[i] / (T * (T +1.0))
            self.T += 1.0

# cdef class ArrayExponentialAverager(ArrayAverager):

#     def __init__(self, beta=0.9):
#         self.beta = beta
#         self.array_average = None
#         self.array_sum = None
#     #
#     cdef init(self, ndim):
#         self.beta_k = 1.0
#         if self.array_average is None:
#             self.array_average = np.zeros(ndim, dtype='d')
#         else:
#             fill_memoryview(self.array_average, 0)
#         if self.array_sum is None:
#             self.array_sum = np.zeros(ndim, dtype='d')
#         else:
#             fill_memoryview(self.array_sum, 0)
#     #
#     cdef double[::1] update(self, double[::1] x):
#         cdef int i, m = self.array_sum.shape[0]
#         cdef double beta = self.beta
#         cdef double[::1] array_average = self.array_average
#         cdef double[::1] array_sum = self.array_sum
    
#         self.beta_k *= beta
#         for i in range(m):
#             array_sum[i] = beta * array_sum[i] + (1.0 - beta) * x[i]
#             array_average[i] = array_sum[i] / (1 - self.beta_k)
#         return self.array_average

cdef class ArraySimpleAverager(ArrayAverager):
    #
    def __init__(self):
        self.array_average = None
        self.array_sum = None
    #
    cdef init(self, ndim):
        self.T = 1.
        if self.array_average is None:
            self.array_average = np.zeros(ndim, dtype='d')
        else:
            fill_memoryview(self.array_average, 0)
        if self.array_sum is None:
            self.array_sum = np.zeros(ndim, dtype='d')
        else:
            fill_memoryview(self.array_sum, 0)
    #
    cdef void update(self, double[::1] x, double h):
        cdef int i, m = self.array_sum.shape[0]
        cdef double T = self.T
        cdef double[::1] array_average = self.array_average
        cdef double[::1] array_sum = self.array_sum
        
        for i in range(m):
            array_sum[i] += h * x[i]
            array_average[i] = array_sum[i] / T
        self.T += 1.0

cdef class ArrayCyclicAverager(ArrayAverager):
    #
    def __init__(self, size):
        self.size = size
        self.array_average = None
        self.array_all = None
    #
    cdef init(self, ndim):
        self.i = 0
        if self.array_all is None:
            self.array_all = np.zeros((self.size, ndim,), dtype='d')
        else:
            fill_memoryview2(self.array_all, 0)
        if self.array_average is None:
            self.array_average = np.zeros(ndim, dtype='d')
        else:
            fill_memoryview(self.array_average, 0)
        if self.array_sum is None:
            self.array_sum = np.zeros(ndim, dtype='d')
        else:
            fill_memoryview(self.array_sum, 0)
    #
    cdef void update(self, double[::1] x, double h):
        cdef int j, n, m = self.array_average.shape[0]
        cdef double[::1] array_average = self.array_average
        cdef double[:,::1] array_all = self.array_all
    
        j = self.i % self.size
        if self.i >= self.size:
            n = self.size
        else:
            n = j+1
        for i in range(m):
            array_average[i] += (x[i] - array_all[j,i]) / n
            array_all[j,i] = x[i]
        self.i += 1


cdef class SArrayAverager:
    #
    cdef init(self, ndim, N):
        pass
    #
    cdef void update(self, double[::1] x, int k):
        pass

cdef class ArrayStochasticAverager(SArrayAverager):

    def __init__(self):
        self.array_average = None
        self.array_table = None
    #
    cdef init(self, m, N):            
        if self.array_average is None:
            self.array_average = np.zeros(m, dtype='d')
        else:
            fill_memoryview(self.array_average, 0)

        if self.array_table is None:
            self.array_table = np.zeros((N, m), dtype='d')
        else:
            fill_memoryview2(self.array_table, 0)
    #
    cdef void update(self, double[::1] x, int k):
        cdef int N = self.array_table.shape[0], m = self.array_average.shape[0]
        cdef double Nd = N
        cdef double[::1] array_average = self.array_average
        cdef double[:,::1] array_table = self.array_table
        
        for i in range(m):
            xi = self.array_table[k,i]
            array_table[k,i] = x[i]
            array_average[i] += (x[i] - xi) / Nd
