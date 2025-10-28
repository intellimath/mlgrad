# coding: utf-8

# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: language_level=3
# cython: embedsignature=True
# cython: initializedcheck=False

# The MIT License (MIT)f
#
# Copyright (c) <2015-2021> <Shibzukhov Zaur, szport at gmail dot com>
#z
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

# from cython.parallel cimport parallel, prange

# cimport mlgrad.inventory as inventory

__averager_dict = {
    '' : ArraySave,
    'AMom':ArrayAMOM,
    'Mom':ArrayMOM,
    'RMS':ArrayRMSProp,
    'AdaM1':ArrayAdaM1,
    'AdaM2':ArrayAdaM2,
}

def _get_averager(str name, default=ArraySave):
    return  __averager_dict.get(name, default)

cdef class ArrayAverager:
    #
    cdef _init(self, ndim):
        pass
    #
    def init(self, ndim):
        self._init(ndim)
    #
    cdef void set_param1(self, double val):
        pass
    #
    cdef void set_param2(self, double val):
        pass
    #
    cdef _update(self, const double[::1] x, const double h):
        pass
    #
    def update(self, x, h):
        self._update(x, h)

cdef class ArraySave(ArrayAverager):

    def __init__(self, beta=Beta, normalize=1):
        self.array_average = None
    #
    cdef _init(self, ndim):
        if self.array_average is None:
            self.array_average = np.zeros(ndim, dtype='d')
        else:
            fill_memoryview(self.array_average, 0)
    #
    cdef _update(self, const double[::1] x, const double h):
        cdef Py_ssize_t i
        cdef double[::1] array_average = self.array_average

#         for i in prange(m, nogil=True, schedule='static', num_threads=inventory.get_num_threads()):
        for i in range(x.shape[0]):
            array_average[i] = h * x[i]

cdef class ArrayMOM(ArrayAverager):

    def __init__(self, beta=Beta, normalize=0):
        self.beta = beta
        self.mgrad = None
        self.array_average = None
        self.normalize = normalize
    #
    cdef _init(self, ndim):
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
    cdef _update(self, const double[::1] x, const double h):
        cdef Py_ssize_t i, m = self.mgrad.shape[0]
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

cdef class ArrayRUD(ArrayAverager):

    def __init__(self, mu=0.8):
        self.mu = mu
        self.vgrad = None
        self.array_average = None
    #
    cdef _init(self, ndim):
        if self.vgrad is None:
            self.vgrad = np.zeros(ndim, dtype='d')
        else:
            fill_memoryview(self.vgrad, 0)

        if self.array_average is None:
            self.array_average = np.zeros(ndim, dtype='d')
        else:
            fill_memoryview(self.array_average, 0)
    #
    cdef void set_param1(self, double val):
        self.mu = val
    #
    cdef _update(self, const double[::1] x, const double h):
        cdef Py_ssize_t i, m = self.vgrad.shape[0]
        cdef double mu = self.mu
        cdef double[::1] vgrad = self.vgrad
        cdef double[::1] array_average = self.array_average

        for i in range(m):
            vgrad[i] = mu * vgrad[i] + x[i]
            array_average[i] = h * vgrad[i]

cdef class ArrayRMSProp(ArrayAverager):

    def __init__(self, beta=Beta, epsilon=Epsilon):
        self.beta = beta
        self.epsilon = epsilon
        self.vgrad = None
        self.array_average = None
    #
    cdef void set_param1(self, double val):
        self.beta = val
    #
    cdef _init(self, ndim):
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
    cdef _update(self, const double[::1] x, const double h):
        cdef Py_ssize_t i, m = self.vgrad.shape[0]
        cdef double v, g, vv, epsilon = self.epsilon
        cdef double beta = self.beta
        cdef double[::1] vvgrad = self.vvgrad
        cdef double[::1] array_average = self.array_average

        self.M *= beta
        self.M += 1-beta
        for i in range(m):
            v = x[i]
            vvgrad[i] = beta * vvgrad[i] + (1-beta) * v*v
            vv = vvgrad[i] / self.M
            array_average[i] = h * v / (sqrt(vv) + epsilon)

cdef class ArrayAdaM2(ArrayAverager):

    def __init__(self, beta1=Beta1, beta2=Beta2, epsilon=Epsilon2):
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
    cdef _init(self, ndim):
        self.beta1_k = 1.
        self.beta2_k = 1.

        if self.mgrad is None:
            self.mgrad = np.zeros(ndim, dtype='d')
        else:
            inventory.clear(self.mgrad)
        if self.vgrad is None:
            self.vgrad = np.zeros(ndim, dtype='d')
        else:
            inventory.clear(self.vgrad)
        if self.array_average is None:
            self.array_average = np.zeros(ndim, dtype='d')
        else:
            inventory.clear(self.array_average)
    #
    cdef _update(self, const double[::1] x, const double h):
        cdef Py_ssize_t i, m = self.mgrad.shape[0]
        cdef double v, v2, mv, vv
        cdef double beta1 = self.beta1, beta2 = self.beta2 
        cdef double[::1] mgrad = self.mgrad
        cdef double[::1] vgrad = self.vgrad
        cdef double[::1] array_average = self.array_average
        cdef double beta1_k, beta2_k
        cdef double epsilon = self.epsilon

        self.beta1_k *= beta1
        self.beta2_k *= beta2
        # self.beta1_k += 1
        # self.beta2_k += 1
        beta1_k = self.beta1_k
        beta2_k = self.beta2_k
        # for i in prange(m, nogil=True, num_threads=inventory.get_num_threads()):
        for i in range(m):
            v = x[i]
            mgrad[i] = beta1 * mgrad[i] + (1-beta1)*v
            mv = mgrad[i] / (1-beta1_k)

            vgrad[i] = beta2 * vgrad[i] + (1-beta2)*v*v
            vv = vgrad[i] / (1-beta2_k)
            if vv < 0:
                raise RuntimeError(f"{vv}")
            v2 = sqrt(vv)
            array_average[i] = h * (mv / (v2 + epsilon))

cdef class ArrayAdaM1(ArrayAverager):

    def __init__(self, beta1=Beta1, beta2=Beta2, epsilon=Epsilon):
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
    cdef _init(self, ndim):
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
    cdef _update(self, const double[::1] x, const double h):
        cdef Py_ssize_t i, m = self.mgrad.shape[0]
        cdef double v, v2, mv, vv
        cdef double beta1 = self.beta1, beta2 = self.beta2
        cdef double[::1] mgrad = self.mgrad
        cdef double[::1] vgrad = self.vgrad
        cdef double[::1] array_average = self.array_average
        cdef double beta1_k, beta2_k
        cdef double epsilon = self.epsilon
    
        self.beta1_k *= beta1
        self.beta2_k *= beta2
        # self.beta1_k += 1
        # self.beta2_k += 1
        beta1_k = self.beta1_k
        beta2_k = self.beta2_k
        # for i in prange(m, nogil=True, schedule='static', num_threads=inventory.get_num_threads()):
        for i in range(m):
            v = x[i]
            mgrad[i] = beta1 * mgrad[i] + (1-beta1)*v
            mv = mgrad[i] / (1 - beta1_k)

            vgrad[i] = beta2 * vgrad[i] + (1-beta1)*fabs(v)
            vv = vgrad[i] / (1 - beta2_k)
            v2 = fabs(vv)
            array_average[i] = h * (mv / (v2 + epsilon))

cdef class ArrayNG(ArrayAverager):

    def __init__(self, beta=Beta, epsilon=1.0e-8):
        self.beta = beta
        self.epsilon = epsilon
        self.mgrad = None
        self.array_average = None
    #
    cdef _init(self, ndim):
        self.beta_k = 1.
        
        if self.mgrad is None:
            self.mgrad = np.zeros(ndim, dtype='d')
        else:
            fill_memoryview(self.mgrad, 0)

        if self.array_average is None:
            self.array_average = np.zeros(ndim, dtype='d')
        else:
            fill_memoryview(self.array_average, 0)
    #
    cdef _update(self, double[::1] x, const double h):
        cdef Py_ssize_t i, m = self.mgrad.shape[0]
        cdef double mn, v, mv
        cdef double beta = self.beta, beta1, beta_k1
        cdef double[::1] mgrad = self.mgrad
        cdef double[::1] array_average = self.array_average
    
        self.beta_k *= beta
        beta_k1 = 1 - self.beta_k
        beta1 = 1 - beta
        mn = 0.
        for i in range(m):
            v = x[i]
            mn += v*v
        mn = sqrt(mn)
            
        for i in range(m):
            v = x[i] / mn
            mgrad[i] = beta * mgrad[i] + beta1 * v 
            mv = mgrad[i] / beta_k1

            array_average[i] = h * mv

cdef class ArrayTAverager(ArrayAverager):
    #
    def __init__(self):
        self.array_average = None
        self.array_sum = None
    #
    cdef _init(self, ndim):
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
    cdef _update(self, const double[::1] x, const double h):
        cdef Py_ssize_t i, m = self.array_sum.shape[0]
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
    cdef _init(self, ndim):
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
    cdef _update(self, const double[::1] x, const double h):
        cdef Py_ssize_t i, m = self.array_sum.shape[0]
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
    cdef _init(self, ndim):
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
    cdef _update(self, const double[::1] x, const double h):
        cdef Py_ssize_t j, n, m = self.array_average.shape[0]
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
    cdef _init(self, ndim, N):
        pass
    #
    def init(self, ndim, N):
        self._init(ndim, N)
    #
    cdef _update(self, double[::1] x, int k):
        pass
    # 
    def update(self, x, k):
        self._update(x, k)

cdef class ArrayStochasticAverager(SArrayAverager):

    def __init__(self):
        self.array_average = None
        self.array_table = None
    #
    cdef _init(self, m, N):            
        if self.array_average is None:
            self.array_average = np.zeros(m, dtype='d')
        else:
            fill_memoryview(self.array_average, 0)

        if self.array_table is None:
            self.array_table = np.zeros((N, m), dtype='d')
        else:
            fill_memoryview2(self.array_table, 0)
    #
    cdef _update(self, const double[::1] x, const int k):
        cdef Py_ssize_t i, N = self.array_table.shape[0], m = self.array_average.shape[0]
        cdef double Nd = N
        cdef double[::1] array_average = self.array_average
        cdef double[:,::1] array_table = self.array_table
        
        for i in range(m):
            xi = self.array_table[k,i]
            array_table[k,i] = x[i]
            array_average[i] += (x[i] - xi) / Nd
