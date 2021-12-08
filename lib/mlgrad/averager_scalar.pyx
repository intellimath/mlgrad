# coding: utf-8

#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: language_level=3
#cython: embedsignature=True
#cython: initializedcheck=False

# The MIT License (MIT)
#
# Copyright (c) <2015-2021> <Shibzukhov Zaur, szport at gmail dot com>
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
#

cdef class ScalarAverager:
    #
    cdef init(self):
        pass
    #
    cdef double update(self, const double x):
        return x
        
cdef class ScalarAdaM2(ScalarAverager):

    def __init__(self, beta1=Beta1, beta2=Beta2, epsilon=Epsilon):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
    #
    cdef init(self):
        self.m = 0
        self.v = 0
        self.beta1_k = 1.
        self.beta2_k = 1.
    #
    cdef double update(self, double x):
        cdef double m_tilde, v_tilde
        cdef double beta1 = self.beta1, beta2 = self.beta2
    
        self.m = (1.0 - beta1) * x + beta1 * self.m
        self.beta1_k *= beta1
        m_tilde = self.m  / (1.0 - self.beta1_k)

        self.v = (1.0 - beta2) * x * x + beta2 * self.v
        self.beta2_k *= beta2
        v_tilde = self.v / (1.0 - self.beta2_k)

        return m_tilde / (sqrtf(v_tilde) + self.epsilon)        

cdef class ScalarAdaM1(ScalarAverager):

    def __init__(self, beta1=Beta1, beta2=Beta2, epsilon=Epsilon):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
    #
    cdef init(self):
        self.m = 0
        self.v = 0
        self.beta1_k = 1.
        self.beta2_k = 1.
    #
    cdef double update(self, const double x):
        cdef double m_tilde, v_tilde
        cdef double beta1 = self.beta1, beta2 = self.beta2
    
        self.m = (1.0 - beta1) * x + beta1 * self.m
        self.beta1_k *= beta1
        m_tilde = self.m  / (1.0 - self.beta1_k)

        self.v = (1.0 - beta2) * fabsf(x) + beta2 * self.v
        self.beta2_k *= beta2
        v_tilde = self.v / (1.0 - self.beta2_k)

        return m_tilde / (v_tilde + self.epsilon)        

cdef class ScalarExponentialScalarAverager(ScalarAverager):

    def __init__(self, beta=Beta):
        self.beta = beta
    #
    cdef init(self):
        self.m = 0
        self.beta_k = 1.
    #
    cdef double update(self, const double x):    
        self.m = (1.0 - self.beta) * x + self.beta * self.m
        self.beta_k *= self.beta
        return self.m  / (1.0 - self.beta_k)

cdef class ScalarWindowAverager(ScalarAverager):

    def __init__(self, size=21):
        self.size = size
        self.buffer = np.zeros((size,), 'd')
    #
    cdef init(self):
        cdef int i
        
        for i in range(self.size):
            self.buffer[i] = 0
            
        self.idx = 0
        self.first = 1
        self.buffer_sum = 0
    #
    cdef double update(self, const double x):
        if self.idx == self.size:
            self.idx = 0
            if self.first:
                self.first = 0

        if self.first:
            self.buffer_sum += x
            self.buffer[self.idx] = x
            self.idx += 1            
            return self.buffer_sum / self.idx
        else:
            self.buffer_sum += x - self.buffer[self.idx]
            self.buffer[self.idx] = x
            self.idx += 1            
            
        return self.buffer_sum / self.size
