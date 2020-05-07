# coding: utf-8

import numpy as np

from mlgrad.func cimport Func
from mlgrad.model cimport Model
from mlgrad.regular cimport Regular

cdef class RK4(GD):
    #
    def __init__(self, Model model, Loss loss, Regular regular=None, 
                 tau=1.0e-3, tol=1.0e-6, h=0.001, n_iter=1000):
        """
        """
        self.model = model
        self.loss = loss
        self.regular = regular
        self.tol = tol
        self.tau = tau
        self.n_iter = n_iter
        self.h = h
        self.W = None
    #
    cpdef fit(self, double_2d X, double_1d Y, double_1d W=None):
        cdef int m = self.model.param.shape[0]

        GD.fit(self, X, Y, W)
        
        self.param_k1 = self.model.param
        self.param_k2 = np.zeros((m,), dtype='d')
        self.param_k3 = np.zeros((m,), dtype='d')
        self.param_k4 = np.zeros((m,), dtype='d')

        self.grad_k1 = self.grad
        self.grad_k2 = np.zeros((m,), dtype='d')
        self.grad_k3 = np.zeros((m,), dtype='d')
        self.grad_k4 = np.zeros((m,), dtype='d')

        self.K = 1
        # пока не превышено максимальное число эпох выполняем
        while self.K < self.n_iter:
            # обучаем в течение очередной эпохи
            self.fit_epoch(X, Y)
            # если выполнено условие остановки алгоритма, то заканчиваем обучение
            if self.stop_condition():
                break
            self.K += 1
    #
    #
    cdef object fit_epoch(self, double_2d X, double_1d Y):
        cdef int m = self.model.param.shape[0]
        cdef int i
        
        copy_memoryview(self.param_prev, self.model.param)
        
        # if self.regular is not None:
        #     self.regular.gradient(self.model.param, self.grad_r)
        #     for i in range(m):
        #         if self.regular is not None:
        #             self.grad_sum[i] += self.tau * self.grad_r[i]

        self.param_k1 = self.param_prev
        self.gradient(X, Y, self.grad_k1, self.param_k1)
        for i in range(m):
            self.param_k2[i] = self.param_prev[i] - 0.5 * self.h * self.grad_k1[i]
        self.gradient(X, Y, self.grad_k2, self.param_k2)
        for i in range(m):
            self.param_k3[i] = self.param_prev[i] - 0.5 * self.h * self.grad_k2[i]
        self.gradient(X, Y, self.grad_k3, self.param_k3)
        for i in range(m):
            self.param_k4[i] = self.param_prev[i] - self.h * self.grad_k3[i]
        self.gradient(X, Y, self.grad_k4, self.param_k4)

        for i in range(m):
            self.model.param[i] -= (self.h / 6.) * (
                  self.grad_k1[i] + 2.0*self.grad_k2[i] + 
                  2.0*self.grad_k3[i] + self.grad_k4[i])

    #
    # cdef double line_search(self, double_1d Xk, double yk, double_1d G):
    #     cdef double_1d param
    #     cdef int i, j
    #     cdef int m = len(self.model.param)
    #     cdef double h = self.h
    #     cdef double GN, lval0, lval, val
    #
    #     GN = 0.0
    #     for i in range(m):
    #         GN += G[i] * G[i]
    #
    #     if GN <= 0.00000001:
    #         return self.h
    #
    #     val = self.model.evaluate(Xk)
    #     lval0 = self.loss.evaluate(val, yk)
    #     param = self.model.param.copy()
    #
    #     for j in range(64):
    #         for i in range(m):
    #             self.model.param[i] = param[i] - h * G[i]
    #
    #         lval = self.loss.evaluate(self.model.evaluate(Xk), yk)
    #         if lval <= lval0 - 0.5*h*GN:
    #             self.model.param = param
    #             break
    #
    #         h *= 0.75
    #
    #     return h
    #
    # cdef bint stop_condition(self):
    #     cdef int i, m = self.model.param.shape[0]
    #     cdef double tol, p
    #
    #     for i in range(m):
    #         p = fabs(self.param_prev[i])
    #         tol = fabs(self.param_prev[i] - self.model.param[i]) / (1 + p)
    #         if tol > self.tol:
    #             self.success = 0
    #             return 0
    #
    #     self.success = 1
    #     return 1
