# coding: utf-8

import numpy as np

from mlgrad.miscfuncs cimport init_rand, rand, fill

cdef class SAG(GD):
    #
    def __init__(self, Model model, Loss loss, Regular regular=None, 
                 tau=1.0e-3, tol=1.0e-6, h=0.001, n_iter=1000, M=20):
        """ 
        """
        self.model = model
        self.loss = loss
        self.regular = regular
        self.tol = tol
        self.tau = tau
        self.n_iter = n_iter
        self.h = h
        
        self.grad = None
        self.grad_r = None
        self.grad_all = None
        self.grad_average = None
        self.param_best = None
        
        self.m = 0
        self.M = M
    #
    cdef object init(self, double[:,::1] X, double[::1] Y):
        GD.init(self, X, Y)
        
        # инициализируем счетчик случайных чисел
        init_rand()
        
        m = self.model.param.shape[0]
        N = Y.shape[0]
        
        if self.grad_all is None:
            self.grad_all = np.zeros((N, m), dtype='f')
        else:
            fill_memoryview2(self.grad_all, 0)
                    
        self.fill_tables(X, Y)
    #
    cdef fill_tables(self, double[:,::1] X, double[::1] Y):
        cdef int N = Y.shape[0]
        cdef int m = self.model.param.shape[0]
        cdef int i, j, k
        cdef double y, lval, yk, v, wk
        cdef double[::1] Xk, W

        W = self.weights.get_weights()

        for k in range(N):
            Xk = X[k]
            yk = Y[k]
                                    
            # вычисляем градиент по модели
            self.model.gradient(Xk, self.grad)
            # значение функции по модели
            y = self.model.evaluate(Xk)
            # значение функции потерь
            lval = self.loss.derivative(y, yk)

            wk = W[k]
            # добавляем в сумму градиентов
            for i in range(m):
                v = self.grad[i] * lval * wk
                self.grad_average[i] += v
                self.grad_all[k, i] = v
    #
    cdef fit_epoch(self, double[:,::1] X, double[::1] Y):
        cdef int N = Y.shape[0]
        cdef int j, k
        cdef double[::1] W

        W = self.weights.get_weights()
                
        for j in range(N):
            # очередной случайный индекс
            k = rand(N)
            self.fit_step_param(X[k], Y[k], k, W[k])
            
        self.loss_average(X, Y)
    #
    cdef fit_step_param(self, double[::1] Xk, double yk, int k, double wk):
        cdef int m = len(self.model.param)
        cdef int i
        cdef double y, lval, lval_dy
        cdef double v
        cdef double[::1] grad_average
        
        # вычисляем градиент по модели
        self.model.gradient(Xk, self.grad)
        # значение функции по модели
        y = self.model.evaluate(Xk)
        # производная функции потерь
        lval_dy = self.loss.derivative(y, yk)
        
        # обновляем параметры модели,
        # таблицу с градиентами и средний градиент
        for i in range(m):
            v = self.grad[i] * lval_dy * wk 

            self.grad_average[i] += v - self.grad_all[k, i]
            self.grad_all[k, i] = v

        # copy_memmoryview(self.param_prev, self.model.param)
    
        # if self.ls:
        #     if self.K <= self.M:
        #         g2 = 0
        #         for i in range(m):
        #             v = G[i]
        #             g2 += v * v
        #         if g2 > 1.0e-8:
        #             h = self.line_search(Xk, yk, G, g2, self.h_all[k])
        #             self.h_mean = 1./(1./self.h_mean + (1./h - 1./self.h_all[k]) / Nd)
        #             self.h_all[k] = h
        #         else:
        #             for i in range(m):
        #                 self.model.param[i] -= self.h_mean * self.grad_average[i]
        #                 if self.regular is not None:
        #                     self.model.param[i] -= self.h_mean * R[i]
        #     else:
        #         for i in range(m):
        #             self.model.param[i] -= self.h_mean *  self.grad_average[i]
        #             if self.regular is not None:
        #                 self.model.param[i] -= self.h_mean * R[i]
        #
        # else:
        
        # градиент регуляризации
        if self.regular is not None:
            self.regular.gradient(self.model.param, self.grad_r)
            for i in range(m):
                self.grad_average[i] += self.tau * self.grad_r[i]
        
        
        if self.grad_averager is not None:
            grad_average = self.grad_averager.update(self.grad_average)
        else:
            grad_average = self.grad_average
        
        for i in range(m):
            self.model.param[i] -= self.h * grad_average[i]
            if self.regular is not None:
                self.model.param[i] -= self.h * self.grad_r[i]

        # if self.check_flag:
        #     for i in range(m):
        #         p = self.model.param[i]
        #         self.param_mean[i] += p
        #         self.param2_mean[i] += p*p
    #
    # cdef double line_search(self, double_1d Xk, double yk, double_1d G, double GN, double h):
    #     cdef int i, j
    #     cdef int m = self.model.param.size
    #     cdef double Nd = Xk.shape[0]
    #     cdef double lval0, lval
    #     cdef double v, a=0.5, b=0.24
    #
    #     lval0 = self.loss.evaluate(self.model.evaluate(Xk), yk)
    #     if self.regular is not None:
    #         lval0 += self.tau * self.regular.evaluate(self.model.param)
    #
    #     for i in range(m):
    #         self.model.param[i] -= h * G[i]
    #
    #     for j in range(64):
    #
    #         lval = self.loss.evaluate(self.model.evaluate(Xk), yk)
    #         if self.regular is not None:
    #             lval += self.tau * self.regular.evaluate(self.model.param)
    #
    #         if lval <= lval0 - b * h * GN:
    #             return h
    #
    #         for i in range(m):
    #             self.model.param[i] += (1-a) * h * G[i]
    #
    #         h *= a
    #
    #     return h
    #
    # cdef bint check_condition(self):
    #     cdef int i, m = self.model.param.shape[0]
    #     cdef double p, tol
    #
    #     tol = 0
    #     # вычисляем максимальный элемент изменения вектора параметров модели
    #     for i in range(m):
    #         p = self.param_prev[i]
    #         tol = fabs(self.model.param[i]- p) / (1 + fabs(p))
    #         if tol > self.tol:
    #             return 0
    #
    #     return 1
    # #
    # cdef bint stop_condition(self):
    #     cdef int i, m = self.model.param.shape[0]
    #     cdef double p, tol
    #
    #     tol = 0
    #     # вычисляем максимальный элемент изменения вектора параметров модели
    #     for i in range(m):
    #         p = fabs(self.param_mean[i])
    #         tol = 0.5 * sqrt(self.param2_mean[i] - p*p) / (1 + p)
    #         if tol > self.tol:
    #             return 0
    #
    #     return 1